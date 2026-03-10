#!/usr/bin/env python3
"""Layer Grafting: replace top N layers of fine-tuned model with base model layers.

Finds the crossover point where Task A performance recovers while Task B is maintained.
Directly tests whether forgetting is localized to specific layers.

Usage:
    python scripts/layer_grafting.py --base-model mistralai/Mistral-7B-v0.3 \
        --ft-model results/finetune/final --task-a mmlu --task-b medqa \
        --output results/grafting.json
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Layer grafting experiment")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--ft-model", type=str, required=True)
    p.add_argument("--task-a", type=str, default="mmlu", help="Task to recover (forgotten)")
    p.add_argument("--task-b", type=str, default="medqa", help="Task to retain (fine-tuned on)")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output", type=str, default="results/grafting.json")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_eval_data(task: str, num_samples: int, seed: int):
    """Load eval data as (prompt, choices, label) tuples."""
    examples = []

    if task == "mmlu":
        subjects = ["abstract_algebra", "anatomy", "computer_security", "machine_learning", "world_religions"]
        per_subj = max(num_samples // len(subjects), 10)
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=seed).select(range(min(per_subj, len(ds))))
                for ex in ds:
                    prompt = (f"Question: {ex['question']}\nA) {ex['choices'][0]}\n"
                              f"B) {ex['choices'][1]}\nC) {ex['choices'][2]}\nD) {ex['choices'][3]}\nAnswer:")
                    examples.append((prompt, [" A", " B", " C", " D"], ex["answer"]))
            except Exception:
                pass

    elif task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        for ex in ds:
            prompt = f"Question: {ex['question']}\nAnswer:"
            choices = ex["choices"]["text"]
            key = ex["answerKey"]
            label = int(key) - 1 if key.isdigit() else ord(key) - ord("A")
            examples.append((prompt, choices, label))

    elif task in ("medqa", "gsm8k"):
        if task == "medqa":
            ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test", trust_remote_code=True)
            ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
            for ex in ds:
                prompt = f"Medical question: {ex['question']}\nAnswer:"
                options = ex.get("options", {})
                if isinstance(options, dict):
                    choices = list(options.values())
                    label = list(options.keys()).index(ex.get("answer_idx", "A"))
                else:
                    choices = [str(o) for o in options]
                    label = 0
                examples.append((prompt, choices, label))
        else:
            ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
            ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
            for ex in ds:
                prompt = f"Question: {ex['question']}\nAnswer:"
                # For GSM8K, just check if final answer matches
                examples.append((prompt, [ex["answer"]], 0))

    return examples[:num_samples]


def evaluate_mc(model, tokenizer, examples, device) -> float:
    """Multiple-choice evaluation by log-likelihood."""
    correct = 0
    for prompt, choices, label in tqdm(examples, desc="Eval", leave=False):
        log_probs = []
        for choice in choices:
            full = prompt + choice
            inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            p_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            plen = p_ids["input_ids"].shape[1]

            with torch.no_grad():
                out = model(**inputs)
            sl = out.logits[:, plen - 1:-1, :]
            sl_labels = inputs["input_ids"][:, plen:]
            lp = torch.nn.functional.log_softmax(sl, dim=-1)
            tlp = lp.gather(2, sl_labels.unsqueeze(-1)).squeeze(-1)
            log_probs.append(tlp.sum().item() / max(sl_labels.shape[1], 1))

        if int(np.argmax(log_probs)) == label:
            correct += 1

    return correct / len(examples) if examples else 0


def get_layers(model):
    """Get the list of transformer layer modules."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot find transformer layers")


def graft_layers(ft_model, base_model, graft_from: int):
    """Replace layers [graft_from:] in ft_model with base_model layers.
    
    Returns a new model (deep copy of ft_model with grafted layers).
    """
    grafted = copy.deepcopy(ft_model)
    
    ft_layers = get_layers(grafted)
    base_layers = get_layers(base_model)
    
    num_layers = len(ft_layers)
    for i in range(graft_from, num_layers):
        ft_layers[i].load_state_dict(base_layers[i].state_dict())

    # Also graft the final layer norm and lm_head
    if hasattr(grafted, "model") and hasattr(grafted.model, "norm"):
        grafted.model.norm.load_state_dict(base_model.model.norm.state_dict())
    if hasattr(grafted, "lm_head"):
        grafted.lm_head.load_state_dict(base_model.lm_head.state_dict())

    return grafted


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval data
    task_a_data = load_eval_data(args.task_a, args.num_samples, args.seed)
    task_b_data = load_eval_data(args.task_b, args.num_samples, args.seed)
    print(f"[grafting] Task A ({args.task_a}): {len(task_a_data)} examples")
    print(f"[grafting] Task B ({args.task_b}): {len(task_b_data)} examples")

    # Load models
    print(f"[grafting] Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True, device_map="cpu"
    )
    base_model.eval()

    print(f"[grafting] Loading FT model: {args.ft_model}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    ft_model.eval()
    device = next(ft_model.parameters()).device

    num_layers = len(get_layers(ft_model))
    print(f"[grafting] Model has {num_layers} layers")

    # Baseline: FT model on both tasks
    print("[grafting] FT model baselines...")
    ft_task_a = evaluate_mc(ft_model, tokenizer, task_a_data, device)
    ft_task_b = evaluate_mc(ft_model, tokenizer, task_b_data, device)
    print(f"[grafting] FT baseline — Task A: {ft_task_a:.4f}, Task B: {ft_task_b:.4f}")

    results = {
        "base_model": args.base_model,
        "ft_model": args.ft_model,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "num_layers": num_layers,
        "ft_task_a": ft_task_a,
        "ft_task_b": ft_task_b,
        "grafts": {},
    }

    # Test grafting from different points
    # Sample ~8 graft points across the model
    graft_points = sorted(set([
        num_layers // 4,
        num_layers // 3,
        num_layers // 2,
        num_layers * 2 // 3,
        num_layers * 3 // 4,
        num_layers - 4,
        num_layers - 2,
        num_layers - 1,
    ]))

    for graft_from in graft_points:
        if graft_from >= num_layers or graft_from < 1:
            continue

        print(f"\n[grafting] Grafting base layers [{graft_from}:{num_layers}] into FT model...")
        grafted = graft_layers(ft_model, base_model, graft_from)
        grafted = grafted.to(device)
        grafted.eval()

        task_a_acc = evaluate_mc(grafted, tokenizer, task_a_data, device)
        task_b_acc = evaluate_mc(grafted, tokenizer, task_b_data, device)

        results["grafts"][str(graft_from)] = {
            "layers_replaced": num_layers - graft_from,
            "task_a_accuracy": task_a_acc,
            "task_b_accuracy": task_b_acc,
            "task_a_recovery": task_a_acc - ft_task_a,
            "task_b_retention": task_b_acc - ft_task_b,
        }

        print(f"[grafting] Graft@{graft_from}: Task A={task_a_acc:.4f} ({task_a_acc - ft_task_a:+.4f}), "
              f"Task B={task_b_acc:.4f} ({task_b_acc - ft_task_b:+.4f})")

        del grafted
        torch.cuda.empty_cache()

    # Find optimal graft point (maximize Task A recovery while retaining >90% Task B)
    viable = {k: v for k, v in results["grafts"].items()
              if v["task_b_accuracy"] >= ft_task_b * 0.9}

    if viable:
        best = max(viable, key=lambda k: viable[k]["task_a_accuracy"])
        results["optimal_graft_point"] = int(best)
        results["optimal_task_a"] = viable[best]["task_a_accuracy"]
        results["optimal_task_b"] = viable[best]["task_b_accuracy"]
        print(f"\n[grafting] OPTIMAL: Graft from layer {best} — "
              f"Task A: {viable[best]['task_a_accuracy']:.4f}, Task B: {viable[best]['task_b_accuracy']:.4f}")
    else:
        print("\n[grafting] No graft point retains >90% Task B performance")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[grafting] Results saved to {args.output}")


if __name__ == "__main__":
    main()
