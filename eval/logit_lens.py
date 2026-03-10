#!/usr/bin/env python3
"""Logit lens: project intermediate layer hidden states through the unembedding matrix.

Tests whether correct answers appear in earlier layers' predictions but get
overridden in later layers — direct evidence of misalignment vs knowledge loss.

Usage:
    python eval/logit_lens.py --model results/finetune/final \
        --base-model mistralai/Mistral-7B-v0.3 --task mmlu \
        --num-samples 50 --output results/logit_lens.json
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Logit lens analysis")
    p.add_argument("--model", type=str, required=True, help="Model to analyze (typically FT model)")
    p.add_argument("--base-model", type=str, default=None, help="Base model for tokenizer (if FT model path)")
    p.add_argument("--task", type=str, default="mmlu")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--output", type=str, default="results/logit_lens.json")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_mc_data(task: str, num_samples: int, seed: int):
    """Load multiple-choice data. Returns list of (prompt, answer_tokens, label)."""
    examples = []

    if task == "mmlu":
        subjects = ["abstract_algebra", "anatomy", "computer_security", "machine_learning", "world_religions"]
        per_subj = max(num_samples // len(subjects), 5)
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=seed).select(range(min(per_subj, len(ds))))
                for ex in ds:
                    prompt = (f"Question: {ex['question']}\n"
                              f"A) {ex['choices'][0]}\nB) {ex['choices'][1]}\n"
                              f"C) {ex['choices'][2]}\nD) {ex['choices'][3]}\nAnswer:")
                    answer_options = [" A", " B", " C", " D"]
                    examples.append((prompt, answer_options, ex["answer"]))
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

    return examples[:num_samples]


def logit_lens_analysis(model, tokenizer, examples, device):
    """Run logit lens: at each layer, project hidden states through unembedding and check predictions."""
    model.eval()
    num_layers = model.config.num_hidden_layers + 1

    # Get the unembedding matrix (lm_head)
    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight.data  # (vocab_size, hidden_dim)
    else:
        raise ValueError("Cannot find lm_head for logit lens")

    # Optional: layer norm before unembedding
    final_ln = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_ln = model.transformer.ln_f

    results = {
        "per_layer_correct": {str(l): 0 for l in range(num_layers)},
        "per_layer_rank": {str(l): [] for l in range(num_layers)},
        "total": 0,
        "examples": [],
    }

    for prompt, answer_options, label in tqdm(examples, desc="Logit lens"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get token IDs for answer options
        answer_token_ids = []
        for opt in answer_options:
            toks = tokenizer.encode(opt, add_special_tokens=False)
            answer_token_ids.append(toks[0] if toks else 0)

        correct_token = answer_token_ids[label] if label < len(answer_token_ids) else answer_token_ids[0]

        example_info = {
            "prompt": prompt[:100] + "...",
            "label": label,
            "layer_predictions": {},
        }

        # At each layer, project the last token's hidden state through unembedding
        for l in range(num_layers):
            hs = outputs.hidden_states[l][:, -1, :]  # (1, hidden_dim) — last token

            # Apply final layer norm if available
            if final_ln is not None:
                hs = final_ln(hs)

            # Project through unembedding: logits = hs @ unembed.T
            logits = hs @ unembed.T  # (1, vocab_size)
            logits = logits.squeeze(0)

            # Check if correct answer is the top prediction among answer options
            answer_logits = [logits[tid].item() for tid in answer_token_ids]
            predicted = int(np.argmax(answer_logits))

            if predicted == label:
                results["per_layer_correct"][str(l)] += 1

            # Rank of correct token in full vocabulary
            rank = (logits > logits[correct_token]).sum().item()
            results["per_layer_rank"][str(l)].append(int(rank))

            example_info["layer_predictions"][str(l)] = {
                "predicted": predicted,
                "correct": predicted == label,
                "correct_rank": int(rank),
            }

        results["total"] += 1
        results["examples"].append(example_info)

    # Compute accuracies and mean ranks
    results["per_layer_accuracy"] = {}
    results["per_layer_mean_rank"] = {}
    for l in range(num_layers):
        ls = str(l)
        results["per_layer_accuracy"][ls] = results["per_layer_correct"][ls] / max(results["total"], 1)
        results["per_layer_mean_rank"][ls] = float(np.mean(results["per_layer_rank"][ls])) if results["per_layer_rank"][ls] else 0

    return results


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    tokenizer_name = args.base_model or args.model
    print(f"[logit_lens] Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[logit_lens] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    device = next(model.parameters()).device

    examples = load_mc_data(args.task, args.num_samples, args.seed)
    print(f"[logit_lens] Loaded {len(examples)} examples")

    results = logit_lens_analysis(model, tokenizer, examples, device)

    # Summary: find where correct answer first appears
    print("\n[logit_lens] Layer-by-layer accuracy (correct answer in top prediction):")
    first_good_layer = None
    last_good_layer = None
    for l in sorted(results["per_layer_accuracy"].keys(), key=int):
        acc = results["per_layer_accuracy"][l]
        rank = results["per_layer_mean_rank"][l]
        marker = ""
        if acc > 0.3 and first_good_layer is None:
            first_good_layer = int(l)
            marker = " ← knowledge emerges"
        print(f"  Layer {l:>2}: acc={acc:.3f}, mean_rank={rank:.0f}{marker}")

    # Find if later layers degrade (evidence of misalignment)
    accs = [(int(l), results["per_layer_accuracy"][l]) for l in results["per_layer_accuracy"]]
    accs.sort()
    if len(accs) > 4:
        mid_acc = np.mean([a for _, a in accs[len(accs)//3:2*len(accs)//3]])
        late_acc = np.mean([a for _, a in accs[-3:]])
        if mid_acc > late_acc + 0.05:
            print(f"\n>>> EVIDENCE OF MISALIGNMENT: Middle layers ({mid_acc:.3f}) outperform late layers ({late_acc:.3f})")
            print(">>> Knowledge exists in representations but gets overridden at output layers")
            results["misalignment_detected"] = True
            results["mid_layer_accuracy"] = float(mid_acc)
            results["late_layer_accuracy"] = float(late_acc)

    # Remove verbose per-example data for clean output
    results_clean = {k: v for k, v in results.items() if k != "examples"}
    results_clean["num_examples_analyzed"] = len(results["examples"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results_clean, f, indent=2)

    # Also save full results with examples
    full_path = args.output.replace(".json", "_full.json")
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[logit_lens] Results saved to {args.output}")


if __name__ == "__main__":
    main()
