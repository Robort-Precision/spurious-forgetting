#!/usr/bin/env python3
"""Realignment mechanisms: recover "forgotten" capabilities without retraining.

Four strategies:
1. Activation Steering — add steering vectors to shift activations
2. Activation Patching — patch specific layer activations from base model
3. Adapter Heads — small trainable layers to realign output
4. Prompt Recovery — optimized prompts that re-activate dormant knowledge

Usage:
    python scripts/realign.py --base-model meta-llama/Llama-3.2-3B \
        --ft-model results/finetune/final --strategy activation_steering \
        --task mmlu --output results/realignment.json
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Realignment strategies")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--ft-model", type=str, required=True)
    p.add_argument("--strategy", type=str, required=True,
                   choices=["activation_steering", "activation_patching", "adapter_heads", "prompt_recovery", "all"])
    p.add_argument("--task", type=str, default="mmlu")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--steering-layers", type=str, default="auto", help="Layers to steer (comma-sep or 'auto')")
    p.add_argument("--steering-alpha", type=float, default=1.0, help="Steering vector magnitude")
    p.add_argument("--patch-layers", type=str, default="auto", help="Layers to patch from base model")
    p.add_argument("--output", type=str, default="results/realignment.json")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


# ── Shared utilities ──

def load_eval_data(task: str, num_samples: int):
    """Load evaluation data. Returns list of (prompt, choices, label)."""
    examples = []

    if task == "mmlu":
        subjects = ["abstract_algebra", "anatomy", "computer_security", "machine_learning", "world_religions"]
        per_subj = max(num_samples // len(subjects), 10)
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=42).select(range(min(per_subj, len(ds))))
                for ex in ds:
                    prompt = f"Question: {ex['question']}\nA) {ex['choices'][0]}\nB) {ex['choices'][1]}\nC) {ex['choices'][2]}\nD) {ex['choices'][3]}\nAnswer:"
                    choices = [" A", " B", " C", " D"]
                    examples.append((prompt, choices, ex["answer"]))
            except Exception:
                pass

    elif task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))
        for ex in ds:
            prompt = f"Question: {ex['question']}\nAnswer:"
            choices = ex["choices"]["text"]
            key = ex["answerKey"]
            label = int(key) - 1 if key.isdigit() else ord(key) - ord("A")
            examples.append((prompt, choices, label))

    return examples


def evaluate_accuracy(model, tokenizer, examples, device) -> float:
    """Evaluate multiple-choice accuracy."""
    correct = 0
    for prompt, choices, label in tqdm(examples, desc="Evaluating", leave=False):
        log_probs = []
        for choice in choices:
            full_text = prompt + choice
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            prompt_len = prompt_ids["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(**inputs)
            shift_logits = outputs.logits[:, prompt_len - 1:-1, :]
            shift_labels = inputs["input_ids"][:, prompt_len:]
            lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            avg_lp = token_lp.sum().item() / max(shift_labels.shape[1], 1)
            log_probs.append(avg_lp)

        if int(np.argmax(log_probs)) == label:
            correct += 1

    return correct / len(examples)


# ── Strategy 1: Activation Steering ──

def compute_steering_vectors(base_model, ft_model, tokenizer, examples, layers, device):
    """Compute steering vectors as the difference in mean activations between base and FT models.
    
    steering_vector[l] = mean(base_hidden[l]) - mean(ft_hidden[l])
    Adding this to FT model pushes activations back toward base behavior.
    """
    vectors = {}
    sample_texts = [ex[0] for ex in examples[:50]]  # Use subset for computing vectors

    for model_tag, model in [("base", base_model), ("ft", ft_model)]:
        model.eval()
        layer_sums = {}
        count = 0

        for text in tqdm(sample_texts, desc=f"Computing activations ({model_tag})"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            for l in layers:
                hs = outputs.hidden_states[l].mean(dim=1).squeeze(0)  # (hidden_dim,)
                if l not in layer_sums:
                    layer_sums[l] = torch.zeros_like(hs)
                layer_sums[l] += hs
            count += 1

        if model_tag == "base":
            base_means = {l: s / count for l, s in layer_sums.items()}
        else:
            ft_means = {l: s / count for l, s in layer_sums.items()}

    for l in layers:
        vectors[l] = base_means[l] - ft_means[l]

    return vectors


class SteeringHook:
    """Forward hook that adds a steering vector to layer activations."""
    def __init__(self, vector, alpha=1.0):
        self.vector = vector
        self.alpha = alpha

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.vector.to(hidden.device)
            return (hidden,) + output[1:]
        return output + self.alpha * self.vector.to(output.device)


def run_activation_steering(base_model, ft_model, tokenizer, examples, args, device):
    """Apply steering vectors to FT model and measure recovery."""
    num_layers = ft_model.config.num_hidden_layers

    if args.steering_layers == "auto":
        # Steer middle and late layers (most relevant for task performance)
        layers = list(range(num_layers // 3, num_layers))
    else:
        layers = [int(x) for x in args.steering_layers.split(",")]

    print(f"[steering] Computing steering vectors for layers {layers}...")
    vectors = compute_steering_vectors(base_model, ft_model, tokenizer, examples, layers, device)

    # Evaluate FT model without steering
    print("[steering] Evaluating FT model (no steering)...")
    ft_acc = evaluate_accuracy(ft_model, tokenizer, examples, device)

    # Apply steering hooks and evaluate
    results = {"ft_accuracy": ft_acc, "alphas": {}}

    for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        hooks = []
        for l, vec in vectors.items():
            # Access the transformer layer
            if hasattr(ft_model, "model") and hasattr(ft_model.model, "layers"):
                layer_module = ft_model.model.layers[l - 1]  # -1 because hidden_states[0] is embeddings
            elif hasattr(ft_model, "transformer") and hasattr(ft_model.transformer, "h"):
                layer_module = ft_model.transformer.h[l - 1]
            else:
                continue
            hook = layer_module.register_forward_hook(SteeringHook(vec, alpha))
            hooks.append(hook)

        print(f"[steering] Evaluating with alpha={alpha}...")
        steered_acc = evaluate_accuracy(ft_model, tokenizer, examples, device)
        results["alphas"][str(alpha)] = {
            "accuracy": steered_acc,
            "recovery": steered_acc - ft_acc,
        }
        print(f"[steering] alpha={alpha}: {steered_acc:.4f} (recovery: {steered_acc - ft_acc:+.4f})")

        for h in hooks:
            h.remove()

    best_alpha = max(results["alphas"], key=lambda a: results["alphas"][a]["accuracy"])
    results["best_alpha"] = float(best_alpha)
    results["best_accuracy"] = results["alphas"][best_alpha]["accuracy"]

    return results


# ── Strategy 2: Activation Patching ──

def run_activation_patching(base_model, ft_model, tokenizer, examples, args, device):
    """Patch activations from base model into FT model at specific layers.
    
    For each layer l, replace FT hidden states with base hidden states
    and measure task accuracy. Layers where patching helps most are
    where misalignment occurs.
    """
    num_layers = ft_model.config.num_hidden_layers

    print("[patching] Evaluating FT model baseline...")
    ft_acc = evaluate_accuracy(ft_model, tokenizer, examples, device)

    results = {"ft_accuracy": ft_acc, "layers": {}}

    # Test patching each layer individually
    test_layers = list(range(0, num_layers, max(1, num_layers // 16)))  # Sample ~16 layers

    for patch_layer in tqdm(test_layers, desc="Patching layers"):
        class PatchHook:
            def __init__(self):
                self.base_hidden = None

            def capture(self, module, input, output):
                if isinstance(output, tuple):
                    self.base_hidden = output[0].clone()
                else:
                    self.base_hidden = output.clone()
                return output

            def inject(self, module, input, output):
                if self.base_hidden is not None:
                    if isinstance(output, tuple):
                        return (self.base_hidden.to(output[0].device),) + output[1:]
                    return self.base_hidden.to(output.device)
                return output

        hook_helper = PatchHook()

        # Get layer modules
        def get_layer(model, idx):
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return model.model.layers[idx]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                return model.transformer.h[idx]
            return None

        base_layer = get_layer(base_model, patch_layer)
        ft_layer = get_layer(ft_model, patch_layer)

        if base_layer is None or ft_layer is None:
            continue

        correct = 0
        for prompt, choices, label in examples:
            # Run base model to capture activations
            h1 = base_layer.register_forward_hook(hook_helper.capture)
            base_inputs = tokenizer(prompt + choices[0], return_tensors="pt", truncation=True, max_length=1024)
            base_inputs = {k: v.to(device) for k, v in base_inputs.items()}
            with torch.no_grad():
                base_model(**base_inputs)
            h1.remove()

            # Run FT model with patched activations
            log_probs = []
            for choice in choices:
                # Capture base hidden for this input
                full_text = prompt + choice
                inputs_base = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
                inputs_base = {k: v.to(device) for k, v in inputs_base.items()}
                h_cap = base_layer.register_forward_hook(hook_helper.capture)
                with torch.no_grad():
                    base_model(**inputs_base)
                h_cap.remove()

                # Inject into FT model
                h_inj = ft_layer.register_forward_hook(hook_helper.inject)
                inputs_ft = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
                inputs_ft = {k: v.to(device) for k, v in inputs_ft.items()}
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                prompt_len = prompt_ids["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = ft_model(**inputs_ft)
                h_inj.remove()

                shift_logits = outputs.logits[:, prompt_len - 1:-1, :]
                shift_labels = inputs_ft["input_ids"][:, prompt_len:]
                lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                log_probs.append(token_lp.sum().item() / max(shift_labels.shape[1], 1))

            if int(np.argmax(log_probs)) == label:
                correct += 1

        patched_acc = correct / len(examples)
        results["layers"][str(patch_layer)] = {
            "accuracy": patched_acc,
            "recovery": patched_acc - ft_acc,
        }
        print(f"[patching] Layer {patch_layer}: {patched_acc:.4f} (recovery: {patched_acc - ft_acc:+.4f})")

    # Find best layer
    if results["layers"]:
        best_layer = max(results["layers"], key=lambda l: results["layers"][l]["accuracy"])
        results["best_layer"] = int(best_layer)
        results["best_accuracy"] = results["layers"][best_layer]["accuracy"]

    return results


# ── Strategy 3: Adapter Heads ──

class AdapterHead(nn.Module):
    """Small MLP adapter that transforms FT hidden states to realign with base model output head."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return self.layer_norm(x + residual)


def run_adapter_heads(base_model, ft_model, tokenizer, examples, args, device):
    """Train a small adapter to realign FT model's final hidden states."""

    print("[adapter] Collecting training data from base and FT models...")
    train_texts = [ex[0] for ex in examples[:100]]
    val_examples = examples[100:]

    # Collect base model's final hidden states as targets
    base_hidden = []
    ft_hidden = []

    for text in tqdm(train_texts, desc="Collecting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            base_out = base_model(**inputs, output_hidden_states=True)
            ft_out = ft_model(**inputs, output_hidden_states=True)

        # Last hidden state, mean pooled
        base_hidden.append(base_out.hidden_states[-1].mean(dim=1).squeeze(0))
        ft_hidden.append(ft_out.hidden_states[-1].mean(dim=1).squeeze(0))

    base_hidden = torch.stack(base_hidden)  # (N, hidden_dim)
    ft_hidden = torch.stack(ft_hidden)

    # Train adapter
    hidden_dim = ft_hidden.shape[1]
    adapter = AdapterHead(hidden_dim, bottleneck_dim=min(256, hidden_dim // 4)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    print("[adapter] Training adapter head...")
    adapter.train()
    for epoch in range(100):
        adapted = adapter(ft_hidden)
        loss = nn.functional.mse_loss(adapted, base_hidden)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: loss={loss.item():.6f}")

    # Evaluate: hook adapter into FT model's final layer
    adapter.eval()

    class AdapterHook:
        def __init__(self, adapter):
            self.adapter = adapter
        def __call__(self, module, input, output):
            if isinstance(output, tuple):
                return (self.adapter(output[0]),) + output[1:]
            return self.adapter(output)

    # Install hook on last transformer layer
    if hasattr(ft_model, "model") and hasattr(ft_model.model, "layers"):
        last_layer = ft_model.model.layers[-1]
    elif hasattr(ft_model, "transformer") and hasattr(ft_model.transformer, "h"):
        last_layer = ft_model.transformer.h[-1]
    else:
        return {"error": "Cannot find transformer layers"}

    hook = last_layer.register_forward_hook(AdapterHook(adapter))

    print("[adapter] Evaluating with adapter...")
    adapted_acc = evaluate_accuracy(ft_model, tokenizer, val_examples, device)
    hook.remove()

    print("[adapter] Evaluating FT baseline...")
    ft_acc = evaluate_accuracy(ft_model, tokenizer, val_examples, device)

    results = {
        "ft_accuracy": ft_acc,
        "adapted_accuracy": adapted_acc,
        "recovery": adapted_acc - ft_acc,
        "adapter_params": sum(p.numel() for p in adapter.parameters()),
        "final_train_loss": loss.item(),
    }
    print(f"[adapter] FT: {ft_acc:.4f} → Adapted: {adapted_acc:.4f} (recovery: {adapted_acc - ft_acc:+.4f})")

    return results


# ── Strategy 4: Prompt Recovery ──

def run_prompt_recovery(base_model, ft_model, tokenizer, examples, args, device):
    """Find prompts that help FT model recover base model performance.
    
    Tests various prompt prefixes that might re-activate dormant knowledge.
    """
    print("[prompt] Evaluating FT model with different prompts...")

    recovery_prompts = [
        "",  # No prefix (baseline)
        "You are a knowledgeable AI assistant. Answer the following question accurately.\n\n",
        "Think step by step and use your full knowledge to answer.\n\n",
        "Important: Use all your knowledge, not just recent training. Answer:\n\n",
        "As a general knowledge expert, answer the following:\n\n",
        "Recall your pre-training knowledge carefully. ",
        "Before answering, consider what you know from your broad training data.\n\n",
        "[SYSTEM] Access full knowledge base. Respond with maximum accuracy.\n\n",
        "The following is a test of general knowledge. Answer to the best of your ability.\n\n",
    ]

    results = {"prompts": {}}

    for i, prefix in enumerate(recovery_prompts):
        modified_examples = [(prefix + prompt, choices, label) for prompt, choices, label in examples]
        acc = evaluate_accuracy(ft_model, tokenizer, modified_examples, device)
        prompt_label = f"prompt_{i}" if prefix else "baseline"
        results["prompts"][prompt_label] = {
            "prefix": prefix[:80] + "..." if len(prefix) > 80 else prefix,
            "accuracy": acc,
        }
        print(f"[prompt] {prompt_label}: {acc:.4f}")

    baseline = results["prompts"]["baseline"]["accuracy"]
    best_prompt = max(results["prompts"], key=lambda p: results["prompts"][p]["accuracy"])
    results["baseline_accuracy"] = baseline
    results["best_prompt"] = best_prompt
    results["best_accuracy"] = results["prompts"][best_prompt]["accuracy"]
    results["best_recovery"] = results["prompts"][best_prompt]["accuracy"] - baseline

    return results


# ── Main ──

STRATEGY_REGISTRY = {
    "activation_steering": run_activation_steering,
    "activation_patching": run_activation_patching,
    "adapter_heads": run_adapter_heads,
    "prompt_recovery": run_prompt_recovery,
}


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"[realign] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    base_model.eval()

    print(f"[realign] Loading FT model: {args.ft_model}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    ft_model.eval()

    device = next(ft_model.parameters()).device

    print(f"[realign] Loading eval data: {args.task}")
    examples = load_eval_data(args.task, args.num_samples)
    print(f"[realign] {len(examples)} examples loaded")

    # Evaluate base model for reference
    print("[realign] Evaluating base model...")
    base_acc = evaluate_accuracy(base_model, tokenizer, examples, device)
    print(f"[realign] Base accuracy: {base_acc:.4f}")

    strategies = list(STRATEGY_REGISTRY.keys()) if args.strategy == "all" else [args.strategy]

    all_results = {
        "base_model": args.base_model,
        "ft_model": args.ft_model,
        "task": args.task,
        "base_accuracy": base_acc,
        "strategies": {},
    }

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy}")
        print(f"{'=' * 60}")

        fn = STRATEGY_REGISTRY[strategy]
        result = fn(base_model, ft_model, tokenizer, examples, args, device)
        all_results["strategies"][strategy] = result

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[realign] All results saved to {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print("REALIGNMENT SUMMARY")
    print(f"Base accuracy: {base_acc:.4f}")
    for s, r in all_results["strategies"].items():
        best = r.get("best_accuracy", r.get("adapted_accuracy", "N/A"))
        print(f"  {s}: best recovered accuracy = {best}")


if __name__ == "__main__":
    main()
