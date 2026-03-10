#!/usr/bin/env python3
"""CKA (Centered Kernel Alignment) between base and fine-tuned model representations.

Measures representational similarity at each layer to identify where
fine-tuning changes representations vs where they stay intact.

Usage:
    python eval/cka.py --base-model mistralai/Mistral-7B-v0.3 \
        --ft-model results/finetune/final --task mmlu --num-samples 100 \
        --output results/cka.json
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
    p = argparse.ArgumentParser(description="CKA similarity between two models")
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--ft-model", type=str, required=True)
    p.add_argument("--task", type=str, default="mmlu")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output", type=str, default="results/cka.json")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two sets of representations.
    
    X, Y: (n_samples, n_features) arrays
    Returns: scalar CKA similarity in [0, 1]
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices (via dot products for efficiency)
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X

    numerator = np.linalg.norm(YTX, "fro") ** 2
    denominator = np.linalg.norm(XTX, "fro") * np.linalg.norm(YTY, "fro")

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def load_texts(task: str, num_samples: int, seed: int) -> list[str]:
    """Load text inputs for CKA computation."""
    if task == "mmlu":
        subjects = ["abstract_algebra", "anatomy", "computer_security", "machine_learning", "world_religions"]
        texts = []
        per_subj = max(num_samples // len(subjects), 10)
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=seed).select(range(min(per_subj, len(ds))))
                for ex in ds:
                    texts.append(f"Question: {ex['question']}\nA) {ex['choices'][0]}\nB) {ex['choices'][1]}\nC) {ex['choices'][2]}\nD) {ex['choices'][3]}")
            except Exception:
                pass
        return texts[:num_samples]

    elif task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        return [f"Question: {ex['question']}" for ex in ds]

    elif task == "triviaqa":
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        return [f"Question: {ex['question']}" for ex in ds]

    return []


def extract_all_layers(model, tokenizer, texts: list[str], device) -> dict[int, np.ndarray]:
    """Extract mean-pooled hidden states from all layers."""
    model.eval()
    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings
    layer_activations = {l: [] for l in range(num_layers)}

    for text in tqdm(texts, desc="Extracting"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for l in range(num_layers):
            hs = outputs.hidden_states[l].mean(dim=1).squeeze(0).cpu().float().numpy()
            layer_activations[l].append(hs)

    return {l: np.stack(vecs) for l, vecs in layer_activations.items()}


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    texts = load_texts(args.task, args.num_samples, args.seed)
    print(f"[cka] {len(texts)} texts loaded for task '{args.task}'")

    results = {"task": args.task, "num_samples": len(texts), "layers": {}}

    # Process models sequentially to save VRAM
    for model_tag, model_name in [("base", args.base_model), ("ft", args.ft_model)]:
        print(f"\n[cka] Loading {model_tag}: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        )
        device = next(model.parameters()).device

        activations = extract_all_layers(model, tokenizer, texts, device)

        if model_tag == "base":
            base_activations = activations
        else:
            ft_activations = activations

        del model
        torch.cuda.empty_cache()

    # Compute CKA at each layer
    print("\n[cka] Computing CKA similarities...")
    for l in sorted(base_activations.keys()):
        if l in ft_activations:
            cka_val = linear_cka(base_activations[l], ft_activations[l])
            results["layers"][str(l)] = {
                "cka": cka_val,
                "high_similarity": cka_val > 0.9,
            }
            print(f"  Layer {l}: CKA = {cka_val:.4f} {'✓' if cka_val > 0.9 else '⚠'}")

    # Summary
    cka_values = [v["cka"] for v in results["layers"].values()]
    results["mean_cka"] = float(np.mean(cka_values))
    results["min_cka"] = float(np.min(cka_values))
    results["max_cka"] = float(np.max(cka_values))

    # Find the "divergence point" — where CKA drops most
    layers_sorted = sorted(results["layers"].keys(), key=int)
    if len(layers_sorted) > 1:
        min_layer = min(layers_sorted, key=lambda l: results["layers"][l]["cka"])
        results["most_changed_layer"] = int(min_layer)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[cka] Mean CKA: {results['mean_cka']:.4f}")
    print(f"[cka] Most changed layer: {results.get('most_changed_layer', 'N/A')}")
    print(f"[cka] Results saved to {args.output}")


if __name__ == "__main__":
    main()
