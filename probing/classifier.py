#!/usr/bin/env python3
"""Probing classifier: test whether knowledge persists in hidden states.

The key experiment: if a fine-tuned model performs worse on a benchmark,
but a linear probe on its hidden states can STILL classify correctly,
then the knowledge wasn't lost — it was misaligned.

Usage:
    python probing/classifier.py --base-model meta-llama/Llama-3.2-3B \
        --ft-model results/finetune/final --task mmlu --layers all \
        --output results/probing.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Probing classifier for hidden state analysis")
    p.add_argument("--base-model", type=str, required=True, help="Base (pre-FT) model")
    p.add_argument("--ft-model", type=str, required=True, help="Fine-tuned model path")
    p.add_argument("--task", type=str, default="mmlu", choices=["mmlu", "arc_challenge", "hellaswag"])
    p.add_argument("--layers", type=str, default="all", help="'all' or comma-separated layer indices")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--output", type=str, default="results/probing.json")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def extract_hidden_states(model, tokenizer, texts: list[str], layers: list[int], device) -> dict[int, np.ndarray]:
    """Extract hidden states from specified layers for all texts.
    
    Returns dict mapping layer_idx -> (num_texts, hidden_dim) array.
    Uses mean pooling over sequence length.
    """
    model.eval()
    hidden_states_by_layer = {l: [] for l in layers}

    for text in tqdm(texts, desc="Extracting hidden states"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Each tensor: (batch=1, seq_len, hidden_dim)
        for layer_idx in layers:
            if layer_idx < len(outputs.hidden_states):
                hs = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
                # Mean pool over sequence
                pooled = hs.mean(dim=1).squeeze(0).cpu().float().numpy()
                hidden_states_by_layer[layer_idx].append(pooled)

    return {l: np.stack(vecs) for l, vecs in hidden_states_by_layer.items() if vecs}


def load_task_data(task: str, num_samples: int) -> tuple[list[str], np.ndarray]:
    """Load task data and return (texts, labels)."""

    if task == "mmlu":
        subjects = ["abstract_algebra", "anatomy", "computer_security", "machine_learning", "world_religions"]
        texts = []
        labels = []
        per_subj = max(num_samples // len(subjects), 20)

        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=42).select(range(min(per_subj, len(ds))))
                for ex in ds:
                    prompt = f"Question: {ex['question']}\nA) {ex['choices'][0]}\nB) {ex['choices'][1]}\nC) {ex['choices'][2]}\nD) {ex['choices'][3]}"
                    texts.append(prompt)
                    labels.append(ex["answer"])
            except Exception as e:
                print(f"  Skipping {subj}: {e}")

        return texts, np.array(labels)

    elif task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

        texts = []
        labels = []
        for ex in ds:
            prompt = f"Question: {ex['question']}\nChoices: {', '.join(ex['choices']['text'])}"
            texts.append(prompt)
            key = ex["answerKey"]
            label = int(key) - 1 if key.isdigit() else ord(key) - ord("A")
            labels.append(label)

        return texts, np.array(labels)

    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

        texts = [ex["ctx"] for ex in ds]
        labels = np.array([int(ex["label"]) for ex in ds])
        return texts, labels


def run_probing(
    base_model_name: str,
    ft_model_name: str,
    task: str,
    layers: list[int],
    num_samples: int,
    bf16: bool,
) -> dict:
    """Run the full probing experiment.
    
    For each model (base, fine-tuned):
      1. Extract hidden states at each layer
      2. Train logistic regression probe to predict correct answer
      3. Compare probe accuracy between base and fine-tuned
    
    If ft probe accuracy ≈ base probe accuracy, knowledge is preserved.
    """

    print(f"[probing] Task: {task}, Samples: {num_samples}")
    texts, labels = load_task_data(task, num_samples)
    print(f"[probing] Loaded {len(texts)} examples, {len(set(labels))} classes")

    results = {"task": task, "num_samples": len(texts), "layers": {}}
    dtype = torch.bfloat16 if bf16 else torch.float32

    for model_tag, model_name in [("base", base_model_name), ("finetuned", ft_model_name)]:
        print(f"\n[probing] Loading {model_tag} model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name if model_tag == "base" else base_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        )
        device = next(model.parameters()).device

        # Determine layer count
        if not layers or layers == [-1]:
            num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
            layer_indices = list(range(num_layers))
        else:
            layer_indices = layers

        print(f"[probing] Extracting hidden states from {len(layer_indices)} layers...")
        hidden_states = extract_hidden_states(model, tokenizer, texts, layer_indices, device)

        # Free model memory
        del model
        torch.cuda.empty_cache()

        # Train probes
        for layer_idx, X in hidden_states.items():
            layer_key = str(layer_idx)
            if layer_key not in results["layers"]:
                results["layers"][layer_key] = {}

            print(f"[probing] Layer {layer_idx}: Training probe on {X.shape}...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
            scores = cross_val_score(clf, X_scaled, labels, cv=5, scoring="accuracy")

            results["layers"][layer_key][model_tag] = {
                "probe_accuracy_mean": float(np.mean(scores)),
                "probe_accuracy_std": float(np.std(scores)),
                "probe_accuracy_per_fold": [float(s) for s in scores],
            }
            print(f"[probing] Layer {layer_idx} ({model_tag}): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Compute deltas
    for layer_key in results["layers"]:
        layer = results["layers"][layer_key]
        if "base" in layer and "finetuned" in layer:
            base_acc = layer["base"]["probe_accuracy_mean"]
            ft_acc = layer["finetuned"]["probe_accuracy_mean"]
            layer["delta"] = ft_acc - base_acc
            layer["knowledge_preserved"] = abs(ft_acc - base_acc) < 0.05  # Within 5%

    return results


def main():
    args = parse_args()

    if args.layers == "all":
        layers = [-1]  # Sentinel for "all layers"
    else:
        layers = [int(x) for x in args.layers.split(",")]

    results = run_probing(
        base_model_name=args.base_model,
        ft_model_name=args.ft_model,
        task=args.task,
        layers=layers,
        num_samples=args.num_samples,
        bf16=args.bf16,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PROBING RESULTS SUMMARY")
    print("=" * 60)
    preserved_layers = sum(1 for l in results["layers"].values() if l.get("knowledge_preserved", False))
    total_layers = len(results["layers"])
    print(f"Knowledge preserved in {preserved_layers}/{total_layers} layers")

    if preserved_layers > total_layers * 0.7:
        print("\n>>> STRONG EVIDENCE: Knowledge persists in hidden states!")
        print(">>> The model didn't forget — it lost alignment to access the knowledge.")
    elif preserved_layers > total_layers * 0.4:
        print("\n>>> MODERATE EVIDENCE: Partial knowledge preservation detected.")
    else:
        print("\n>>> WEAK EVIDENCE: Significant knowledge degradation in hidden states.")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
