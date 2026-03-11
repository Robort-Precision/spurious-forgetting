#!/usr/bin/env python3
"""SVD Intruder Dimension Recovery.

Identifies "intruder dimensions" in LoRA weight deltas — high-ranking singular
vectors in the fine-tuned adapter that are NOT present in the base model's
weight space — and generates scaled-down versions.

Reference: Shuttleworth et al. 2024, "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
Key insight: Catastrophic forgetting localizes to intruder dimensions (new
directions introduced by fine-tuning that weren't in the base model's subspace).

Usage:
    python scripts/svd_recovery.py \
        --base-model mistralai/Mistral-7B-v0.3 \
        --adapter results/finetune/final \
        --output results/svd_recovery/ \
        --scales 0.0 0.25 0.5 0.75 1.0
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file


def parse_args():
    p = argparse.ArgumentParser(description="SVD intruder dimension recovery for LoRA adapters")
    p.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    p.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    p.add_argument("--output", type=str, default="results/svd_recovery/", help="Output directory")
    p.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="Scaling factors for intruder dimensions")
    p.add_argument("--top-k", type=int, default=10,
                    help="Number of top singular vectors to analyze for intruder detection")
    p.add_argument("--intruder-threshold", type=float, default=0.5,
                    help="Cosine similarity threshold below which a direction is 'intruder' (0-1)")
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def compute_weight_delta(base_weight: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor,
                         scaling: float = 1.0) -> torch.Tensor:
    """Compute the effective weight delta from LoRA matrices: delta_W = scaling * B @ A."""
    return scaling * (lora_B @ lora_A).float()


def identify_intruder_dims(base_weight: torch.Tensor, delta_W: torch.Tensor,
                           top_k: int = 10, threshold: float = 0.5):
    """Identify intruder dimensions in the weight delta.

    An intruder dimension is a high-ranking right singular vector of delta_W
    that has low alignment (cosine similarity) with the top singular vectors
    of the base weight matrix.

    Returns:
        dict with 'U', 'S', 'Vt' of delta_W SVD, 'intruder_mask' boolean array,
        'alignment_scores' per singular vector, and 'num_intruders'.
    """
    W_base = base_weight.float()
    dW = delta_W.float()

    # SVD of weight delta
    U_d, S_d, Vt_d = torch.linalg.svd(dW, full_matrices=False)

    # SVD of base weight (we only need right singular vectors for subspace comparison)
    _, _, Vt_base = torch.linalg.svd(W_base, full_matrices=False)

    # Take top-k singular vectors from base model
    V_base_top = Vt_base[:min(top_k * 5, Vt_base.shape[0])].T  # (hidden, k*5) — generous subspace

    # For each top-k singular vector of the delta, compute max alignment with base subspace
    alignment_scores = []
    for i in range(min(top_k, Vt_d.shape[0])):
        v_delta = Vt_d[i]  # (hidden,)
        # Project onto base subspace and measure how much is captured
        proj = V_base_top @ (V_base_top.T @ v_delta)
        cos_sim = torch.nn.functional.cosine_similarity(v_delta.unsqueeze(0), proj.unsqueeze(0)).item()
        alignment_scores.append(abs(cos_sim))

    # Intruder = low alignment with base subspace
    intruder_mask = [score < threshold for score in alignment_scores]

    return {
        "U": U_d[:, :top_k],
        "S": S_d[:top_k],
        "Vt": Vt_d[:top_k],
        "intruder_mask": intruder_mask,
        "alignment_scores": alignment_scores,
        "num_intruders": sum(intruder_mask),
    }


def scale_intruder_dims(lora_A: torch.Tensor, lora_B: torch.Tensor,
                        svd_info: dict, scale: float,
                        lora_scaling: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale intruder dimensions in the LoRA weight delta and recompose into LoRA A/B.

    Approach: Reconstruct delta_W with intruder singular vectors scaled, then
    re-factorize into LoRA-compatible A/B matrices of the same rank.
    """
    U = svd_info["U"]
    S = svd_info["S"]
    Vt = svd_info["Vt"]
    mask = svd_info["intruder_mask"]

    # Scale intruder singular values
    S_modified = S.clone()
    for i, is_intruder in enumerate(mask):
        if is_intruder:
            S_modified[i] *= scale

    # Reconstruct the modified top-k component
    delta_modified_topk = U @ torch.diag(S_modified) @ Vt

    # Get original delta and subtract old top-k, add modified top-k
    delta_original = lora_scaling * (lora_B @ lora_A).float()
    delta_original_topk = U @ torch.diag(S) @ Vt

    delta_modified = delta_original - delta_original_topk + delta_modified_topk

    # Re-factorize into LoRA-compatible matrices via SVD
    rank = lora_A.shape[0]  # LoRA rank
    U_new, S_new, Vt_new = torch.linalg.svd(delta_modified, full_matrices=False)

    # New A = sqrt(S[:r]) @ Vt[:r]  and  B = U[:,:r] @ sqrt(S[:r])
    sqrt_S = torch.sqrt(S_new[:rank])
    new_A = (torch.diag(sqrt_S) @ Vt_new[:rank]).to(lora_A.dtype)
    new_B = (U_new[:, :rank] @ torch.diag(sqrt_S)).to(lora_B.dtype)

    # Adjust for lora_scaling: delta = scaling * B @ A, so divide out scaling
    if lora_scaling != 0:
        new_A = new_A / (lora_scaling ** 0.5)
        new_B = new_B / (lora_scaling ** 0.5)

    return new_A, new_B


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[svd_recovery] Base model: {args.base_model}")
    print(f"[svd_recovery] Adapter: {args.adapter}")
    print(f"[svd_recovery] Scales: {args.scales}")
    print(f"[svd_recovery] Intruder threshold: {args.intruder_threshold}")

    # Load base model
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print("[svd_recovery] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True, device_map="cpu"
    )

    # Load adapter config
    peft_config = PeftConfig.from_pretrained(args.adapter)
    lora_alpha = peft_config.lora_alpha
    lora_r = peft_config.r
    lora_scaling = lora_alpha / lora_r

    # Load adapter weights
    adapter_path = Path(args.adapter)
    adapter_files = list(adapter_path.glob("adapter_model*.safetensors"))
    if adapter_files:
        adapter_weights = {}
        for f in adapter_files:
            adapter_weights.update(load_file(str(f)))
    else:
        bin_files = list(adapter_path.glob("adapter_model*.bin"))
        if bin_files:
            adapter_weights = torch.load(str(bin_files[0]), map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {args.adapter}")

    # Group LoRA A/B pairs
    lora_pairs = {}
    for key in adapter_weights:
        if "lora_A" in key:
            base_key = key.replace("lora_A.weight", "").replace("lora_A.default.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = key
        elif "lora_B" in key:
            base_key = key.replace("lora_B.weight", "").replace("lora_B.default.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = key

    print(f"[svd_recovery] Found {len(lora_pairs)} LoRA layer pairs")

    # Analyze each layer pair
    analysis_results = {"layers": {}, "config": vars(args)}
    total_intruders = 0

    for layer_key, pair in tqdm(lora_pairs.items(), desc="Analyzing layers"):
        if "A" not in pair or "B" not in pair:
            continue

        lora_A = adapter_weights[pair["A"]].float()
        lora_B = adapter_weights[pair["B"]].float()

        # Find corresponding base weight
        # Convert adapter key to base model key
        # e.g., "base_model.model.model.layers.0.self_attn.q_proj." → model.layers.0.self_attn.q_proj.weight
        base_key_parts = layer_key.replace("base_model.model.", "").strip(".")
        base_param_name = base_key_parts + ".weight" if not base_key_parts.endswith(".weight") else base_key_parts

        base_weight = None
        for name, param in base_model.named_parameters():
            if name == base_param_name or name.endswith(base_param_name):
                base_weight = param.data.float()
                break

        if base_weight is None:
            print(f"  [skip] No base weight found for {layer_key}")
            continue

        # Compute weight delta
        delta_W = compute_weight_delta(base_weight, lora_A, lora_B, scaling=lora_scaling)

        # Identify intruder dimensions
        svd_info = identify_intruder_dims(
            base_weight, delta_W,
            top_k=args.top_k,
            threshold=args.intruder_threshold
        )

        layer_name = layer_key.strip(".")
        analysis_results["layers"][layer_name] = {
            "num_intruders": svd_info["num_intruders"],
            "alignment_scores": svd_info["alignment_scores"],
            "intruder_mask": svd_info["intruder_mask"],
            "top_singular_values": svd_info["S"].tolist(),
        }
        total_intruders += svd_info["num_intruders"]

        # Store SVD info for later use
        lora_pairs[layer_key]["svd_info"] = svd_info

    print(f"\n[svd_recovery] Total intruder dimensions found: {total_intruders}")

    # Save analysis
    analysis_path = output_dir / "intruder_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"[svd_recovery] Analysis saved to {analysis_path}")

    # Generate scaled adapters
    for scale in args.scales:
        scale_dir = output_dir / f"scale_{scale:.2f}"
        scale_dir.mkdir(parents=True, exist_ok=True)

        # Copy config files
        for config_file in ["adapter_config.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = adapter_path / config_file
            if src.exists():
                shutil.copy2(str(src), str(scale_dir / config_file))

        # Generate modified weights
        modified_weights = dict(adapter_weights)

        for layer_key, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair or "svd_info" not in pair:
                continue

            lora_A = adapter_weights[pair["A"]]
            lora_B = adapter_weights[pair["B"]]
            svd_info = pair["svd_info"]

            if svd_info["num_intruders"] == 0:
                continue  # No intruders to scale

            new_A, new_B = scale_intruder_dims(
                lora_A.float(), lora_B.float(),
                svd_info, scale=scale,
                lora_scaling=lora_scaling
            )

            modified_weights[pair["A"]] = new_A.to(lora_A.dtype)
            modified_weights[pair["B"]] = new_B.to(lora_B.dtype)

        # Save modified adapter
        save_file(modified_weights, str(scale_dir / "adapter_model.safetensors"))
        print(f"[svd_recovery] Saved scale={scale:.2f} adapter to {scale_dir}")

    # Summary
    print(f"\n[svd_recovery] Done! Generated {len(args.scales)} scaled adapters in {output_dir}")
    print(f"[svd_recovery] scale=0.00 → intruder dims completely removed (max recovery)")
    print(f"[svd_recovery] scale=1.00 → original adapter (no modification)")


if __name__ == "__main__":
    main()
