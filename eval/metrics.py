#!/usr/bin/env python3
"""Compute paper-specific metrics: PTDS, LDI, FEFR, and combined analysis.

Reads probing + CKA + eval results and produces a unified metrics table.

Usage:
    python eval/metrics.py --probing results/probing.json --cka results/cka.json \
        --base-eval results/eval_base.json --ft-eval results/eval_ft.json \
        --output results/metrics.json
"""

import argparse
import json
import os
import csv

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute paper metrics (PTDS, LDI, FEFR)")
    p.add_argument("--probing", type=str, required=True, help="Probing results JSON")
    p.add_argument("--cka", type=str, default=None, help="CKA results JSON")
    p.add_argument("--base-eval", type=str, default=None, help="Base model eval JSON")
    p.add_argument("--ft-eval", type=str, default=None, help="Fine-tuned eval JSON")
    p.add_argument("--realignment-eval", type=str, default=None, help="Realigned model eval JSON")
    p.add_argument("--output", type=str, default="results/metrics.json")
    p.add_argument("--csv", type=str, default=None, help="Also output as CSV")
    return p.parse_args()


def compute_ptds(probe_acc: float, task_acc: float) -> float:
    """Probe-Task Divergence Score.
    
    PTDS_l = (probe_acc_l - task_acc) / probe_acc_l
    
    High PTDS = knowledge present in representations but not expressed in output.
    PTDS > 0.3 = strong evidence of spurious forgetting.
    """
    if probe_acc < 1e-6:
        return 0.0
    return (probe_acc - task_acc) / probe_acc


def compute_ldi(cka_value: float) -> float:
    """Layer Disruption Index.
    
    LDI_l = 1 - CKA(base_layer_l, ft_layer_l)
    
    High LDI = representations changed significantly at this layer.
    """
    return 1.0 - cka_value


def compute_fefr(base_acc: float, ft_acc: float) -> float:
    """Fine-tuning Efficiency-Forgetting Ratio.
    
    FEFR = |drop in non-target tasks| / |gain in target task|
    High FEFR = lots of forgetting per unit of learning.
    """
    if abs(ft_acc) < 1e-6:
        return float("inf")
    return abs(base_acc - ft_acc) / abs(ft_acc)


def compute_recovery_efficiency(base_acc: float, ft_acc: float, recovered_acc: float) -> float:
    """Recovery efficiency η.
    
    η = (recovered_acc - ft_acc) / (base_acc - ft_acc)
    
    η = 1.0 means full recovery, η = 0 means no recovery.
    """
    gap = base_acc - ft_acc
    if abs(gap) < 1e-6:
        return 1.0
    return (recovered_acc - ft_acc) / gap


def main():
    args = parse_args()

    probing = json.load(open(args.probing))
    cka = json.load(open(args.cka)) if args.cka else None
    base_eval = json.load(open(args.base_eval)) if args.base_eval else None
    ft_eval = json.load(open(args.ft_eval)) if args.ft_eval else None
    realign_eval = json.load(open(args.realignment_eval)) if args.realignment_eval else None

    results = {"layers": [], "summary": {}}

    # Get task accuracy from eval files
    task_accs = {}
    if ft_eval and "tasks" in ft_eval:
        for task, data in ft_eval["tasks"].items():
            task_accs[task] = data.get("accuracy", 0)
    ft_mean_acc = np.mean(list(task_accs.values())) if task_accs else 0

    base_accs = {}
    if base_eval and "tasks" in base_eval:
        for task, data in base_eval["tasks"].items():
            base_accs[task] = data.get("accuracy", 0)

    # Process each probed layer
    for layer_key, layer_data in probing.get("layers", {}).items():
        layer_idx = int(layer_key)
        row = {"layer": layer_idx}

        # Probe accuracy (from fine-tuned model)
        ft_probe = layer_data.get("finetuned", {})
        base_probe = layer_data.get("base", {})

        ft_probe_acc = ft_probe.get("probe_accuracy_mean", 0)
        base_probe_acc = base_probe.get("probe_accuracy_mean", 0)

        row["base_probe_acc"] = base_probe_acc
        row["ft_probe_acc"] = ft_probe_acc
        row["probe_delta"] = ft_probe_acc - base_probe_acc

        # PTDS
        task = probing.get("task", "mmlu")
        task_acc = task_accs.get(task, ft_mean_acc)
        row["task_acc"] = task_acc
        row["ptds"] = compute_ptds(ft_probe_acc, task_acc)

        # LDI from CKA
        if cka and layer_key in cka.get("layers", {}):
            cka_val = cka["layers"][layer_key]["cka"]
            row["cka"] = cka_val
            row["ldi"] = compute_ldi(cka_val)
        else:
            row["cka"] = None
            row["ldi"] = None

        # Knowledge preserved flag
        row["knowledge_preserved"] = layer_data.get("knowledge_preserved", abs(ft_probe_acc - base_probe_acc) < 0.05)

        results["layers"].append(row)

    # Sort by layer
    results["layers"].sort(key=lambda r: r["layer"])

    # Summary metrics
    if base_accs and task_accs:
        for task in set(base_accs.keys()) & set(task_accs.keys()):
            results["summary"][f"fefr_{task}"] = compute_fefr(base_accs[task], task_accs[task])

    preserved = sum(1 for r in results["layers"] if r.get("knowledge_preserved"))
    total = len(results["layers"])
    results["summary"]["knowledge_preservation_rate"] = preserved / total if total > 0 else 0
    results["summary"]["mean_ptds"] = float(np.mean([r["ptds"] for r in results["layers"]])) if results["layers"] else 0

    if cka:
        ldi_vals = [r["ldi"] for r in results["layers"] if r["ldi"] is not None]
        results["summary"]["mean_ldi"] = float(np.mean(ldi_vals)) if ldi_vals else 0

    # Recovery efficiency
    if realign_eval and base_eval and ft_eval:
        for task in base_accs:
            if task in task_accs:
                realign_acc = realign_eval.get("tasks", {}).get(task, {}).get("accuracy", task_accs[task])
                results["summary"][f"recovery_eta_{task}"] = compute_recovery_efficiency(
                    base_accs[task], task_accs[task], realign_acc
                )

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV if requested
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            fieldnames = ["layer", "benchmark", "base_probe_acc", "ft_probe_acc", "task_acc", "ptds", "cka", "ldi"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results["layers"]:
                writer.writerow({
                    "layer": row["layer"],
                    "benchmark": probing.get("task", "mmlu"),
                    "base_probe_acc": f"{row['base_probe_acc']:.4f}",
                    "ft_probe_acc": f"{row['ft_probe_acc']:.4f}",
                    "task_acc": f"{row['task_acc']:.4f}",
                    "ptds": f"{row['ptds']:.4f}",
                    "cka": f"{row['cka']:.4f}" if row["cka"] is not None else "",
                    "ldi": f"{row['ldi']:.4f}" if row["ldi"] is not None else "",
                })

    # Print
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Layer':>6} {'Base Probe':>11} {'FT Probe':>10} {'Task Acc':>9} {'PTDS':>8} {'CKA':>8} {'LDI':>8} {'Preserved':>10}")
    print("-" * 80)
    for r in results["layers"]:
        cka_str = f"{r['cka']:.4f}" if r["cka"] is not None else "N/A"
        ldi_str = f"{r['ldi']:.4f}" if r["ldi"] is not None else "N/A"
        print(f"{r['layer']:>6} {r['base_probe_acc']:>11.4f} {r['ft_probe_acc']:>10.4f} {r['task_acc']:>9.4f} {r['ptds']:>8.4f} {cka_str:>8} {ldi_str:>8} {'✓' if r.get('knowledge_preserved') else '✗':>10}")

    print(f"\nMean PTDS: {results['summary']['mean_ptds']:.4f}")
    print(f"Knowledge preserved: {preserved}/{total} layers ({results['summary']['knowledge_preservation_rate']:.1%})")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
