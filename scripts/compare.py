#!/usr/bin/env python3
"""Compare base vs fine-tuned vs realigned model performance.

Reads results JSON files and produces summary tables and plots.

Usage:
    python scripts/compare.py --base-eval results/eval_base.json \
        --ft-eval results/eval_ft.json --probing results/probing.json \
        --realignment results/realignment.json --output results/summary.json
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-eval", type=str, required=True)
    p.add_argument("--ft-eval", type=str, required=True)
    p.add_argument("--probing", type=str, default=None)
    p.add_argument("--realignment", type=str, default=None)
    p.add_argument("--output", type=str, default="results/summary.json")
    p.add_argument("--plot-dir", type=str, default="results/plots")
    return p.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_forgetting(base_results, ft_results, output_dir):
    """Bar chart comparing base vs FT accuracy across tasks."""
    tasks = sorted(set(base_results["tasks"].keys()) & set(ft_results["tasks"].keys()))

    base_accs = [base_results["tasks"][t]["accuracy"] for t in tasks]
    ft_accs = [ft_results["tasks"][t]["accuracy"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, base_accs, width, label="Base Model", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, ft_accs, width, label="Fine-Tuned", color="#E45756")

    ax.set_ylabel("Accuracy")
    ax.set_title("Catastrophic Forgetting: Base vs Fine-Tuned Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add delta annotations
    for i, (b, f) in enumerate(zip(base_accs, ft_accs)):
        delta = f - b
        color = "green" if delta >= 0 else "red"
        ax.annotate(f"{delta:+.1%}", xy=(x[i] + width / 2, f), ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forgetting.png"), dpi=150)
    plt.close()


def plot_probing(probing_results, output_dir):
    """Line chart showing probe accuracy across layers for base vs FT."""
    layers = sorted(probing_results["layers"].keys(), key=int)
    base_accs = [probing_results["layers"][l]["base"]["probe_accuracy_mean"] for l in layers]
    ft_accs = [probing_results["layers"][l]["finetuned"]["probe_accuracy_mean"] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([int(l) for l in layers], base_accs, "o-", label="Base Model Probe", color="#4C78A8")
    ax.plot([int(l) for l in layers], ft_accs, "s-", label="Fine-Tuned Probe", color="#E45756")

    # Highlight preserved layers
    for i, l in enumerate(layers):
        if probing_results["layers"][l].get("knowledge_preserved"):
            ax.axvspan(int(l) - 0.5, int(l) + 0.5, alpha=0.1, color="green")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Knowledge Preservation in Hidden States\n(Green = knowledge preserved despite task accuracy drop)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probing.png"), dpi=150)
    plt.close()


def plot_realignment(realignment_results, output_dir):
    """Bar chart showing accuracy: base → FT → realigned for each strategy."""
    base_acc = realignment_results["base_accuracy"]
    strategies = realignment_results["strategies"]

    labels = ["Base"]
    accs = [base_acc]
    colors = ["#4C78A8"]

    for name, data in strategies.items():
        if "ft_accuracy" in data and "ft_accuracy" not in [l for l in labels]:
            labels.append("Fine-Tuned")
            accs.append(data["ft_accuracy"])
            colors.append("#E45756")

        best = data.get("best_accuracy", data.get("adapted_accuracy"))
        if best:
            labels.append(name.replace("_", " ").title())
            accs.append(best)
            colors.append("#72B7B2")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), accs, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Knowledge Realignment: Recovery of Forgotten Capabilities")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    for i, (l, a) in enumerate(zip(labels, accs)):
        ax.text(i, a + 0.01, f"{a:.1%}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "realignment.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    base = load_json(args.base_eval)
    ft = load_json(args.ft_eval)

    # Build summary
    summary = {
        "base_model": base.get("model"),
        "ft_model": ft.get("model"),
        "forgetting": {},
    }

    # Task-by-task forgetting
    for task in base["tasks"]:
        if task in ft["tasks"]:
            b = base["tasks"][task]["accuracy"]
            f = ft["tasks"][task]["accuracy"]
            summary["forgetting"][task] = {
                "base": b,
                "finetuned": f,
                "delta": f - b,
                "pct_drop": (b - f) / b * 100 if b > 0 else 0,
            }

    summary["mean_forgetting"] = np.mean([v["delta"] for v in summary["forgetting"].values()])

    plot_forgetting(base, ft, args.plot_dir)
    print("[compare] Forgetting plot saved")

    if args.probing:
        probing = load_json(args.probing)
        preserved = sum(1 for l in probing["layers"].values() if l.get("knowledge_preserved", False))
        total = len(probing["layers"])
        summary["probing"] = {
            "preserved_layers": preserved,
            "total_layers": total,
            "preservation_rate": preserved / total if total > 0 else 0,
        }
        plot_probing(probing, args.plot_dir)
        print("[compare] Probing plot saved")

    if args.realignment:
        realignment = load_json(args.realignment)
        summary["realignment"] = {}
        for strat, data in realignment["strategies"].items():
            best = data.get("best_accuracy", data.get("adapted_accuracy", 0))
            ft_acc = data.get("ft_accuracy", 0)
            summary["realignment"][strat] = {
                "best_accuracy": best,
                "recovery": best - ft_acc,
                "pct_recovered": ((best - ft_acc) / (realignment["base_accuracy"] - ft_acc) * 100)
                if realignment["base_accuracy"] != ft_acc else 0,
            }
        plot_realignment(realignment, args.plot_dir)
        print("[compare] Realignment plot saved")

    # Save summary
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for task, v in summary["forgetting"].items():
        print(f"  {task}: {v['base']:.4f} → {v['finetuned']:.4f} ({v['delta']:+.4f}, {v['pct_drop']:.1f}% drop)")
    print(f"\n  Mean forgetting: {summary['mean_forgetting']:+.4f}")

    if "probing" in summary:
        p = summary["probing"]
        print(f"\n  Probing: {p['preserved_layers']}/{p['total_layers']} layers preserved ({p['preservation_rate']:.1%})")

    if "realignment" in summary:
        for strat, v in summary["realignment"].items():
            print(f"\n  {strat}: recovered to {v['best_accuracy']:.4f} ({v['pct_recovered']:.1f}% of lost performance)")


if __name__ == "__main__":
    main()
