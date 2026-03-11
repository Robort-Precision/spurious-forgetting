#!/usr/bin/env python3
"""Comprehensive mechanistic analysis pipeline.

Runs logit lens + CKA + probing on a given checkpoint, producing a full
analysis report. Single entry point for all mechanistic diagnostics.

Usage:
    python scripts/run_analysis.py \
        --base mistralai/Mistral-7B-v0.3 \
        --adapter results/targeted_D/best \
        --output results/analysis_D/

    # With merged model instead of adapter:
    python scripts/run_analysis.py \
        --base mistralai/Mistral-7B-v0.3 \
        --merged results/targeted_D/merged \
        --output results/analysis_D/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run full mechanistic analysis pipeline")
    p.add_argument("--base", type=str, required=True, help="Base model name or path")
    p.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    p.add_argument("--merged", type=str, default=None, help="Pre-merged model path (alternative to --adapter)")
    p.add_argument("--output", type=str, required=True, help="Output directory for all results")
    p.add_argument("--num-samples", type=int, default=100, help="Samples for CKA and logit lens")
    p.add_argument("--probe-epochs", type=int, default=10, help="Probing classifier epochs")
    p.add_argument("--tasks", nargs="+", default=["mmlu", "arc_challenge"],
                    help="Tasks for CKA and logit lens")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--skip-merge", action="store_true", help="Skip merge step (use --merged)")
    p.add_argument("--skip-cka", action="store_true")
    p.add_argument("--skip-logit-lens", action="store_true")
    p.add_argument("--skip-probing", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def run_cmd(cmd: list[str], desc: str) -> tuple[int, str]:
    """Run a command, print output, return (returncode, stdout)."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[WARN] {desc} exited with code {result.returncode}")

    return result.returncode, result.stdout


def merge_adapter(base: str, adapter: str, output: str, bf16: bool = True) -> str:
    """Merge LoRA adapter with base model."""
    print(f"[merge] Merging {adapter} into {base}...")

    merge_script = f"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

dtype = torch.bfloat16 if {bf16} else torch.float32
model = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=dtype, trust_remote_code=True, device_map="cpu")
model = PeftModel.from_pretrained(model, "{adapter}")
model = model.merge_and_unload()
model.save_pretrained("{output}")
tokenizer = AutoTokenizer.from_pretrained("{base}", trust_remote_code=True)
tokenizer.save_pretrained("{output}")
print("[merge] Done!")
"""
    script_path = Path(output).parent / "_merge_temp.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(merge_script)

    rc, _ = run_cmd([sys.executable, str(script_path)], "Merging adapter")
    script_path.unlink(missing_ok=True)

    if rc != 0:
        raise RuntimeError("Adapter merge failed")
    return output


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine the model to analyze
    if args.merged:
        ft_model = args.merged
    elif args.adapter:
        if args.skip_merge:
            raise ValueError("--adapter requires merge step. Use --merged or remove --skip-merge")
        merged_path = str(output_dir / "merged_model")
        ft_model = merge_adapter(args.base, args.adapter, merged_path, args.bf16)
    else:
        raise ValueError("Provide either --adapter or --merged")

    # Find the repo root (where eval/ and probing/ dirs are)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    start_time = time.time()
    report = {
        "config": {
            "base_model": args.base,
            "adapter": args.adapter,
            "merged_model": ft_model,
            "num_samples": args.num_samples,
            "tasks": args.tasks,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": {},
        "timing": {},
    }

    # 1. EVAL HARNESS
    if not args.skip_eval:
        t0 = time.time()
        eval_output = str(output_dir / "eval_results.json")
        rc, _ = run_cmd([
            sys.executable, str(repo_root / "eval" / "harness.py"),
            "--model", ft_model,
            "--tasks", "hellaswag", "arc_challenge", "mmlu", "winogrande",
            "--num-samples", str(args.num_samples),
            "--output", eval_output,
        ], "Evaluation Harness")

        if rc == 0 and Path(eval_output).exists():
            with open(eval_output) as f:
                report["results"]["eval"] = json.load(f)
        report["timing"]["eval"] = time.time() - t0

    # 2. CKA ANALYSIS
    if not args.skip_cka:
        for task in args.tasks:
            t0 = time.time()
            cka_output = str(output_dir / f"cka_{task}.json")
            rc, _ = run_cmd([
                sys.executable, str(repo_root / "eval" / "cka.py"),
                "--base-model", args.base,
                "--ft-model", ft_model,
                "--task", task,
                "--num-samples", str(args.num_samples),
                "--output", cka_output,
            ], f"CKA Analysis ({task})")

            if rc == 0 and Path(cka_output).exists():
                with open(cka_output) as f:
                    report["results"][f"cka_{task}"] = json.load(f)
            report["timing"][f"cka_{task}"] = time.time() - t0

    # 3. LOGIT LENS
    if not args.skip_logit_lens:
        for task in args.tasks:
            t0 = time.time()
            ll_output = str(output_dir / f"logit_lens_{task}.json")
            rc, _ = run_cmd([
                sys.executable, str(repo_root / "eval" / "logit_lens.py"),
                "--model", ft_model,
                "--base-model", args.base,
                "--task", task,
                "--num-samples", str(args.num_samples),
                "--output", ll_output,
            ], f"Logit Lens ({task})")

            if rc == 0 and Path(ll_output).exists():
                with open(ll_output) as f:
                    report["results"][f"logit_lens_{task}"] = json.load(f)
            report["timing"][f"logit_lens_{task}"] = time.time() - t0

    # 4. PROBING
    if not args.skip_probing:
        t0 = time.time()
        probe_output = str(output_dir / "probing_results.json")
        rc, _ = run_cmd([
            sys.executable, str(repo_root / "probing" / "classifier.py"),
            "--model", ft_model,
            "--base-model", args.base,
            "--output", probe_output,
            "--epochs", str(args.probe_epochs),
            "--num-samples", str(args.num_samples),
        ], "Linear Probing")

        if rc == 0 and Path(probe_output).exists():
            with open(probe_output) as f:
                report["results"]["probing"] = json.load(f)
        report["timing"]["probing"] = time.time() - t0

    # Generate summary report
    total_time = time.time() - start_time
    report["timing"]["total"] = total_time

    # Summary section
    summary = {"total_time_seconds": round(total_time, 1)}

    # Extract key findings
    if "eval" in report["results"]:
        eval_data = report["results"]["eval"]
        summary["benchmark_scores"] = {
            task: f"{data['accuracy']:.1%}"
            for task, data in eval_data.get("tasks", {}).items()
        }

    for task in args.tasks:
        cka_key = f"cka_{task}"
        if cka_key in report["results"]:
            cka_data = report["results"][cka_key]
            summary[f"cka_{task}_mean"] = round(cka_data.get("mean_cka", 0), 4)
            summary[f"cka_{task}_most_changed"] = cka_data.get("most_changed_layer", "N/A")

        ll_key = f"logit_lens_{task}"
        if ll_key in report["results"]:
            ll_data = report["results"][ll_key]
            summary[f"logit_lens_{task}_misalignment"] = ll_data.get("misalignment_detected", False)

    report["summary"] = summary

    # Save full report
    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE — {total_time:.0f}s")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
