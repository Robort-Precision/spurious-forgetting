#!/usr/bin/env python3
"""Evaluation harness: benchmark a model across multiple tasks.

Measures performance on diverse benchmarks to detect catastrophic forgetting.
Designed to run on base model, fine-tuned model, and realigned model.

Usage:
    python eval/harness.py --model meta-llama/Llama-3.2-3B --tasks hellaswag arc_easy arc_challenge mmlu \
        --num-samples 200 --output results/eval_base.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on multiple benchmarks")
    p.add_argument("--model", type=str, required=True, help="Model name or path")
    p.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy", "arc_challenge", "mmlu", "winogrande"])
    p.add_argument("--num-samples", type=int, default=200, help="Samples per task")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=str, default="results/eval.json")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--mmlu-subjects", nargs="*", default=None, help="MMLU subjects (None=all)")
    return p.parse_args()


class BenchmarkTask:
    """Base class for benchmark tasks."""

    def __init__(self, name: str, model, tokenizer, device, num_samples: int = 200):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples

    def score_choices(self, prompt: str, choices: list[str]) -> int:
        """Score multiple choice options by log-likelihood, return best index."""
        log_probs = []
        for choice in choices:
            full_text = prompt + choice
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            prompt_len = prompt_ids["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Log-prob of completion tokens only
            shift_logits = logits[:, prompt_len - 1 : -1, :]
            shift_labels = inputs["input_ids"][:, prompt_len:]

            log_probs_per_token = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs_per_token.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Length-normalized log probability
            avg_log_prob = token_log_probs.sum().item() / max(shift_labels.shape[1], 1)
            log_probs.append(avg_log_prob)

        return int(np.argmax(log_probs))


class HellaSwag(BenchmarkTask):
    def evaluate(self) -> dict:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(self.num_samples, len(ds))))

        correct = 0
        total = 0
        for ex in tqdm(ds, desc="HellaSwag"):
            prompt = ex["ctx"]
            choices = ex["endings"]
            label = int(ex["label"])
            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1
            total += 1

        return {"accuracy": correct / total, "correct": correct, "total": total}


class ARC(BenchmarkTask):
    def __init__(self, difficulty: str = "easy", **kwargs):
        name = f"arc_{difficulty}"
        super().__init__(name=name, **kwargs)
        self.difficulty = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"

    def evaluate(self) -> dict:
        ds = load_dataset("allenai/ai2_arc", self.difficulty, split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(self.num_samples, len(ds))))

        correct = 0
        total = 0
        for ex in tqdm(ds, desc=self.name):
            prompt = f"Question: {ex['question']}\nAnswer:"
            choices = ex["choices"]["text"]
            label_key = ex["answerKey"]
            # Convert A/B/C/D or 1/2/3/4 to index
            if label_key.isdigit():
                label = int(label_key) - 1
            else:
                label = ord(label_key) - ord("A")

            pred = self.score_choices(prompt, choices)
            if pred == label:
                correct += 1
            total += 1

        return {"accuracy": correct / total, "correct": correct, "total": total}


class MMLU(BenchmarkTask):
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_physics", "computer_security",
        "high_school_biology", "high_school_chemistry", "high_school_physics",
        "machine_learning", "world_religions",
    ]

    # Category groupings for per-category breakdown
    CATEGORIES = {
        "STEM": [
            "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
            "college_physics", "computer_security", "high_school_biology",
            "high_school_chemistry", "high_school_physics", "machine_learning",
            "college_computer_science", "college_mathematics", "electrical_engineering",
            "high_school_computer_science", "high_school_mathematics", "high_school_statistics",
        ],
        "Humanities": [
            "formal_logic", "high_school_european_history", "high_school_us_history",
            "high_school_world_history", "international_law", "jurisprudence",
            "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
            "prehistory", "professional_law", "world_religions",
        ],
        "Social_Sciences": [
            "econometrics", "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
            "human_sexuality", "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy",
        ],
        "Other": [
            "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
            "global_facts", "human_aging", "management", "marketing", "medical_genetics",
            "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
            "virology",
        ],
    }

    def __init__(self, subjects: Optional[list] = None, **kwargs):
        super().__init__(name="mmlu", **kwargs)
        self.subjects = subjects or self.SUBJECTS

    def _get_category(self, subject: str) -> str:
        for cat, subjs in self.CATEGORIES.items():
            if subject in subjs:
                return cat
        return "Other"

    def evaluate(self) -> dict:
        results_by_subject = {}
        total_correct = 0
        total_count = 0

        for subject in self.subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            except Exception:
                print(f"  [mmlu] Skipping {subject} — failed to load")
                continue

            samples_per = max(self.num_samples // len(self.subjects), 10)
            ds = ds.shuffle(seed=42).select(range(min(samples_per, len(ds))))

            correct = 0
            for ex in tqdm(ds, desc=f"MMLU/{subject}", leave=False):
                prompt = f"Question: {ex['question']}\nA) {ex['choices'][0]}\nB) {ex['choices'][1]}\nC) {ex['choices'][2]}\nD) {ex['choices'][3]}\nAnswer:"
                choices = [" A", " B", " C", " D"]
                label = ex["answer"]
                pred = self.score_choices(prompt, choices)
                if pred == label:
                    correct += 1

            acc = correct / len(ds) if len(ds) > 0 else 0
            results_by_subject[subject] = {"accuracy": acc, "correct": correct, "total": len(ds)}
            total_correct += correct
            total_count += len(ds)

        # Per-category aggregation
        by_category = {}
        for cat in self.CATEGORIES:
            cat_correct = sum(results_by_subject.get(s, {}).get("correct", 0) for s in self.CATEGORIES[cat])
            cat_total = sum(results_by_subject.get(s, {}).get("total", 0) for s in self.CATEGORIES[cat])
            if cat_total > 0:
                by_category[cat] = {
                    "accuracy": cat_correct / cat_total,
                    "correct": cat_correct,
                    "total": cat_total,
                    "subjects_evaluated": [s for s in self.CATEGORIES[cat] if s in results_by_subject],
                }

        return {
            "accuracy": total_correct / total_count if total_count > 0 else 0,
            "correct": total_correct,
            "total": total_count,
            "by_subject": results_by_subject,
            "by_category": by_category,
        }


class Winogrande(BenchmarkTask):
    def evaluate(self) -> dict:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(self.num_samples, len(ds))))

        correct = 0
        total = 0
        for ex in tqdm(ds, desc="Winogrande"):
            sentence = ex["sentence"]
            option1 = sentence.replace("_", ex["option1"])
            option2 = sentence.replace("_", ex["option2"])
            label = int(ex["answer"]) - 1  # 1-indexed

            # Score full sentences
            log_probs = []
            for option in [option1, option2]:
                inputs = self.tokenizer(option, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                shift_logits = outputs.logits[:, :-1, :]
                shift_labels = inputs["input_ids"][:, 1:]
                lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_lp = lp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                log_probs.append(token_lp.mean().item())

            pred = int(np.argmax(log_probs))
            if pred == label:
                correct += 1
            total += 1

        return {"accuracy": correct / total, "correct": correct, "total": total}


class MedQA(BenchmarkTask):
    def evaluate(self) -> dict:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(self.num_samples, len(ds))))

        correct = 0
        total = 0
        for ex in tqdm(ds, desc="MedQA"):
            question = ex.get("question", ex.get("QUESTION", ""))
            options = ex.get("options", {})
            answer_idx = ex.get("answer_idx", ex.get("ANSWER", ""))

            if isinstance(options, dict):
                choice_keys = sorted(options.keys())
                choices = [options[k] for k in choice_keys]
                prompt = f"Question: {question}\n" + "\n".join(
                    f"{k}) {options[k]}" for k in choice_keys
                ) + "\nAnswer:"
                label = choice_keys.index(answer_idx) if answer_idx in choice_keys else 0
            else:
                choices = list(options) if options else ["A", "B", "C", "D"]
                prompt = f"Question: {question}\nAnswer:"
                label = 0

            pred = self.score_choices(prompt, [f" {c}" for c in ["A", "B", "C", "D"][:len(choices)]])
            if pred == label:
                correct += 1
            total += 1

        return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


class TriviaQA(BenchmarkTask):
    def evaluate(self) -> dict:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(self.num_samples, len(ds))))

        correct = 0
        total = 0
        for ex in tqdm(ds, desc="TriviaQA"):
            question = ex["question"]
            answers = ex["answer"]["aliases"] + [ex["answer"]["value"]]
            answers_lower = [a.lower().strip() for a in answers]

            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated_clean = generated.strip().lower().split("\n")[0]

            if any(ans in generated_clean or generated_clean in ans for ans in answers_lower if ans):
                correct += 1
            total += 1

        return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


TASK_REGISTRY = {
    "hellaswag": HellaSwag,
    "arc_easy": lambda **kw: ARC(difficulty="easy", **kw),
    "arc_challenge": lambda **kw: ARC(difficulty="challenge", **kw),
    "mmlu": MMLU,
    "winogrande": Winogrande,
    "medqa": MedQA,
    "triviaqa": TriviaQA,
}


def main():
    args = parse_args()

    print(f"[eval] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    results = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": {},
    }

    for task_name in args.tasks:
        if task_name not in TASK_REGISTRY:
            print(f"[eval] Unknown task: {task_name}, skipping")
            continue

        print(f"\n[eval] Running: {task_name}")
        task_cls = TASK_REGISTRY[task_name]

        kwargs = dict(model=model, tokenizer=tokenizer, device=device, num_samples=args.num_samples)
        if task_name == "mmlu" and args.mmlu_subjects:
            task = task_cls(subjects=args.mmlu_subjects, **kwargs)
        else:
            task = task_cls(**kwargs)

        task_result = task.evaluate()
        results["tasks"][task_name] = task_result
        print(f"[eval] {task_name}: {task_result['accuracy']:.4f} ({task_result['correct']}/{task_result['total']})")

    # Summary
    accs = [r["accuracy"] for r in results["tasks"].values()]
    results["mean_accuracy"] = float(np.mean(accs)) if accs else 0

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[eval] Mean accuracy: {results['mean_accuracy']:.4f}")
    print(f"[eval] Results saved to {args.output}")


if __name__ == "__main__":
    main()
