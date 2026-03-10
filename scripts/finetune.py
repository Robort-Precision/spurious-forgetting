#!/usr/bin/env python3
"""Fine-tune a model on a narrow task to induce catastrophic forgetting.

Usage:
    python scripts/finetune.py --model meta-llama/Llama-3.2-3B --dataset gsm8k \
        --output-dir results/finetune --epochs 3 --save-checkpoints
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune LLM on narrow task")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "code_alpaca", "squad"])
    p.add_argument("--output-dir", type=str, default="results/finetune")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--save-checkpoints", action="store_true", help="Save intermediate checkpoints")
    p.add_argument("--checkpoint-steps", type=int, default=50, help="Steps between checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--lora", action="store_true", help="Use LoRA instead of full fine-tuning")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--merge-and-save", action="store_true", help="Merge LoRA weights into base model before saving")
    p.add_argument("--merge-alpha", type=float, default=1.0, help="Scaling factor for LoRA merge (0-1 for partial merge)")
    return p.parse_args()


def load_and_format_dataset(name: str, tokenizer, max_length: int):
    """Load dataset and format for causal LM fine-tuning."""

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")

        def format_gsm8k(example):
            text = f"Question: {example['question']}\nAnswer: {example['answer']}\n"
            return tokenizer(text, truncation=True, max_length=max_length, padding=False)

        ds = ds.map(format_gsm8k, remove_columns=ds.column_names, num_proc=4)

    elif name == "code_alpaca":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

        def format_code(example):
            text = f"Instruction: {example['instruction']}\n"
            if example.get("input"):
                text += f"Input: {example['input']}\n"
            text += f"Output: {example['output']}\n"
            return tokenizer(text, truncation=True, max_length=max_length, padding=False)

        ds = ds.map(format_code, remove_columns=ds.column_names, num_proc=4)

    elif name == "squad":
        ds = load_dataset("rajpurkar/squad", split="train")

        def format_squad(example):
            text = (
                f"Context: {example['context']}\n"
                f"Question: {example['question']}\n"
                f"Answer: {example['answers']['text'][0]}\n"
            )
            return tokenizer(text, truncation=True, max_length=max_length, padding=False)

        ds = ds.map(format_squad, remove_columns=ds.column_names, num_proc=4)

    return ds


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[finetune] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print(f"[finetune] Loading dataset: {args.dataset}")
    dataset = load_and_format_dataset(args.dataset, tokenizer, args.max_length)

    # Split for eval
    split = dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"[finetune] Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps" if args.save_checkpoints else "epoch",
        eval_steps=args.checkpoint_steps if args.save_checkpoints else None,
        save_strategy="steps" if args.save_checkpoints else "epoch",
        save_steps=args.checkpoint_steps if args.save_checkpoints else None,
        save_total_limit=20,  # Keep many checkpoints for analysis
        load_best_model_at_end=False,  # We want the most-forgetting model
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[finetune] Starting training...")
    train_result = trainer.train()

    # Save final model
    if args.lora and args.merge_and_save:
        print(f"[finetune] Merging LoRA weights (alpha={args.merge_alpha})...")
        if args.merge_alpha != 1.0:
            # Scale LoRA weights for partial merge
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data *= args.merge_alpha
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output_dir, "final"))
    else:
        trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[finetune] Done. Model saved to {args.output_dir}/final")
    print(f"[finetune] Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
