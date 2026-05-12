#!/usr/bin/env python3
"""
SFT for dialogue-teacher LoRA.

Goal:
- Teach a base model to generate high-quality dialogue examples from limited data.
- Keep the adapter less overfitted and more generalizable for synthetic data generation.

Usage example:
python sft_train_dialog.py \
  --csv-path datasets/chats/dialog_sft_ira_assistant.csv \
  --output-dir models/qwen35_4b_dialog_teacher_ira
"""

import argparse
import os

from unsloth import FastLanguageModel
import pandas as pd
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainingArguments
import inspect
from trl import SFTTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train dialogue-teacher LoRA with Unsloth + TRL")

    # Backward-compatible positional args:
    # python sft_train_dialog.py <csv_path> <output_dir>
    parser.add_argument("csv_path_pos", nargs="?", default=None, help="(optional) positional csv path")
    parser.add_argument("output_dir_pos", nargs="?", default=None, help="(optional) positional output dir")

    # Data
    parser.add_argument("--csv-path", type=str, default=None, help="CSV with raw_text, ready_text, optional system")
    parser.add_argument("--eval-split", type=float, default=0.03, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=3407)

    # Model
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")

    # LoRA (teacher-oriented defaults for small datasets)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)

    # Prompt style
    parser.add_argument(
        "--default-system",
        type=str,
        default=(
            "Ты агент-учитель. Генерируй естественные, разнообразные и реалистичные ответы "
            "в стиле диалога. Избегай шаблонности, сохраняй краткость и уместный тон."
        ),
    )

    return parser


def to_chat_text(row: pd.Series, eos_token: str, has_system: bool, default_system: str) -> str:
    system_msg = default_system
    if has_system:
        value = row.get("system")
        if pd.notna(value) and str(value).strip():
            system_msg = str(value).strip()

    raw_text = str(row["raw_text"]).strip()
    ready_text = str(row["ready_text"]).strip()

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{raw_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{ready_text}<|im_end|>{eos_token}"
    )


def main() -> None:
    args = build_parser().parse_args()
    args.csv_path = args.csv_path or args.csv_path_pos
    args.output_dir = args.output_dir or args.output_dir_pos

    if not args.csv_path or not args.output_dir:
        raise SystemExit(
            "Provide dataset and output dir either as flags:\n"
            "  --csv-path <path> --output-dir <path>\n"
            "or as positional args:\n"
            "  sft_train_dialog.py <csv_path> <output_dir>"
        )

    # Convenience fallback for common path typo: datasets/dialog_*.csv vs datasets/chats/dialog_*.csv
    if not os.path.isfile(args.csv_path):
        candidate = os.path.join("datasets", "chats", os.path.basename(args.csv_path))
        if os.path.isfile(candidate):
            print(f"CSV not found at '{args.csv_path}', using '{candidate}'")
            args.csv_path = candidate

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading model ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    print("=== Applying LoRA ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print("=== Loading dataset ===")
    df = pd.read_csv(args.csv_path)
    if "raw_text" not in df.columns or "ready_text" not in df.columns:
        raise ValueError("CSV must contain columns: raw_text, ready_text")

    # Remove empty rows and duplicates to reduce memorization on small corpora
    df = df.dropna(subset=["raw_text", "ready_text"]).copy()
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["ready_text"] = df["ready_text"].astype(str).str.strip()
    df = df[(df["raw_text"] != "") & (df["ready_text"] != "")]
    df = df.drop_duplicates(subset=["raw_text", "ready_text"])

    has_system = "system" in df.columns
    eos_token = tokenizer.eos_token

    df["text"] = df.apply(
        lambda row: to_chat_text(
            row=row,
            eos_token=eos_token,
            has_system=has_system,
            default_system=args.default_system,
        ),
        axis=1,
    )

    dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Train size: {len(train_dataset)} | Eval disabled")

    print("=== Building trainer ===")
    # Build TrainingArguments with backward compatibility across transformers versions.
    ta_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": (eval_dataset is not None),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
    }

    # transformers in some environments uses evaluation_strategy, not eval_strategy.
    ta_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_signature:
        ta_kwargs["eval_strategy"] = "steps" if eval_dataset is not None else "no"
    else:
        ta_kwargs["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"

    if "group_by_length" in ta_signature:
        ta_kwargs["group_by_length"] = True
    if "dataloader_num_workers" in ta_signature:
        ta_kwargs["dataloader_num_workers"] = 2

    training_args = TrainingArguments(**ta_kwargs)

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
        callbacks=callbacks,
    )

    print("\n=== Training started ===\n")
    trainer.train()

    print("\n=== Saving adapter/tokenizer ===")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
