#!/usr/bin/env python3
"""
SFT/QLoRA training script for dialog datasets on google/gemma-4-E2B-it.

Input CSV columns:
- raw_text
- ready_text
- optional: system

Supports:
- HF token from .env (HF_TOKEN / HUGGINGFACE_TOKEN)
- train/eval split
- checkpoints + auto-resume
- OOM wait/retry loop
"""

import argparse
import gc
import inspect
import os
from pathlib import Path
import re
import subprocess
import time

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GEMMA4_TARGET_MODULES = [
    "q_proj.linear",
    "k_proj.linear",
    "v_proj.linear",
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Gemma dialog LoRA")

    # Compat positional args
    p.add_argument("csv_path_pos", nargs="?", default=None)
    p.add_argument("output_dir_pos", nargs="?", default=None)

    p.add_argument("--csv-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default="google/gemma-4-E2B-it")
    p.add_argument("--max-seq-length", type=int, default=1536)

    p.add_argument("--eval-split", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=3407)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    p.add_argument("--num-train-epochs", type=float, default=1.5)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=40)

    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=3)

    p.add_argument("--default-system", default=(
        "Ты нейтральный эмпатичный ассистент. Отвечай одной уместной репликой, "
        "без флирта и без продолжения диалога за пользователя."
    ))
    p.add_argument("--min-ready-words", type=int, default=3)

    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")

    p.add_argument("--wait-on-oom", action="store_true", default=True)
    p.add_argument("--no-wait-on-oom", dest="wait_on_oom", action="store_false")
    p.add_argument("--oom-wait-seconds", type=int, default=120)
    p.add_argument("--max-oom-retries", type=int, default=200)
    p.add_argument("--min-free-gpu-mb", type=int, default=2048)

    p.add_argument("--use-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="use_4bit", action="store_false")

    return p.parse_args()


def clean_ready_text(text: str) -> str:
    t = str(text).strip()
    markers = [r"\nПользователь:", r"\nАссистент:", r"<\|im_start\|>user", r"<\|im_start\|>assistant", r"\nuser:", r"\nassistant:"]
    cut_positions = []
    for m in markers:
        found = re.search(m, t, flags=re.IGNORECASE)
        if found:
            cut_positions.append(found.start())
    if cut_positions:
        t = t[:min(cut_positions)].strip()
    t = re.sub(r"^\s*(Ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def query_free_gpu_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if not out:
            return None
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def wait_for_gpu(min_free_mb: int, wait_sec: int) -> None:
    if not torch.cuda.is_available():
        return
    while True:
        free_mb = query_free_gpu_mb()
        if free_mb is None or free_mb >= min_free_mb:
            return
        print(f"Waiting GPU memory: {free_mb} < {min_free_mb} MiB, sleep {wait_sec}s")
        time.sleep(wait_sec)


def build_prompt(row: pd.Series, has_system: bool, default_system: str, eos_token: str) -> str:
    system_msg = default_system
    if has_system:
        sv = row.get("system")
        if pd.notna(sv) and str(sv).strip():
            system_msg = str(sv).strip()

    raw_text = str(row["raw_text"]).strip()
    ready_text = str(row["ready_text"]).strip()

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{raw_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{ready_text}<|im_end|>{eos_token}"
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    args.csv_path = args.csv_path or args.csv_path_pos
    args.output_dir = args.output_dir or args.output_dir_pos
    if not args.csv_path or not args.output_dir:
        raise SystemExit("Pass dataset and output dir via flags or positional args")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading tokenizer/model ===")
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=GEMMA4_TARGET_MODULES,
    )
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as e:
        msg = str(e)
        if "Gemma4ClippableLinear" in msg:
            raise SystemExit(
                "PEFT cannot inject LoRA into Gemma4 wrapper modules in your environment.\\n"
                "Try one of these:\\n"
                "1) upgrade peft/transformers/bitsandbytes\\n"
                "2) run with --no-4bit\\n"
                "3) keep this script and use a newer peft that supports Gemma4 path\\n"
                f"Original error: {e}"
            )
        raise

    print("=== Loading/cleaning dataset ===")
    df = pd.read_csv(args.csv_path)
    if "raw_text" not in df.columns or "ready_text" not in df.columns:
        raise ValueError("CSV must contain raw_text and ready_text")

    df = df.dropna(subset=["raw_text", "ready_text"]).copy()
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["ready_text"] = df["ready_text"].astype(str).apply(clean_ready_text)
    df = df[(df["raw_text"] != "") & (df["ready_text"] != "")]
    if args.min_ready_words > 0:
        df = df[df["ready_text"].str.split().str.len() >= args.min_ready_words]
    df = df.drop_duplicates(subset=["raw_text", "ready_text"])

    has_system = "system" in df.columns
    eos = tokenizer.eos_token or ""
    df["text"] = df.apply(lambda row: build_prompt(row, has_system, args.default_system, eos), axis=1)

    ds = Dataset.from_pandas(df[["text"]], preserve_index=False)
    if args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    if eval_ds is not None and args.save_steps % args.eval_steps != 0:
        print(f"Adjust save_steps {args.save_steps} -> {args.eval_steps} for compatibility")
        args.save_steps = args.eval_steps

    ta_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": eval_ds is not None,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "eval_steps": args.eval_steps if eval_ds is not None else None,
    }

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "steps" if eval_ds is not None else "no"
    else:
        ta_kwargs["evaluation_strategy"] = "steps" if eval_ds is not None else "no"
    if "group_by_length" in sig:
        ta_kwargs["group_by_length"] = True
    if "dataloader_num_workers" in sig:
        ta_kwargs["dataloader_num_workers"] = 2

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "packing": False,
        "args": TrainingArguments(**ta_kwargs),
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_ds is not None else [],
    }

    # TRL compatibility across versions:
    # pass only kwargs supported by current SFTTrainer signature.
    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sft_sig}

    if "tokenizer" in sft_sig:
        filtered_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_sig:
        filtered_kwargs["processing_class"] = tokenizer

    # If neither dataset_text_field nor formatting_func is supported, keep dataset with "text" column;
    # newer/older TRL versions may infer defaults.
    trainer = None
    try:
        trainer = SFTTrainer(**filtered_kwargs)
    except KeyError as e:
        if "push_to_hub_token" not in str(e):
            raise
        print("SFTTrainer compatibility issue detected (push_to_hub_token). Falling back to transformers.Trainer.")

        def tok_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )

        train_tok = train_ds.map(tok_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c != "text"])
        train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c == "text"])

        eval_tok = None
        if eval_ds is not None:
            eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=[c for c in eval_ds.column_names if c != "text"])
            eval_tok = eval_tok.remove_columns([c for c in eval_tok.column_names if c == "text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer_kwargs_tf = {
            "model": model,
            "args": TrainingArguments(**ta_kwargs),
            "train_dataset": train_tok,
            "eval_dataset": eval_tok,
            "data_collator": data_collator,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_tok is not None else [],
        }
        tr_sig = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in tr_sig:
            trainer_kwargs_tf["tokenizer"] = tokenizer
        elif "processing_class" in tr_sig:
            trainer_kwargs_tf["processing_class"] = tokenizer

        trainer = Trainer(**trainer_kwargs_tf)

    print("=== Training ===")
    retries = 0
    while True:
        if args.wait_on_oom:
            wait_for_gpu(args.min_free_gpu_mb, args.oom_wait_seconds)

        ckpt = latest_checkpoint(args.output_dir) if args.resume else None
        try:
            if ckpt:
                print(f"Resuming from checkpoint: {ckpt}")
                trainer.train(resume_from_checkpoint=ckpt)
            else:
                trainer.train()
            break
        except torch.OutOfMemoryError:
            if not args.wait_on_oom:
                raise
            retries += 1
            if retries > args.max_oom_retries:
                raise
            print(f"OOM retry {retries}/{args.max_oom_retries}. Sleep {args.oom_wait_seconds}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(args.oom_wait_seconds)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
