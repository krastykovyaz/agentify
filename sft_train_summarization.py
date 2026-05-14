#!/usr/bin/env python3
import argparse
import gc
import inspect
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from unsloth import FastLanguageModel
import pandas as pd
import torch
from datasets import Dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    p.add_argument("--eval-split", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--num-train-epochs", type=float, default=1.8)
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

    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--wait-on-oom", action="store_true", default=True)
    p.add_argument("--no-wait-on-oom", dest="wait_on_oom", action="store_false")
    p.add_argument("--oom-wait-seconds", type=int, default=120)
    p.add_argument("--max-oom-retries", type=int, default=200)
    p.add_argument("--min-free-gpu-mb", type=int, default=2048)
    return p.parse_args()


def format_row(row, eos_token, default_system):
    system = str(row.get("system") or "").strip() or default_system
    raw = str(row["raw_text"]).strip()
    ready = str(row["ready_text"]).strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{raw}<|im_end|>\n"
        f"<|im_start|>assistant\n{ready}<|im_end|>{eos_token}"
    )


def latest_checkpoint(out_dir: str):
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def query_free_gpu_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return int(out.splitlines()[0].strip()) if out else None
    except Exception:
        return None


def wait_for_free_memory(min_free_mb: int, sleep_s: int):
    if not torch.cuda.is_available():
        return
    while True:
        free_mb = query_free_gpu_mb()
        if free_mb is None or free_mb >= min_free_mb:
            return
        print(f"Waiting GPU memory: free {free_mb} MiB < {min_free_mb} MiB. Sleep {sleep_s}s...")
        time.sleep(sleep_s)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    df = pd.read_csv(args.csv_path)
    if "raw_text" not in df.columns or "ready_text" not in df.columns:
        raise ValueError("CSV must contain raw_text and ready_text")

    df = df.dropna(subset=["raw_text", "ready_text"]).copy()
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["ready_text"] = df["ready_text"].astype(str).str.strip()
    df = df[(df["raw_text"] != "") & (df["ready_text"] != "")]
    df = df.drop_duplicates(subset=["raw_text", "ready_text"])

    default_system = "Сделай краткую и точную выжимку текста без выдумок."
    eos = tokenizer.eos_token
    df["text"] = df.apply(lambda r: format_row(r, eos, default_system), axis=1)

    ds = Dataset.from_pandas(df[["text"]], preserve_index=False)
    if args.eval_split > 0:
        split = ds.train_test_split(test_size=args.eval_split, seed=args.seed, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds, None

    if eval_ds is not None and args.save_steps % args.eval_steps != 0:
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
        "optim": "adamw_8bit",
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(**ta_kwargs),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_ds is not None else [],
    )

    retries = 0
    while True:
        if args.wait_on_oom:
            wait_for_free_memory(args.min_free_gpu_mb, args.oom_wait_seconds)
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
            print(f"OOM retry {retries}/{args.max_oom_retries} after {args.oom_wait_seconds}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(args.oom_wait_seconds)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
