#!/usr/bin/env python3
import argparse
import gc
import inspect
import os
from pathlib import Path
import subprocess
import time

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

TARGET_MODULES = [
    "q_proj.linear",
    "k_proj.linear",
    "v_proj.linear",
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train Gemma universal LoRA")
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

    p.add_argument("--num-train-epochs", type=float, default=1.2)
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

    p.add_argument(
        "--default-system",
        default=(
            "Ты универсальный ассистент для задач summarization, telegram_post, qa и extraction. "
            "Отвечай точно по задаче, без выдумки фактов."
        ),
    )

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


def latest_checkpoint(output_dir: str):
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def query_free_gpu_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        return int(out.splitlines()[0].strip()) if out else None
    except Exception:
        return None


def wait_for_gpu(min_free_mb: int, wait_sec: int):
    if not torch.cuda.is_available():
        return
    while True:
        free_mb = query_free_gpu_mb()
        if free_mb is None or free_mb >= min_free_mb:
            return
        print(f"Waiting GPU memory: {free_mb} < {min_free_mb} MiB, sleep {wait_sec}s")
        time.sleep(wait_sec)


def make_user_input(row: pd.Series) -> str:
    # If task/domain column exists, inject it explicitly into instruction to stabilize universal behavior.
    task = ""
    for c in ["task", "domain"]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            task = str(row[c]).strip()
            break

    raw = str(row["raw_text"]).strip()
    if task:
        return f"[{task}] {raw}"
    return raw


def build_messages(row: pd.Series, default_system: str) -> list[dict]:
    sys_msg = str(row.get("system") or "").strip() or default_system
    user_text = make_user_input(row)
    ready = str(row["ready_text"]).strip()
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ready},
    ]


def main():
    load_dotenv()
    args = parse_args()
    args.csv_path = args.csv_path or args.csv_path_pos
    args.output_dir = args.output_dir or args.output_dir_pos
    if not args.csv_path or not args.output_dir:
        raise SystemExit("Pass csv and output dir")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.makedirs(args.output_dir, exist_ok=True)

    bnb = None
    if args.use_4bit:
        bnb = BitsAndBytesConfig(
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
        quantization_config=bnb,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)

    print("=== Loading/cleaning dataset ===")
    df = pd.read_csv(args.csv_path)
    if "raw_text" not in df.columns or "ready_text" not in df.columns:
        raise ValueError("CSV must contain raw_text and ready_text")

    df = df.dropna(subset=["raw_text", "ready_text"]).copy()
    df["raw_text"] = df["raw_text"].astype(str).str.strip()
    df["ready_text"] = df["ready_text"].astype(str).str.strip()
    df = df[(df["raw_text"] != "") & (df["ready_text"] != "")]
    df = df.drop_duplicates(subset=["raw_text", "ready_text"])

    df["text"] = df.apply(
        lambda r: tokenizer.apply_chat_template(
            build_messages(r, args.default_system),
            tokenize=False,
            add_generation_prompt=False,
        ),
        axis=1,
    )

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

    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    filtered = {k: v for k, v in trainer_kwargs.items() if k in sft_sig}
    if "tokenizer" in sft_sig:
        filtered["tokenizer"] = tokenizer
    elif "processing_class" in sft_sig:
        filtered["processing_class"] = tokenizer

    trainer = None
    try:
        trainer = SFTTrainer(**filtered)
    except Exception as e:
        print(f"SFTTrainer fallback to Trainer: {e}")

        def tok_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length, padding=False)

        train_tok = train_ds.map(tok_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c != "text"])
        if "text" in train_tok.column_names:
            train_tok = train_tok.remove_columns(["text"])

        eval_tok = None
        if eval_ds is not None:
            eval_tok = eval_ds.map(tok_fn, batched=True, remove_columns=[c for c in eval_ds.column_names if c != "text"])
            if "text" in eval_tok.column_names:
                eval_tok = eval_tok.remove_columns(["text"])

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        t_kwargs = {
            "model": model,
            "args": TrainingArguments(**ta_kwargs),
            "train_dataset": train_tok,
            "eval_dataset": eval_tok,
            "data_collator": collator,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if eval_tok is not None else [],
        }
        tr_sig = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in tr_sig:
            t_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in tr_sig:
            t_kwargs["processing_class"] = tokenizer

        trainer = Trainer(**t_kwargs)

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
