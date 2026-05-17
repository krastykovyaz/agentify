#!/usr/bin/env python3
"""
Generate synthetic samples with target task distribution using universal LoRA model.

Input (seed CSV): raw_text, ready_text, optional task/domain/system
Output CSV: raw_text, ready_text, task, source

Purpose: test if models_universal_v1 can produce usable synthetic data while keeping requested distribution.
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = ["summarization", "telegram_post", "qa", "extraction"]

PROMPT_BY_TASK = {
    "summarization": "Сделай краткую фактическую выжимку текста.",
    "telegram_post": "Сделай пост для Telegram по заметке: заголовок + 1-3 абзаца.",
    "qa": "Ответь на вопрос по контексту кратко и по фактам.",
    "extraction": "Извлеки структурированные поля в JSON: title, category, location, date, price, contacts, summary.",
}


def parse_distribution(s: str):
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k in TASKS:
            out[k] = max(0.0, float(v))
    if not out:
        return {t: 1.0 / len(TASKS) for t in TASKS}
    total = sum(out.values())
    for t in TASKS:
        out.setdefault(t, 0.0)
    return {k: v / total for k, v in out.items()}


def weighted_task_sequence(n: int, dist: dict):
    seq = []
    for t in TASKS:
        seq.extend([t] * int(round(n * dist[t])))
    while len(seq) < n:
        seq.append(random.choices(TASKS, weights=[dist[t] for t in TASKS], k=1)[0])
    if len(seq) > n:
        seq = seq[:n]
    random.shuffle(seq)
    return seq


def clean_output(task: str, text: str) -> str:
    t = text.strip()
    markers = ["\nПользователь:", "\nАссистент:", "<|im_start|>user", "<|im_start|>assistant"]
    cuts = [t.find(m) for m in markers if t.find(m) != -1]
    if cuts:
        t = t[: min(cuts)]
    if task == "extraction":
        # keep json-ish part if present
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0)
    return t.strip()


def read_seed_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            raw = (row.get("raw_text") or "").strip()
            if raw:
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--seed-csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--distribution", default="summarization=0.35,telegram_post=0.25,qa=0.25,extraction=0.15")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--system", default="Ты универсальный ассистент. Выполняй строго указанную задачу.")
    args = ap.parse_args()

    dist = parse_distribution(args.distribution)
    tasks = weighted_task_sequence(args.num_samples, dist)
    seed_rows = read_seed_rows(Path(args.seed_csv))
    if not seed_rows:
        raise SystemExit("No seed rows found")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    out_rows = []
    got = Counter()

    for task in tasks:
        seed = random.choice(seed_rows)
        base_raw = (seed.get("raw_text") or "").strip()
        user_input = f"[{task}] {PROMPT_BY_TASK[task]}\n\nВход:\n{base_raw}"

        prompt = (
            f"<|im_start|>system\n{args.system}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.1,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = clean_output(task, gen)
        if not gen:
            continue

        out_rows.append({
            "raw_text": user_input,
            "ready_text": gen,
            "task": task,
            "source": "synthetic_universal_v1",
        })
        got[task] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "task", "source"])
        w.writeheader()
        w.writerows(out_rows)

    print("Done")
    print(f"saved: {out_path}")
    print(f"rows: {len(out_rows)}")
    print("target_distribution:")
    for t in TASKS:
        print(f"  {t}: {dist[t]:.3f}")
    print("actual_distribution:")
    total = max(1, len(out_rows))
    for t in TASKS:
        print(f"  {t}: {got[t]} ({got[t]/total:.3f})")


if __name__ == "__main__":
    main()
