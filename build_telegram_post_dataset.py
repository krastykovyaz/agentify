#!/usr/bin/env python3
"""Build telegram_post SFT dataset from real posts.

Input CSV columns: text (required), optionally channel/date.
Output CSV columns: raw_text, ready_text, system

Strategy:
- Use real post as target (ready_text)
- Create source note/instruction as input (raw_text)
- Optional light paraphrase via Ollama (low ratio), default off
"""

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SYSTEM_PROMPT = (
    "Ты редактор Telegram-канала. Пиши информативный пост на русском: "
    "четкий заголовок, 1-3 абзаца, без выдумок, без кликбейта."
)


def clean_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def valid_post(t: str, min_chars: int, max_chars: int) -> bool:
    if not t:
        return False
    if len(t) < min_chars or len(t) > max_chars:
        return False
    if t.count("\n") < 1:
        return False
    return True


def make_input_from_post(post: str) -> str:
    first_line = post.splitlines()[0][:140]
    return (
        "Подготовь пост для Telegram-канала на основе заметки. "
        "Сохрани факты, сделай подачу живой и читабельной.\n\n"
        f"Черновая заметка: {first_line}"
    )


def ollama_paraphrase(url: str, model: str, text: str, timeout: int = 180):
    prompt = (
        "Перепиши пост для Telegram, сохрани факты и смысл. "
        "Верни только готовый пост.\n\n"
        f"Пост:\n{text}"
    )
    r = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.45}},
        timeout=timeout,
    )
    r.raise_for_status()
    return clean_text((r.json().get("response") or ""))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--min-chars", type=int, default=140)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--paraphrase-ratio", type=float, default=0.0, help="0..1 fraction of rows to paraphrase via Ollama")
    ap.add_argument("--paraphrase-retries", type=int, default=3)
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    rows = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            txt = clean_text(row.get(args.text_column, ""))
            if valid_post(txt, args.min_chars, args.max_chars):
                rows.append(txt)

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out = []
    for post in tqdm(rows, desc="build-telegram-post", unit="row"):
        target = post
        if args.paraphrase_ratio > 0 and random.random() < args.paraphrase_ratio:
            ok = False
            for _ in range(args.paraphrase_retries):
                try:
                    cand = ollama_paraphrase(ollama_url, ollama_model, post)
                    if valid_post(cand, args.min_chars, args.max_chars):
                        target = cand
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(0.6)
            if not ok:
                # Keep original target if paraphrase fails
                target = post

        out.append({
            "raw_text": make_input_from_post(post),
            "ready_text": target,
            "system": SYSTEM_PROMPT,
        })

    # dedup
    uniq = {}
    for x in out:
        uniq[(x["raw_text"], x["ready_text"])] = x
    out = list(uniq.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        w.writeheader()
        w.writerows(out)

    print(f"saved: {output}")
    print(f"rows: {len(out)}")


if __name__ == "__main__":
    main()
