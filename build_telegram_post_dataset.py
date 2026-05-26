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
    "Ты редактор Telegram-канала. На вход получаешь исходный текст, на выходе возвращаешь "
    "ровно один готовый пост для публикации в канал. Без вариантов, без рекомендаций, "
    "без пояснений, без саммари. Только финальный текст поста."
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


def strip_meta_sections(text: str) -> str:
    """Drop common assistant artifacts like variants/recommendations/summaries."""
    t = text.strip()
    lines = [x for x in t.splitlines()]
    banned = (
        "вариант 1", "вариант 2", "вариант 3",
        "рекомендац", "саммари", "summary",
        "пояснен", "объяснен", "примечан",
    )
    out = []
    for ln in lines:
        low = ln.lower().strip(" :.-")
        if any(b in low for b in banned):
            continue
        out.append(ln)
    t = "\n".join(out).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def make_input_from_text(source_text: str) -> str:
    return (
        "Сформируй готовый пост для Telegram-канала по исходному тексту. "
        "Верни только один финальный пост без вариантов, рекомендаций и саммари.\n\n"
        "Исходный текст:\n"
        f"{source_text}"
    )


def ollama_paraphrase(url: str, model: str, text: str, timeout: int = 180):
    prompt = (
        "Перепиши текст в формат готового поста для Telegram-канала. "
        "Сохрани факты и смысл. Верни только один финальный пост без вариантов, "
        "без рекомендаций и без саммари.\n\n"
        f"Текст:\n{text}"
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
    ap.add_argument("--strict-forward", action="store_true", default=True, help="Force single forward-ready post output")
    ap.add_argument("--no-strict-forward", dest="strict_forward", action="store_false")
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
        target = strip_meta_sections(post) if args.strict_forward else post
        if args.paraphrase_ratio > 0 and random.random() < args.paraphrase_ratio:
            ok = False
            for _ in range(args.paraphrase_retries):
                try:
                    cand = ollama_paraphrase(ollama_url, ollama_model, post)
                    if args.strict_forward:
                        cand = strip_meta_sections(cand)
                    if valid_post(cand, args.min_chars, args.max_chars):
                        target = cand
                        ok = True
                        break
                except Exception:
                    pass
                time.sleep(0.6)
            if not ok:
                # Keep original target if paraphrase fails
                target = strip_meta_sections(post) if args.strict_forward else post

        if args.strict_forward:
            # Remove explicit "multiple options" leftovers.
            low = target.lower()
            if ("вариант 1" in low and "вариант 2" in low) or ("рекомендац" in low) or ("саммари" in low):
                continue

        out.append({
            "raw_text": make_input_from_text(post),
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
