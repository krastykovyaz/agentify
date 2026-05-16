#!/usr/bin/env python3
"""
Clean persona/dialog SFT datasets to neutral-professional style.

Input CSV columns: raw_text, ready_text, optional system
Output CSV columns: raw_text, ready_text, system

Modes:
- drop: remove personal/romantic/offline-like replies
- rewrite (optional): rewrite flagged replies via Ollama into neutral support
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Conservative patterns that often indicate over-personal style
PERSONAL_PATTERNS = [
    r"\bлюблю\b",
    r"\bскучаю\b",
    r"\bрад тебя видеть\b",
    r"\bувидимся\b",
    r"\bвстретимся\b",
    r"\bя дома\b",
    r"\bдомой\b",
    r"\bноч[ьюи]\b",
    r"\bобнима(ю|ю тебя|шки)\b",
    r"\bцелую\b",
    r"\bсолнышк\b",
    r"\bзай\b",
    r"\bмалыш\b",
    r"🍌|😘|😍|💋|💘|💞|💗",
]

SERVICE_NOISE_PATTERNS = [
    r"\bвстаньте в очередь\b",
    r"\bпопробуйте еще раз\b",
    r"\bошибка\b",
    r"\bотменили покупку\b",
    r"\bтранзакц\b",
]

ROLE_MARKERS = [
    "\nПользователь:",
    "\nАссистент:",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "\nuser:",
    "\nassistant:",
]

NEUTRAL_SYSTEM = (
    "Ты нейтральный эмпатичный ассистент. Отвечай одной уместной репликой, "
    "без флирта, личных отношений и офлайн-договоренностей."
)


def clean_spaces(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trim_multi_turn(text: str) -> str:
    t = text.strip()
    cut = []
    for m in ROLE_MARKERS:
        p = t.lower().find(m.lower())
        if p != -1:
            cut.append(p)
    if cut:
        t = t[: min(cut)]
    t = re.sub(r"^\s*(Ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return clean_spaces(t)


def is_flagged(text: str) -> bool:
    low = text.lower()
    for p in PERSONAL_PATTERNS + SERVICE_NOISE_PATTERNS:
        if re.search(p, low, flags=re.IGNORECASE):
            return True
    return False


def should_drop_by_length(text: str, min_words: int, max_chars: int) -> bool:
    if not text:
        return True
    if len(text) > max_chars:
        return True
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < min_words:
        return True
    return False


def call_ollama(url: str, model: str, prompt: str, timeout: int, temperature: float) -> str:
    r = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def rewrite_neutral(
    raw_text: str,
    bad_reply: str,
    ollama_url: str,
    ollama_model: str,
    retries: int,
    timeout: int,
    temperature: float,
    min_words: int,
    max_chars: int,
) -> str:
    prompt = (
        "Перепиши ответ ассистента в нейтрально-эмпатичном стиле.\n"
        "Правила:\n"
        "- одна реплика\n"
        "- без флирта, романтики, личных отношений\n"
        "- без офлайн-приглашений и бытовых деталей\n"
        "- по смыслу отвечай на контекст\n\n"
        f"Контекст:\n{raw_text}\n\n"
        f"Плохой ответ:\n{bad_reply}\n\n"
        "Новый ответ (только ответ):"
    )

    for _ in range(retries):
        try:
            out = trim_multi_turn(call_ollama(ollama_url, ollama_model, prompt, timeout, temperature))
            if not should_drop_by_length(out, min_words=min_words, max_chars=max_chars) and not is_flagged(out):
                return out
        except Exception:
            pass
        time.sleep(0.6)
    return ""


def process_rows(
    rows: List[Dict[str, str]],
    min_words: int,
    max_chars: int,
    rewrite_flagged: bool,
    ollama_url: str,
    ollama_model: str,
    retries: int,
    timeout: int,
    temperature: float,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    out = []
    seen = set()
    stats = {
        "input": 0,
        "kept": 0,
        "dropped_empty_or_len": 0,
        "flagged_total": 0,
        "flagged_rewritten": 0,
        "flagged_dropped": 0,
        "duplicates_dropped": 0,
    }

    bar = tqdm(rows, desc="neutral-clean", unit="row")
    for row in bar:
        stats["input"] += 1

        raw_text = clean_spaces(str(row.get("raw_text", "")))
        ready_text = trim_multi_turn(str(row.get("ready_text", "")))
        system = str(row.get("system", "")).strip() or NEUTRAL_SYSTEM

        if not raw_text or should_drop_by_length(ready_text, min_words=min_words, max_chars=max_chars):
            stats["dropped_empty_or_len"] += 1
            continue

        flagged = is_flagged(ready_text)
        if flagged:
            stats["flagged_total"] += 1
            if rewrite_flagged:
                new_reply = rewrite_neutral(
                    raw_text=raw_text,
                    bad_reply=ready_text,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    retries=retries,
                    timeout=timeout,
                    temperature=temperature,
                    min_words=min_words,
                    max_chars=max_chars,
                )
                if new_reply:
                    ready_text = new_reply
                    stats["flagged_rewritten"] += 1
                else:
                    stats["flagged_dropped"] += 1
                    continue
            else:
                stats["flagged_dropped"] += 1
                continue

        key = (raw_text, ready_text)
        if key in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(key)

        out.append({"raw_text": raw_text, "ready_text": ready_text, "system": system})
        stats["kept"] += 1
        bar.set_postfix(kept=stats["kept"], flagged=stats["flagged_total"], rewritten=stats["flagged_rewritten"])

    return out, stats


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Clean dialog SFT dataset into neutral style")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=260)
    ap.add_argument("--rewrite-flagged", action="store_true", default=False)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = process all rows")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N rows before processing")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.35)
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    cleaned, stats = process_rows(
        rows=rows,
        min_words=args.min_words,
        max_chars=args.max_chars,
        rewrite_flagged=args.rewrite_flagged,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        retries=args.max_retries,
        timeout=args.timeout,
        temperature=args.temperature,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        w.writeheader()
        w.writerows(cleaned)

    print("Done")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"saved: {out_path}")
    if args.rewrite_flagged:
        print(f"ollama_model: {ollama_model}")


if __name__ == "__main__":
    main()
