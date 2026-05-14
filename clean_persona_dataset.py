#!/usr/bin/env python3
"""
Clean persona dialogue dataset for stable assistant-only SFT.

Input CSV columns: raw_text, ready_text, optional system
Output CSV columns: raw_text, ready_text, system
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

BAD_PATTERNS = [
    r"\b(секс|сексу|эрот|порно|nude|nsfw)\b",
    r"\b(каждую ночь|хочу тебя|возбуд|banana|🍌)\b",
    r"\b(встаньте в очередь|попробуйте еще раз|отменили покупку|ошибка оплаты)\b",
]

ROLE_MARKERS = [
    "\nПользователь:",
    "\nАссистент:",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "\nuser:",
    "\nassistant:",
]


def merge_consecutive_turns(raw_text: str) -> str:
    """
    Merge consecutive lines from the same speaker:
    Пользователь: ...
    Пользователь: ...
    Ассистент: ...
    ->
    Пользователь: ... ...
    Ассистент: ...
    """
    lines = [ln.strip() for ln in str(raw_text).splitlines() if ln.strip()]
    merged: List[Tuple[str, str]] = []
    for ln in lines:
        m = re.match(r"^(Пользователь|Ассистент)\s*:\s*(.*)$", ln, flags=re.IGNORECASE)
        if not m:
            # Keep non-labeled lines by attaching to previous utterance if possible.
            if merged:
                role, text = merged[-1]
                merged[-1] = (role, f"{text} {ln}".strip())
            continue
        role = m.group(1).capitalize()
        text = m.group(2).strip()
        if not text:
            continue
        if merged and merged[-1][0].lower() == role.lower():
            prev_role, prev_text = merged[-1]
            merged[-1] = (prev_role, f"{prev_text} {text}".strip())
        else:
            merged.append((role, text))
    return "\n".join([f"{r}: {t}" for r, t in merged]).strip()


def cut_at_role_marker(text: str) -> str:
    t = text.strip()
    cut_positions = []
    for marker in ROLE_MARKERS:
        pos = t.lower().find(marker.lower())
        if pos != -1:
            cut_positions.append(pos)
    if cut_positions:
        t = t[: min(cut_positions)]
    t = re.sub(r"^\s*(ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def looks_bad(text: str) -> bool:
    low = text.lower()
    for p in BAD_PATTERNS:
        if re.search(p, low, flags=re.IGNORECASE):
            return True
    return False


def too_repetitive(text: str, max_repeat_ratio: float = 0.45) -> bool:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 6:
        return False
    uniq = len(set(words))
    ratio = 1.0 - (uniq / max(1, len(words)))
    return ratio > max_repeat_ratio


def is_valid_reply(text: str, min_words: int, max_chars: int) -> bool:
    if not text:
        return False
    if len(text) > max_chars:
        return False
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    if len(words) < min_words:
        return False
    if looks_bad(text):
        return False
    if too_repetitive(text):
        return False
    return True


def clean_rows(
    rows: List[Dict[str, str]],
    min_words: int,
    max_chars: int,
    dedup: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    out: List[Dict[str, str]] = []
    stats = {
        "input": 0,
        "dropped_empty": 0,
        "dropped_invalid": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }
    seen = set()

    for row in rows:
        stats["input"] += 1
        raw = merge_consecutive_turns(str(row.get("raw_text", "")))
        raw = normalize_spaces(raw.replace("\n", " \n ").replace(" \n  ", "\n")).replace(" \n ", "\n")
        ready = cut_at_role_marker(str(row.get("ready_text", "")))
        ready = normalize_spaces(ready)
        system = str(row.get("system", "")).strip()

        if not raw or not ready:
            stats["dropped_empty"] += 1
            continue

        if not is_valid_reply(ready, min_words=min_words, max_chars=max_chars):
            stats["dropped_invalid"] += 1
            continue

        key = (raw, ready)
        if dedup and key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)

        out.append({"raw_text": raw, "ready_text": ready, "system": system})

    stats["kept"] = len(out)
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean persona SFT CSV")
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=280)
    ap.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    cleaned, stats = clean_rows(
        rows,
        min_words=args.min_words,
        max_chars=args.max_chars,
        dedup=not args.no_dedup,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        w.writeheader()
        w.writerows(cleaned)

    print("Done")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
