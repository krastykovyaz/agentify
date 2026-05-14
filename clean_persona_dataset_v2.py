#!/usr/bin/env python3
"""
Stronger cleaner for persona dialogue SFT datasets.

Adds v2 heuristics:
- merge consecutive same-speaker turns in raw_text
- trim multi-turn leakage in ready_text
- drop toxic/sexual/service-noise replies
- drop too short/too long/repetitive replies
- relevance filter: reply must weakly align with last user turn
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

BAD_PATTERNS = [
    r"\b(секс|сексу|эрот|порно|nude|nsfw|хочу тебя|каждую ночь|возбуд)\b",
    r"\b(встаньте в очередь|попробуйте еще раз|отменили покупку|ошибка оплаты|транзакц|карта|оплата)\b",
]

ROLE_MARKERS = [
    "\nПользователь:",
    "\nАссистент:",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "\nuser:",
    "\nassistant:",
]

STOPWORDS = {
    "и","в","во","не","что","он","она","оно","мы","вы","ты","я","на","с","со","за","по","от",
    "до","к","ко","о","об","а","но","или","если","то","же","ли","да","нет","ну","это","как",
    "так","у","из","для","под","над","при","бы","был","была","были","есть","еще","ещё","уже",
    "мне","тебе","тебя","его","ее","её","их","мой","моя","мои","твой","твоя","твои","просто",
}

EMPATHY_HINTS = {
    "понимаю","рядом","поддерж","обнимаю","держись","справишься","всё","все","нормально","отдохни",
    "ты","тебе","тебя","давай","вдох","выдох","помогу","с тобой",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def merge_consecutive_turns(raw_text: str) -> str:
    lines = [ln.strip() for ln in str(raw_text).splitlines() if ln.strip()]
    merged: List[Tuple[str, str]] = []
    for ln in lines:
        m = re.match(r"^(Пользователь|Ассистент)\s*:\s*(.*)$", ln, flags=re.IGNORECASE)
        if not m:
            if merged:
                role, text = merged[-1]
                merged[-1] = (role, f"{text} {ln}".strip())
            continue
        role = m.group(1).capitalize()
        text = m.group(2).strip()
        if not text:
            continue
        if merged and merged[-1][0].lower() == role.lower():
            r0, t0 = merged[-1]
            merged[-1] = (r0, f"{t0} {text}".strip())
        else:
            merged.append((role, text))
    return "\n".join([f"{r}: {t}" for r, t in merged]).strip()


def cut_at_role_marker(text: str) -> str:
    t = str(text).strip()
    cut_positions = []
    for marker in ROLE_MARKERS:
        pos = t.lower().find(marker.lower())
        if pos != -1:
            cut_positions.append(pos)
    if cut_positions:
        t = t[: min(cut_positions)]
    t = re.sub(r"^\s*(ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def looks_bad(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low, flags=re.IGNORECASE) for p in BAD_PATTERNS)


def too_repetitive(text: str, max_repeat_ratio: float = 0.45) -> bool:
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < 6:
        return False
    uniq = len(set(words))
    ratio = 1.0 - (uniq / max(1, len(words)))
    return ratio > max_repeat_ratio


def extract_last_user_text(raw_text: str) -> str:
    last = ""
    for ln in str(raw_text).splitlines():
        m = re.match(r"^Пользователь\s*:\s*(.*)$", ln.strip(), flags=re.IGNORECASE)
        if m:
            last = m.group(1).strip()
    return last


def tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower(), flags=re.UNICODE)
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS]


def is_relevant_reply(last_user_text: str, reply_text: str, min_overlap: int) -> bool:
    # If user text is too short, allow empathy-only replies.
    user_toks = set(tokens(last_user_text))
    reply_toks = set(tokens(reply_text))

    if len(user_toks) <= 2:
        if any(h in reply_text.lower() for h in EMPATHY_HINTS):
            return True

    overlap = user_toks.intersection(reply_toks)
    if len(overlap) >= min_overlap:
        return True

    # Allow direct supportive phrasing for distress-like prompts.
    distress = any(k in last_user_text.lower() for k in ["переж", "тяжел", "страш", "трев", "устал", "плохо"])
    if distress and any(h in reply_text.lower() for h in EMPATHY_HINTS):
        return True

    return False


def clean_rows(
    rows: List[Dict[str, str]],
    min_words: int,
    max_chars: int,
    min_overlap: int,
    dedup: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    out: List[Dict[str, str]] = []
    seen = set()
    stats = {
        "input": 0,
        "dropped_empty": 0,
        "dropped_invalid": 0,
        "dropped_irrelevant": 0,
        "dropped_duplicate": 0,
        "kept": 0,
    }

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

        words = re.findall(r"\w+", ready, flags=re.UNICODE)
        if len(words) < min_words or len(ready) > max_chars or looks_bad(ready) or too_repetitive(ready):
            stats["dropped_invalid"] += 1
            continue

        last_user = extract_last_user_text(raw)
        if not is_relevant_reply(last_user, ready, min_overlap=min_overlap):
            stats["dropped_irrelevant"] += 1
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
    ap = argparse.ArgumentParser(description="Clean persona SFT CSV (v2)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=240)
    ap.add_argument("--min-overlap", type=int, default=1)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    cleaned, stats = clean_rows(
        rows=rows,
        min_words=args.min_words,
        max_chars=args.max_chars,
        min_overlap=args.min_overlap,
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
