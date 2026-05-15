#!/usr/bin/env python3
"""
Build SFT dataset for text augmentation agent.

Task: expand text while preserving facts.
Output CSV columns: raw_text, ready_text, system
"""

import argparse
import csv
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SYSTEM_PROMPT = (
    "Ты текстовый аугментатор. Расширяй текст без выдумки новых фактов. "
    "Сохраняй смысл, имена, числа, даты, ссылки и географию. "
    "Возвращай только итоговый текст."
)

MODE_CFG = {
    "light": (1.3, 1.8),
    "medium": (1.8, 2.8),
    "deep": (2.8, 4.0),
}

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|\b\d{4}\b")
NUM_RE = re.compile(r"\b\d+[\d.,]*\b")
URL_RE = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"(?:\+?7|8)\D*\d{3}\D*\d{3}\D*\d{2}\D*\d{2}")
CAP_WORD_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")


def clean_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def split_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += step
    return out


def read_inputs(path: Path, text_column: str) -> List[str]:
    if path.suffix.lower() in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                t = clean_text(str(row.get(text_column, "")))
                if t:
                    rows.append(t)
        return rows
    raise ValueError("Supported input formats: .txt, .md, .csv")


def extract_facts(text: str) -> Dict[str, set]:
    return {
        "dates": set(DATE_RE.findall(text)),
        "nums": set(NUM_RE.findall(text)),
        "urls": set(URL_RE.findall(text)),
        "phones": set(PHONE_RE.findall(text)),
        "caps": set(CAP_WORD_RE.findall(text)),
    }


def missing_fact_ratio(src: Dict[str, set], out: Dict[str, set], keys: List[str]) -> float:
    total = 0
    miss = 0
    for k in keys:
        for v in src[k]:
            total += 1
            if v not in out[k]:
                miss += 1
    if total == 0:
        return 0.0
    return miss / total


def has_forbidden_new_numbers(src: Dict[str, set], out: Dict[str, set], threshold: int = 2) -> bool:
    new_nums = [n for n in out["nums"] if n not in src["nums"]]
    return len(new_nums) > threshold


def build_instruction(mode: str, src_text: str) -> str:
    lo, hi = MODE_CFG[mode]
    return (
        f"Расширь текст в режиме {mode} (примерно x{lo:.1f}-x{hi:.1f} длины). "
        "Не добавляй новых фактов. Сохрани имена, цифры, даты, ссылки.\n\n"
        f"Исходный текст:\n{src_text}"
    )


def call_ollama(url: str, model: str, prompt: str, temperature: float, timeout: int) -> str:
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


def validate_augmented(src: str, out: str, mode: str, min_chars: int, max_chars: int) -> Tuple[bool, str]:
    if not out:
        return False, "empty"
    if len(out) < min_chars or len(out) > max_chars:
        return False, "length"

    ratio = len(out) / max(1, len(src))
    lo, hi = MODE_CFG[mode]
    if ratio < (lo * 0.9):
        return False, "too_short_for_mode"
    if ratio > (hi * 1.25):
        return False, "too_long_for_mode"

    src_f = extract_facts(src)
    out_f = extract_facts(out)

    # Important source facts should mostly survive
    miss_ratio = missing_fact_ratio(src_f, out_f, ["dates", "nums", "urls", "phones"])
    if miss_ratio > 0.5:
        return False, "fact_loss"

    # Too many new numeric facts often means hallucination
    if has_forbidden_new_numbers(src_f, out_f):
        return False, "new_numbers"

    return True, "ok"


def generate_one(
    src_text: str,
    mode: str,
    ollama_url: str,
    ollama_model: str,
    retries: int,
    temperature: float,
    timeout: int,
    min_chars: int,
    max_chars: int,
) -> Optional[Dict[str, str]]:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_instruction(mode, src_text)}\n\n"
        "Верни только итоговый расширенный текст."
    )

    for _ in range(retries):
        try:
            out = clean_text(call_ollama(ollama_url, ollama_model, prompt, temperature, timeout))
            ok, _reason = validate_augmented(src_text, out, mode, min_chars=min_chars, max_chars=max_chars)
            if ok:
                return {
                    "raw_text": build_instruction(mode, src_text),
                    "ready_text": out,
                    "system": SYSTEM_PROMPT,
                }
        except Exception:
            pass
        time.sleep(0.6)

    return None


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Build augmentation SFT dataset")
    ap.add_argument("--input", required=True, help=".txt/.md/.csv")
    ap.add_argument("--output", required=True, help="output CSV")
    ap.add_argument("--text-column", default="raw_text")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--source-min-chars", type=int, default=120)
    ap.add_argument("--source-max-chars", type=int, default=1200)
    ap.add_argument("--target-min-chars", type=int, default=180)
    ap.add_argument("--target-max-chars", type=int, default=3500)
    ap.add_argument("--modes", default="light,medium,deep", help="comma list")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODE_CFG]
    if not modes:
        raise SystemExit("No valid modes. Use light,medium,deep")

    src = read_inputs(Path(args.input), args.text_column)
    units: List[str] = []
    for t in src:
        for c in split_chunks(t, args.chunk_size, args.overlap):
            c = clean_text(c)
            if args.source_min_chars <= len(c) <= args.source_max_chars:
                units.append(c)

    random.shuffle(units)

    samples = []
    skipped = 0
    total_jobs = len(units) * len(modes)
    pbar = tqdm(total=total_jobs, desc="build-augmentation", unit="sample")

    for text in units:
        for mode in modes:
            row = generate_one(
                src_text=text,
                mode=mode,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                retries=args.max_retries,
                temperature=args.temperature,
                timeout=args.timeout,
                min_chars=args.target_min_chars,
                max_chars=args.target_max_chars,
            )
            if row is None:
                skipped += 1
            else:
                samples.append(row)
            pbar.update(1)
            pbar.set_postfix(ok=len(samples), skipped=skipped)
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
            time.sleep(args.sleep)
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

    # dedup
    uniq = {}
    for s in samples:
        uniq[(s["raw_text"], s["ready_text"])] = s
    samples = list(uniq.values())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        w.writeheader()
        w.writerows(samples)

    print("Done")
    print(f"model: {ollama_model}")
    print(f"rows: {len(samples)}")
    print(f"skipped: {skipped}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
