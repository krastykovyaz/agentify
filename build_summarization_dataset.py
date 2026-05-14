#!/usr/bin/env python3
"""
Build summarization SFT dataset from local text/csv using Ollama.

Output CSV schema:
- raw_text: source text chunk
- ready_text: summary
- system: optional instruction

Designed to be robust when JSON generation is unstable:
- multiple retries
- plain-text summary fallback
- skip bad generations
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm


DEFAULT_SYSTEM = (
    "Сделай краткую и фактическую выжимку текста на русском языке. "
    "Без выдумок, без новых фактов, 2-5 предложений."
)


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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


def clean_summary(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*(summary|суммаризация|кратко)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def valid_summary(s: str, min_chars: int, max_chars: int) -> bool:
    if not s:
        return False
    if len(s) < min_chars or len(s) > max_chars:
        return False
    if "{" in s or "}" in s:
        return False
    return True


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


def summarize_chunk(
    chunk: str,
    system_prompt: str,
    ollama_url: str,
    model: str,
    retries: int,
    timeout: int,
    temperature: float,
    min_chars: int,
    max_chars: int,
) -> Optional[str]:
    prompt = (
        f"{system_prompt}\n\n"
        "Верни только итоговый текст саммари, без JSON и без пояснений.\n\n"
        f"Текст:\n{chunk}\n"
    )

    for _ in range(retries):
        try:
            raw = call_ollama(ollama_url, model, prompt, temperature, timeout)
            sm = clean_summary(raw)
            if valid_summary(sm, min_chars=min_chars, max_chars=max_chars):
                return sm
        except Exception:
            pass
        time.sleep(0.6)
    return None


def read_inputs(input_path: Path, text_column: str) -> List[str]:
    if input_path.suffix.lower() in {".txt", ".md"}:
        return [input_path.read_text(encoding="utf-8")]

    if input_path.suffix.lower() == ".csv":
        rows = []
        with input_path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                value = (row.get(text_column) or "").strip()
                if value:
                    rows.append(value)
        return rows

    raise ValueError("Supported input formats: .txt, .md, .csv")


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Build summarization dataset with Ollama")
    ap.add_argument("--input", required=True, help="Path to .txt/.md/.csv source")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--text-column", default="raw_text", help="CSV source text column")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--chunk-size", type=int, default=1600)
    ap.add_argument("--overlap", type=int, default=180)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=650)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    input_path = Path(args.input)
    output_path = Path(args.output)

    src_texts = read_inputs(input_path, args.text_column)

    samples = []
    skipped = 0

    all_chunks = []
    for src in src_texts:
        all_chunks.extend(split_text(src, chunk_size=args.chunk_size, overlap=args.overlap))

    progress = tqdm(all_chunks, desc="summarization-build", unit="chunk")
    for chunk in progress:
            summary = summarize_chunk(
                chunk=chunk,
                system_prompt=args.system,
                ollama_url=ollama_url,
                model=model,
                retries=args.max_retries,
                timeout=args.timeout,
                temperature=args.temperature,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
            if summary is None:
                skipped += 1
                progress.set_postfix(ok=len(samples), skipped=skipped)
                continue
            samples.append(
                {
                    "raw_text": chunk.strip(),
                    "ready_text": summary.strip(),
                    "system": args.system,
                }
            )
            progress.set_postfix(ok=len(samples), skipped=skipped)
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
            time.sleep(args.sleep)

    # dedup
    uniq = {}
    for x in samples:
        uniq[(x["raw_text"], x["ready_text"])] = x
    samples = list(uniq.values())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        w.writeheader()
        w.writerows(samples)

    print("Done")
    print(f"model: {model}")
    print(f"saved: {output_path}")
    print(f"rows: {len(samples)}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
