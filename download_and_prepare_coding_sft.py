#!/usr/bin/env python3
"""
Download + normalize open coding datasets into a single SFT CSV.

Output schema:
  task,instruction,input,output,source,language

Example:
  python3 download_and_prepare_coding_sft.py \
    --output /Users/aleksandr.koviazin/Desktop/agentify/datasets/coding_sft.csv \
    --max-per-dataset 120000 \
    --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_dataset, concatenate_datasets


@dataclass
class Row:
    task: str
    instruction: str
    input: str
    output: str
    source: str
    language: str


def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def guess_language(text: str) -> str:
    t = text.lower()
    if "def " in t and "import " in t:
        return "python"
    if "public static void main" in t or "system.out" in t:
        return "java"
    if "#include" in t or "std::" in t:
        return "cpp"
    if "function " in t or "console.log" in t or "=>" in t:
        return "javascript"
    if "fn " in t and "let " in t:
        return "rust"
    if "select " in t and " from " in t:
        return "sql"
    return "unknown"


def safe_get(example: Dict, keys: List[str]) -> str:
    for k in keys:
        if k in example and example[k] is not None:
            return str(example[k])
    return ""


def normalize_row(instruction: str, inp: str, out: str, source: str) -> Optional[Row]:
    instruction = clean_text(instruction)
    inp = clean_text(inp)
    out = clean_text(out)

    if not instruction or not out:
        return None
    if len(out) < 20:
        return None
    if len(out) > 20000:
        return None

    # Light safety filters for broken/placeholder data
    bad_markers = ["lorem ipsum", "todo", "coming soon", "n/a"]
    low = (instruction + "\n" + out).lower()
    if any(b in low for b in bad_markers):
        return None

    lang = guess_language(inp + "\n" + out)
    return Row(
        task="coding",
        instruction=instruction,
        input=inp,
        output=out,
        source=source,
        language=lang,
    )


def map_evol_instruct_code(ds: Dataset, source: str) -> Iterable[Row]:
    for ex in ds:
        instr = safe_get(ex, ["instruction", "prompt", "question"])
        inp = safe_get(ex, ["input", "context"])
        out = safe_get(ex, ["output", "response", "answer", "completion"])
        row = normalize_row(instr, inp, out, source)
        if row:
            yield row


def map_code_instructions_filtered(ds: Dataset, source: str) -> Iterable[Row]:
    for ex in ds:
        # Common formats in filtered instruction datasets
        instr = safe_get(ex, ["instruction", "prompt", "question"])
        inp = safe_get(ex, ["input", "context"])
        out = safe_get(ex, ["output", "response", "answer", "completion"])
        # fallback: chat-like structures
        if not out:
            conv = ex.get("conversations")
            if isinstance(conv, list) and len(conv) >= 2:
                instr = instr or str(conv[0].get("value", ""))
                out = str(conv[1].get("value", ""))
        row = normalize_row(instr, inp, out, source)
        if row:
            yield row


def map_openhermes_style(ds: Dataset, source: str) -> Iterable[Row]:
    for ex in ds:
        # fallback mapper for instruction-style sets with variant fields
        instr = safe_get(ex, ["instruction", "prompt", "query", "question"])
        inp = safe_get(ex, ["input", "context"])
        out = safe_get(ex, ["output", "response", "answer", "completion"])
        row = normalize_row(instr, inp, out, source)
        if row:
            yield row


def row_hash(r: Row) -> str:
    key = "\n".join([r.instruction[:2000], r.input[:2000], r.output[:4000]]).lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def dedupe(rows: List[Row]) -> List[Row]:
    seen = set()
    out: List[Row] = []
    for r in rows:
        h = row_hash(r)
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out


def stratified_split(rows: List[Row], val_ratio: float, seed: int) -> Tuple[List[Row], List[Row]]:
    if val_ratio <= 0:
        return rows, []
    by_lang: Dict[str, List[Row]] = {}
    for r in rows:
        by_lang.setdefault(r.language, []).append(r)

    import random

    rnd = random.Random(seed)
    train, val = [], []
    for _, grp in by_lang.items():
        rnd.shuffle(grp)
        n_val = max(1, int(len(grp) * val_ratio)) if len(grp) >= 20 else 0
        val.extend(grp[:n_val])
        train.extend(grp[n_val:])
    rnd.shuffle(train)
    rnd.shuffle(val)
    return train, val


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["task", "instruction", "input", "output", "source", "language"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def load_any_dataset(dataset_id: str, split: str):
    return load_dataset(dataset_id, split=split)


def maybe_take(ds: Dataset, max_rows: int, seed: int) -> Dataset:
    if max_rows <= 0 or len(ds) <= max_rows:
        return ds
    ds = ds.shuffle(seed=seed)
    return ds.select(range(max_rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and prepare open coding datasets for Gemma SFT")
    ap.add_argument("--output", required=True, help="Output train csv path")
    ap.add_argument("--val-output", default="", help="Optional val csv path")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-dataset", type=int, default=120000)

    # Default open sources (can be overridden)
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="HF dataset spec: dataset_id[:split[:mapper]]. mapper in {evol,filtered,generic}",
    )
    args = ap.parse_args()

    default_specs = [
        "artificial-citizen/Evol-Instruct-Code:train:evol",
        "vikp/code_instructions_filtered:train:filtered",
        "HydraLM/Evol-Instruct-Code-80k-v1-standardized:train:evol",
    ]
    specs = args.dataset or default_specs

    mapper_fn = {
        "evol": map_evol_instruct_code,
        "filtered": map_code_instructions_filtered,
        "generic": map_openhermes_style,
    }

    all_rows: List[Row] = []
    for spec in specs:
        parts = spec.split(":")
        dataset_id = parts[0]
        split = parts[1] if len(parts) >= 2 and parts[1] else "train"
        mapper_key = parts[2] if len(parts) >= 3 and parts[2] else "generic"

        if mapper_key not in mapper_fn:
            raise ValueError(f"Unknown mapper '{mapper_key}' in spec '{spec}'")

        print(f"loading {dataset_id} [{split}] ...")
        ds = load_any_dataset(dataset_id, split)

        # Handle DatasetDict-like concatenation if split returns multiple parts
        if isinstance(ds, dict):
            ds = concatenate_datasets(list(ds.values()))

        ds = maybe_take(ds, args.max_per_dataset, args.seed)
        print(f"  rows selected: {len(ds)}")

        rows_before = len(all_rows)
        for r in mapper_fn[mapper_key](ds, dataset_id):
            all_rows.append(r)
        print(f"  normalized added: {len(all_rows) - rows_before}")

    print(f"total normalized before dedupe: {len(all_rows)}")
    all_rows = dedupe(all_rows)
    print(f"total after dedupe: {len(all_rows)}")

    train_rows, val_rows = stratified_split(all_rows, val_ratio=args.val_ratio, seed=args.seed)

    out_train = Path(args.output)
    out_val = Path(args.val_output) if args.val_output else out_train.with_name(out_train.stem + ".val.csv")

    write_csv(out_train, train_rows)
    write_csv(out_val, val_rows)

    print(f"saved train: {out_train} ({len(train_rows)} rows)")
    print(f"saved val:   {out_val} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
