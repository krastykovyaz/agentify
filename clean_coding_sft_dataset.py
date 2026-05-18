#!/usr/bin/env python3
"""
Clean coding SFT dataset (CSV) for stable Gemma training.

Input columns expected:
- instruction
- output
Optional:
- input, source, language, task, system

Example:
python3 clean_coding_sft_dataset.py \
  --input /home/aleksandr.koviazin/kovyaz/agentify/datasets/coding_sft_train.csv \
  --output /home/aleksandr.koviazin/kovyaz/agentify/datasets/coding_sft_train_clean.csv
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Clean coding SFT CSV")
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", required=True, help="Output cleaned CSV path")
    p.add_argument("--report", default="", help="Optional JSON report path")

    p.add_argument("--min-instruction-len", type=int, default=80)
    p.add_argument("--max-instruction-len", type=int, default=4000)
    p.add_argument("--min-output-len", type=int, default=60)
    p.add_argument("--max-output-len", type=int, default=5000)

    p.add_argument("--drop-bad-markers", action="store_true", default=True)
    p.add_argument("--no-drop-bad-markers", dest="drop_bad_markers", action="store_false")

    p.add_argument("--require-code-if-output-gt", type=int, default=1200)
    p.add_argument("--drop-if-no-code-and-output-gt", type=int, default=3000)

    p.add_argument("--dedupe", action="store_true", default=True)
    p.add_argument("--no-dedupe", dest="dedupe", action="store_false")

    p.add_argument("--keep-ratio", type=float, default=1.0, help="Random keep ratio after filters, 0..1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compile_patterns():
    bad_markers = re.compile(r"(?i)\\b(todo|lorem ipsum|coming soon|n/?a|placeholder|fixme|tbd)\\b")
    code_hint = re.compile(
        r"```|\\bdef\\s+|\\bclass\\s+|#include|\\bSELECT\\b|\\bINSERT\\b|\\bUPDATE\\b|\\bDELETE\\b|"
        r"\\bpublic\\s+class\\b|\\bfunction\\s+|\\bimport\\s+|\\bfrom\\s+|\\bconst\\s+|\\blet\\s+|"
        r"\\btry\\s*\\{|\\bcatch\\s*\\(|\\bif\\s*\\(|\\bfor\\s*\\(|\\bwhile\\s*\\(|"
        r"\\breturn\\b|=>|::|\\{\\s*$",
        flags=re.IGNORECASE,
    )
    return bad_markers, code_hint


def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def quantile(vals, q):
    if not vals:
        return 0
    v = sorted(vals)
    idx = int(q * (len(v) - 1))
    return v[idx]


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    rpt = Path(args.report) if args.report else dst.with_suffix(".report.json")

    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    bad_markers, code_hint = compile_patterns()

    import random

    rng = random.Random(args.seed)

    total = 0
    kept = 0
    dropped = 0
    reasons = Counter()

    instr_lens = []
    out_lens = []

    seen = set()

    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8-sig", newline="") as f_in, dst.open("w", encoding="utf-8-sig", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = reader.fieldnames or []
        need = {"instruction", "output"}
        if not need.issubset(set(fields)):
            raise SystemExit(f"CSV must contain columns {need}, got: {fields}")

        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()

        for row in reader:
            total += 1

            ins = clean_text(str(row.get("instruction", "")))
            out = clean_text(str(row.get("output", "")))
            inp = clean_text(str(row.get("input", ""))) if "input" in row else ""

            row["instruction"] = ins
            row["output"] = out
            if "input" in row:
                row["input"] = inp

            li = len(ins)
            lo = len(out)

            if li < args.min_instruction_len:
                dropped += 1
                reasons["instruction_too_short"] += 1
                continue
            if li > args.max_instruction_len:
                dropped += 1
                reasons["instruction_too_long"] += 1
                continue

            if lo < args.min_output_len:
                dropped += 1
                reasons["output_too_short"] += 1
                continue
            if lo > args.max_output_len:
                dropped += 1
                reasons["output_too_long"] += 1
                continue

            if args.drop_bad_markers and (bad_markers.search(ins) or bad_markers.search(out)):
                dropped += 1
                reasons["bad_markers"] += 1
                continue

            has_code = bool(code_hint.search(out))

            if lo > args.require_code_if_output_gt and not has_code:
                dropped += 1
                reasons["long_output_no_code"] += 1
                continue

            if lo > args.drop_if_no_code_and_output_gt and not has_code:
                dropped += 1
                reasons["very_long_output_no_code"] += 1
                continue

            if args.dedupe:
                key = (ins, inp, out)
                if key in seen:
                    dropped += 1
                    reasons["duplicate"] += 1
                    continue
                seen.add(key)

            if args.keep_ratio < 1.0 and rng.random() > args.keep_ratio:
                dropped += 1
                reasons["downsampled"] += 1
                continue

            writer.writerow(row)
            kept += 1
            instr_lens.append(li)
            out_lens.append(lo)

    report = {
        "input": str(src),
        "output": str(dst),
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "keep_ratio_actual": (kept / total) if total else 0,
        "drop_reasons": dict(reasons),
        "instruction_len": {
            "avg": (sum(instr_lens) / len(instr_lens)) if instr_lens else 0,
            "p50": quantile(instr_lens, 0.50),
            "p95": quantile(instr_lens, 0.95),
            "max": max(instr_lens) if instr_lens else 0,
        },
        "output_len": {
            "avg": (sum(out_lens) / len(out_lens)) if out_lens else 0,
            "p50": quantile(out_lens, 0.50),
            "p95": quantile(out_lens, 0.95),
            "max": max(out_lens) if out_lens else 0,
        },
        "params": vars(args),
    }

    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {dst}")
    print(f"report: {rpt}")
    print(f"total: {total} kept: {kept} dropped: {dropped}")


if __name__ == "__main__":
    main()
