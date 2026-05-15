#!/usr/bin/env python3
"""
Universal augmentation dataset builder with domain-aware rules and distribution control.

Input: one or many files (.csv/.txt/.md)
Output:
- CSV with columns: raw_text, ready_text, system, domain, mode, source_file
- JSON report with source/final distributions

Domains supported:
- telegram
- ads
- article
- dialog
- mixed (auto heuristic -> one of above)
"""

import argparse
import csv
import json
import os
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SYSTEM_BY_DOMAIN = {
    "telegram": (
        "Ты текстовый аугментатор для Telegram-постов. Расширяй текст без выдумки новых фактов, "
        "сохраняй стиль и читабельность."
    ),
    "ads": (
        "Ты текстовый аугментатор объявлений. Расширяй описание без выдумки новых фактов, "
        "сохраняй цену, контакты, условия и предмет продажи."
    ),
    "article": (
        "Ты текстовый аугментатор статей. Расширяй текст без выдумки новых фактов, "
        "сохраняй даты, цифры, имена и причинно-следственные связи."
    ),
    "dialog": (
        "Ты текстовый аугментатор диалогов. Расширяй реплики естественно, без смены ролей и без выдумки фактов."
    ),
}

MODE_CFG = {
    "light": (1.2, 1.7),
    "medium": (1.7, 2.6),
    "deep": (2.6, 3.8),
}

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|\b\d{4}\b")
NUM_RE = re.compile(r"\b\d+[\d.,]*\b")
URL_RE = re.compile(r"https?://\S+")
PHONE_RE = re.compile(r"(?:\+?7|8)\D*\d{3}\D*\d{3}\D*\d{2}\D*\d{2}")
ROLE_RE = re.compile(r"\b(Пользователь|Ассистент|user|assistant)\s*:", re.IGNORECASE)


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
        out.append(text[i:i + chunk_size])
        i += step
    return out


def split_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def smart_split_chunks(text: str, domain: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    cfg = {
        "dialog": {"target": 700, "max": 900, "overlap_sent": 1},
        "ads": {"target": 900, "max": 1200, "overlap_sent": 1},
        "telegram": {"target": 1200, "max": 1600, "overlap_sent": 1},
        "article": {"target": 1500, "max": 2200, "overlap_sent": 2},
    }.get(domain, {"target": 1200, "max": 1600, "overlap_sent": 1})

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current = ""
    prev_tail_sentences: List[str] = []

    def flush():
        nonlocal current, prev_tail_sentences
        c = clean_text(current)
        if c:
            chunks.append(c)
            sents = split_sentences(c)
            prev_tail_sentences = sents[-cfg["overlap_sent"] :] if sents else []
        current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= cfg["target"]:
            current = candidate
            continue

        if current:
            flush()
            if prev_tail_sentences:
                current = " ".join(prev_tail_sentences)

        # if paragraph itself too large, split by sentences
        if len(para) > cfg["max"]:
            sent_buf = current.strip()
            for s in split_sentences(para):
                cand2 = (sent_buf + " " + s).strip() if sent_buf else s
                if len(cand2) <= cfg["max"]:
                    sent_buf = cand2
                else:
                    if sent_buf:
                        chunks.append(clean_text(sent_buf))
                        tail = split_sentences(sent_buf)
                        sent_buf = " ".join(tail[-cfg["overlap_sent"] :]) if tail else ""
                    sent_buf = (sent_buf + " " + s).strip() if sent_buf else s
            current = sent_buf
        else:
            current = para

    if current:
        flush()

    return [c for c in chunks if c]


def infer_domain(text: str) -> str:
    low = text.lower()
    if ROLE_RE.search(text):
        return "dialog"
    if any(x in low for x in ["цена", "продам", "сдам", "торг", "руб", "₽", "контакт"]):
        return "ads"
    if any(x in low for x in ["подписывайтесь", "канал", "telegram", "t.me", "пост"]):
        return "telegram"
    return "article"


def extract_facts(text: str) -> Dict[str, set]:
    return {
        "dates": set(DATE_RE.findall(text)),
        "nums": set(NUM_RE.findall(text)),
        "urls": set(URL_RE.findall(text)),
        "phones": set(PHONE_RE.findall(text)),
    }


def missing_fact_ratio(src: Dict[str, set], out: Dict[str, set]) -> float:
    total = 0
    miss = 0
    for k in ["dates", "nums", "urls", "phones"]:
        for v in src[k]:
            total += 1
            if v not in out[k]:
                miss += 1
    return (miss / total) if total else 0.0


def too_many_new_numbers(src: Dict[str, set], out: Dict[str, set], domain: str) -> bool:
    new_nums = [x for x in out["nums"] if x not in src["nums"]]
    limit = 4 if domain == "article" else 2
    return len(new_nums) > limit


def domain_ranges(domain: str):
    if domain == "dialog":
        return {"src_min": 40, "src_max": 900, "tgt_min": 70, "tgt_max": 1800}
    if domain == "ads":
        return {"src_min": 60, "src_max": 1200, "tgt_min": 100, "tgt_max": 2200}
    if domain == "telegram":
        return {"src_min": 90, "src_max": 1800, "tgt_min": 140, "tgt_max": 2600}
    return {"src_min": 120, "src_max": 2200, "tgt_min": 180, "tgt_max": 3200}


def build_instruction(domain: str, mode: str, text: str) -> str:
    lo, hi = MODE_CFG[mode]
    style_hint = {
        "dialog": "Сохраняй роли и разговорный тон.",
        "ads": "Сохраняй факты объявления, цену и контакты.",
        "telegram": "Сохраняй постовый стиль и структуру абзацев.",
        "article": "Сохраняй нейтрально-информационный стиль.",
    }[domain]

    return (
        f"Расширь текст в режиме {mode} (примерно x{lo:.1f}-x{hi:.1f}). "
        "Не добавляй новых фактов. "
        f"{style_hint}\n\n"
        f"Исходный текст:\n{text}"
    )


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


def validate(src: str, out: str, mode: str, domain: str) -> bool:
    if not out:
        return False

    rng = domain_ranges(domain)
    if len(out) < rng["tgt_min"] or len(out) > rng["tgt_max"]:
        return False

    ratio = len(out) / max(1, len(src))
    lo, hi = MODE_CFG[mode]
    if ratio < lo * 0.85 or ratio > hi * 1.3:
        return False

    src_f = extract_facts(src)
    out_f = extract_facts(out)
    if missing_fact_ratio(src_f, out_f) > 0.55:
        return False
    if too_many_new_numbers(src_f, out_f, domain):
        return False

    if domain == "dialog" and ROLE_RE.search(src) and not ROLE_RE.search(out):
        # dialog should keep explicit roles if they were present
        return False

    return True


def parse_source_spec(spec: str) -> List[Dict[str, str]]:
    """
    spec format (repeatable via --source):
      path.csv:text_col:domain
      path.txt::domain
      path.csv:text_col:mixed
      sqlite:/abs/path.db:sql=SELECT ...:domain
    """
    out = []
    if spec.startswith("sqlite:"):
        parts = spec.split(":", 3)
        db_path = parts[1] if len(parts) > 1 else ""
        sql_part = parts[2] if len(parts) > 2 else ""
        domain = parts[3] if len(parts) > 3 else "mixed"
        if sql_part.startswith("sql="):
            sql_part = sql_part[4:]
        out.append({"path": db_path, "col": "", "domain": domain or "mixed", "kind": "sqlite", "sql": sql_part})
        return out

    parts = spec.split(":")
    if len(parts) == 1:
        out.append({"path": parts[0], "col": "raw_text", "domain": "mixed", "kind": "file", "sql": ""})
    elif len(parts) == 2:
        out.append({"path": parts[0], "col": parts[1] or "raw_text", "domain": "mixed", "kind": "file", "sql": ""})
    else:
        out.append({"path": parts[0], "col": parts[1] or "raw_text", "domain": parts[2] or "mixed", "kind": "file", "sql": ""})
    return out


def load_texts(path: Path, text_col: str) -> List[str]:
    if path.suffix.lower() in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                t = clean_text(str(row.get(text_col, "")))
                if t:
                    rows.append(t)
        return rows
    return []


def load_texts_sqlite(db_path: str, sql: str) -> List[str]:
    if not db_path or not sql:
        return []
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(sql).fetchall()
        out: List[str] = []
        for row in rows:
            if not row:
                continue
            # take first selected column as text payload
            t = clean_text(str(row[0] or ""))
            if t:
                out.append(t)
        return out
    finally:
        con.close()


def parse_target_distribution(s: str) -> Dict[str, float]:
    """format: telegram=0.3,ads=0.3,article=0.25,dialog=0.15"""
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in {"telegram", "ads", "article", "dialog"}:
            continue
        out[k] = max(0.0, float(v.strip()))
    total = sum(out.values())
    if total > 0:
        for k in list(out.keys()):
            out[k] = out[k] / total
    return out


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Universal domain-aware augmentation builder")
    ap.add_argument("--source", action="append", required=True, help="path[:text_col[:domain]]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--target-distribution", default="", help="telegram=0.3,ads=0.3,article=0.25,dialog=0.15")
    ap.add_argument("--modes", default="light,medium,deep")
    ap.add_argument("--chunking", choices=["fixed", "smart"], default="smart")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODE_CFG]
    if not modes:
        raise SystemExit("Invalid modes")

    src_specs = []
    for s in args.source:
        src_specs.extend(parse_source_spec(s))

    pool = []
    src_dist = Counter()

    for spec in src_specs:
        if spec.get("kind") == "sqlite":
            texts = load_texts_sqlite(spec["path"], spec.get("sql", ""))
            src_name = f"sqlite:{spec['path']}"
        else:
            p = Path(spec["path"])
            texts = load_texts(p, spec["col"])
            src_name = str(p)
        for t in texts:
            base_domain = spec["domain"] if spec["domain"] != "mixed" else infer_domain(t)
            if args.chunking == "smart":
                text_chunks = smart_split_chunks(t, base_domain)
            else:
                text_chunks = split_chunks(t, args.chunk_size, args.overlap)
            for ch in text_chunks:
                ch = clean_text(ch)
                dom = spec["domain"] if spec["domain"] != "mixed" else infer_domain(ch)
                rng = domain_ranges(dom)
                if rng["src_min"] <= len(ch) <= rng["src_max"]:
                    pool.append({"text": ch, "domain": dom, "source_file": src_name})
                    src_dist[dom] += 1

    if not pool:
        raise SystemExit("No eligible source chunks")

    random.shuffle(pool)

    target_dist = parse_target_distribution(args.target_distribution)
    final_quota = {}
    if args.max_samples > 0 and target_dist:
        for dom, w in target_dist.items():
            final_quota[dom] = int(args.max_samples * w)

    out_rows = []
    out_dist = Counter()
    skipped = 0

    # conservative upper bound for progress bar
    total_jobs = len(pool) * len(modes)
    pbar = tqdm(total=total_jobs, desc="universal-augment", unit="job")

    for item in pool:
        dom = item["domain"]
        if args.max_samples > 0 and len(out_rows) >= args.max_samples:
            break
        if final_quota and dom in final_quota and out_dist[dom] >= final_quota[dom]:
            pbar.update(len(modes))
            continue

        for mode in modes:
            if args.max_samples > 0 and len(out_rows) >= args.max_samples:
                break
            if final_quota and dom in final_quota and out_dist[dom] >= final_quota[dom]:
                pbar.update(1)
                continue

            src_text = item["text"]
            system = SYSTEM_BY_DOMAIN[dom]
            instruction = build_instruction(dom, mode, src_text)
            prompt = f"{system}\n\n{instruction}\n\nВерни только итоговый текст."

            got = None
            for _ in range(args.max_retries):
                try:
                    candidate = clean_text(call_ollama(ollama_url, ollama_model, prompt, args.timeout, args.temperature))
                    if validate(src_text, candidate, mode, dom):
                        got = candidate
                        break
                except Exception:
                    pass
                time.sleep(0.6)

            if got is None:
                skipped += 1
            else:
                out_rows.append(
                    {
                        "raw_text": instruction,
                        "ready_text": got,
                        "system": system,
                        "domain": dom,
                        "mode": mode,
                        "source_file": item["source_file"],
                    }
                )
                out_dist[dom] += 1

            pbar.update(1)
            pbar.set_postfix(ok=len(out_rows), skipped=skipped)
            time.sleep(args.sleep)

    # dedup
    uniq = {}
    for r in out_rows:
        uniq[(r["raw_text"], r["ready_text"], r["domain"], r["mode"])] = r
    out_rows = list(uniq.values())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system", "domain", "mode", "source_file"])
        w.writeheader()
        w.writerows(out_rows)

    rep = {
        "ollama_model": ollama_model,
        "source_distribution": dict(src_dist),
        "final_distribution": dict(Counter([r["domain"] for r in out_rows])),
        "mode_distribution": dict(Counter([r["mode"] for r in out_rows])),
        "rows": len(out_rows),
        "skipped": skipped,
        "target_distribution": target_dist,
    }

    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done")
    print(f"saved: {output}")
    print(f"report: {report_path}")
    print(f"rows: {len(out_rows)}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
