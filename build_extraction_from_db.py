#!/usr/bin/env python3
"""
Build extraction SFT dataset from sbaraholka.db.

Approach:
- Read core fields from DB (stable ground truth)
- Ask Ollama to extract only weak fields in SIMPLE key:value format
- Convert parsed result into strict JSON ourselves

Output CSV columns:
- raw_text
- ready_text  (JSON string)
- system
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

SYSTEM_PROMPT = (
    "Извлеки структурированные поля из текста объявления. "
    "Верни только факты, без выдумок."
)

SIMPLE_FORMAT_INSTRUCTION = """
Верни строго в формате строк key: value (без JSON):
location: ...
contacts: ...
condition: ...
category: ...
summary: ...

Правила:
- если поле не найдено: n/a
- contacts: телефоны/юзернеймы/ссылки через ;
- summary: 1 короткое предложение
""".strip()

PHONE_RE = re.compile(r"(?:\+?7|8)\D*\d{3}\D*\d{3}\D*\d{2}\D*\d{2}")
USERNAME_RE = re.compile(r"@[A-Za-z0-9_]{4,}")
URL_RE = re.compile(r"https?://\S+")


def normalize_phone(s: str) -> str:
    d = re.sub(r"\D", "", s)
    if len(d) == 11 and d.startswith("8"):
        d = "7" + d[1:]
    if len(d) == 11 and d.startswith("7"):
        return "+" + d
    return s.strip()


def extract_contacts_from_text(text: str) -> List[str]:
    hits = set()
    for m in PHONE_RE.findall(text):
        hits.add(normalize_phone(m))
    for m in USERNAME_RE.findall(text):
        hits.add(m)
    for m in URL_RE.findall(text):
        hits.add(m.rstrip('.,)'))
    return sorted(hits)


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


def parse_kv_output(text: str) -> Dict[str, str]:
    out = {"location": "", "contacts": "", "condition": "", "category": "", "summary": ""}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower()
        val = v.strip()
        if key in out:
            out[key] = val
    return out


def clean_na(v: str) -> str:
    if not v:
        return ""
    if v.strip().lower() in {"n/a", "na", "none", "null", "не найдено", "нет"}:
        return ""
    return v.strip()


def guess_category(tags: str, caption: str) -> str:
    t = (tags + " " + caption).lower()
    if any(x in t for x in ["квартира", "дом", "аренда", "недвиж"]):
        return "real_estate"
    if any(x in t for x in ["авто", "машин", "мото"]):
        return "auto"
    if any(x in t for x in ["работ", "ваканс", "резюме"]):
        return "jobs"
    if any(x in t for x in ["телефон", "ноут", "комп", "техник"]):
        return "electronics"
    return "other"


def build_raw_text(row: sqlite3.Row) -> str:
    parts = [
        f"Заголовок: {row['item_name']}",
        f"Цена: {row['item_price']}",
        f"Теги: {row['tags']}",
        f"Текст объявления: {row['item_caption']}",
    ]
    return "\n".join(parts)


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Build extraction dataset from SQLite + Ollama")
    ap.add_argument("--db-path", default="sbaraholka.db")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--min-caption-chars", type=int, default=80)
    args = ap.parse_args()

    ollama_url = os.getenv("OLLAMA_URL", "http://10.6.33.8:11434/api/generate")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")

    con = sqlite3.connect(args.db_path)
    con.row_factory = sqlite3.Row

    sql = (
        "SELECT id, source, source_date, source_link, published_date, item_name, item_caption, item_price, tags, status "
        "FROM items WHERE item_caption IS NOT NULL AND TRIM(item_caption) <> '' "
        "ORDER BY id"
    )
    if args.limit > 0:
        sql += f" LIMIT {int(args.limit)} OFFSET {int(args.offset)}"

    rows = con.execute(sql).fetchall()

    samples = []
    skipped = 0

    for row in tqdm(rows, desc="build-extraction", unit="item"):
        caption = (row["item_caption"] or "").strip()
        if len(caption) < args.min_caption_chars:
            skipped += 1
            continue

        raw_text = build_raw_text(row)

        prompt = (
            f"{SIMPLE_FORMAT_INSTRUCTION}\n\n"
            f"Текст объявления:\n{caption}\n"
        )

        parsed = None
        for _ in range(args.max_retries):
            try:
                resp = call_ollama(
                    url=ollama_url,
                    model=ollama_model,
                    prompt=prompt,
                    timeout=args.timeout,
                    temperature=args.temperature,
                )
                kv = parse_kv_output(resp)
                # accept if at least one auxiliary field extracted
                if any(clean_na(kv[k]) for k in ["location", "contacts", "condition", "summary"]):
                    parsed = kv
                    break
            except Exception:
                pass
            time.sleep(0.6)

        if parsed is None:
            parsed = {"location": "", "contacts": "", "condition": "", "category": "", "summary": ""}

        contacts = []
        llm_contacts = clean_na(parsed.get("contacts", ""))
        if llm_contacts:
            for c in [x.strip() for x in llm_contacts.split(";") if x.strip()]:
                contacts.append(c)
        contacts.extend(extract_contacts_from_text(caption))
        contacts = sorted(set(contacts))

        category = clean_na(parsed.get("category", "")) or guess_category(str(row["tags"] or ""), caption)

        obj = {
            "id": row["id"],
            "title": str(row["item_name"] or "").strip(),
            "price": row["item_price"],
            "currency": "RUB",
            "category": category,
            "location": clean_na(parsed.get("location", "")),
            "condition": clean_na(parsed.get("condition", "")),
            "contacts": contacts,
            "source": str(row["source"] or "").strip(),
            "source_link": str(row["source_link"] or "").strip(),
            "published_date": str(row["published_date"] or "").strip(),
            "tags": str(row["tags"] or "").strip(),
            "status": str(row["status"] or "").strip(),
            "summary": clean_na(parsed.get("summary", "")),
        }

        samples.append({
            "raw_text": raw_text,
            "ready_text": json.dumps(obj, ensure_ascii=False),
            "system": SYSTEM_PROMPT,
        })

        time.sleep(args.sleep)

    con.close()

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
