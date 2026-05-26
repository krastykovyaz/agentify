#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv


def read_yaml_like(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # tiny fallback: config as json-compatible yaml subset
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except Exception as e:
            raise SystemExit(f"Cannot parse config {path}. Install pyyaml or provide JSON config. {e}")


def load_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def infer_task(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["продам", "сдам", "цена", "руб", "тел", "whatsapp", "@"]):
        return "extraction"
    if any(k in t for k in ["вопрос:", "ответь", "что", "когда", "где", "почему"]):
        return "qa"
    if any(k in t for k in ["пост", "канал", "подпис", "новость", "анонс"]):
        return "telegram"
    if any(k in t for k in ["пережива", "устал", "тяжелый день", "поддерж", "поговорить"]):
        return "dialogue"
    if len(t) > 500:
        return "summary"
    return "universal"


def read_input_rows(path: Path) -> List[str]:
    ext = path.suffix.lower()
    if ext in {".txt"}:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        chunks = [normalize_text(x) for x in re.split(r"\n\s*\n", txt) if normalize_text(x)]
        return chunks
    if ext in {".csv"}:
        out = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames or []
            col = "text" if "text" in cols else (cols[0] if cols else None)
            if not col:
                return []
            for row in r:
                t = normalize_text(str(row.get(col, "")))
                if t:
                    out.append(t)
        return out
    if ext in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        if isinstance(data, list):
            for x in data:
                if isinstance(x, str):
                    t = normalize_text(x)
                elif isinstance(x, dict):
                    t = normalize_text(str(x.get("text", "")))
                else:
                    t = ""
                if t:
                    out.append(t)
        return out
    raise SystemExit(f"Unsupported input extension: {ext}")


def call_ollama(base_url: str, model: str, system: str, user_text: str, timeout: int = 180) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "options": {"temperature": 0.35, "top_p": 0.9, "num_ctx": 8192},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return normalize_text((r.json().get("message") or {}).get("content", ""))


def make_train_row(text: str, task: str, wrapper_prompt: str) -> dict:
    instruction = {
        "summary": "Сделай краткое резюме исходного текста.",
        "qa": "Ответь на вопрос по исходному тексту.",
        "extraction": "Извлеки структурированные поля из текста.",
        "dialogue": "Ответь эмпатично и уместно.",
        "telegram": "Сделай один готовый пост для Telegram.",
        "universal": "Реши задачу по тексту и выдай результат в правильном формате.",
    }.get(task, "Реши задачу по тексту.")
    return {
        "task": task,
        "instruction": instruction,
        "input": text,
        "output": "",
        "source": "real",
        "system": wrapper_prompt,
    }


def stratified_trim(rows: List[dict], target: int, seed: int) -> List[dict]:
    if len(rows) <= target:
        return rows
    rnd = random.Random(seed)
    buckets: Dict[str, List[dict]] = {}
    for r in rows:
        buckets.setdefault(r["task"], []).append(r)
    for b in buckets.values():
        rnd.shuffle(b)

    trimmed = []
    keys = sorted(buckets.keys())
    while len(trimmed) < target and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(trimmed) < target:
                trimmed.append(buckets[k].pop())
    rnd.shuffle(trimmed)
    return trimmed


def augment_to_target(rows: List[dict], cfg: dict, target: int, seed: int) -> List[dict]:
    if len(rows) >= target:
        return rows
    rnd = random.Random(seed)

    agents = cfg["agents"]
    base_url = cfg["ollama_base_url"]

    counts = Counter(r["task"] for r in rows)
    tasks = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
    min_count = min((counts.get(t, 0) for t in tasks), default=0)

    synth_limit = int(target * float(cfg.get("synthetic_max_ratio", 0.35)))
    synth_added = 0

    seeds = [r for r in rows if r.get("input")]
    if not seeds:
        return rows

    while len(rows) < target and synth_added < synth_limit:
        # pick underrepresented task first
        t = min(tasks, key=lambda x: counts.get(x, 0))
        a = agents[t]
        system = load_prompt(a["prompt_file"])
        model = a["model"]

        base = rnd.choice(seeds)
        src_text = base["input"]

        try:
            out = call_ollama(base_url, model, system, src_text)
        except Exception:
            # fallback through universal
            try:
                ua = agents["universal"]
                out = call_ollama(base_url, ua["model"], load_prompt(ua["prompt_file"]), src_text)
            except Exception:
                break

        if not out:
            continue

        row = {
            "task": t,
            "instruction": base["instruction"],
            "input": src_text,
            "output": out,
            "source": f"synthetic:{model}",
            "system": base.get("system", ""),
        }
        rows.append(row)
        counts[t] += 1
        synth_added += 1

    return rows


def materialize_outputs(rows: List[dict], cfg: dict) -> List[dict]:
    base_url = cfg["ollama_base_url"]
    agents = cfg["agents"]
    out_rows = []
    for r in rows:
        if r.get("output"):
            out_rows.append(r)
            continue
        task = r["task"]
        a = agents.get(task, agents["universal"])
        system = load_prompt(a["prompt_file"])
        try:
            out = call_ollama(base_url, a["model"], system, r["input"])
        except Exception:
            out = ""
        if not out:
            continue
        rr = dict(r)
        rr["output"] = out
        out_rows.append(rr)
    return out_rows


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Dataset pipeline: normalize + trim/augment to 1000")
    ap.add_argument("--input", required=True)
    ap.add_argument("--config", default="/home/aleksandr.koviazin/kovyaz/agentify/pipeline/pipeline_config.yaml")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()

    cfg = read_yaml_like(Path(args.config))
    target = int(cfg.get("target_samples", 1000))
    seed = int(cfg.get("random_seed", 42))
    wrapper_prompt = load_prompt(cfg.get("final_wrapper_prompt_file", ""))

    texts = read_input_rows(Path(args.input))
    rows = [make_train_row(t, infer_task(t), wrapper_prompt) for t in texts]

    original_n = len(rows)
    if original_n > target:
        rows = stratified_trim(rows, target, seed)

    rows = augment_to_target(rows, cfg, target, seed)
    rows = materialize_outputs(rows, cfg)

    # final trim if over target
    if len(rows) > target:
        rows = stratified_trim(rows, target, seed)

    # write csv
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "instruction", "input", "output", "source", "system"]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    report = {
        "input": args.input,
        "target": target,
        "original_rows": original_n,
        "final_rows": len(rows),
        "task_distribution": dict(Counter(r["task"] for r in rows)),
        "source_distribution": dict(Counter(r["source"] for r in rows)),
        "config": args.config,
    }
    out_report = Path(args.report_json)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved dataset: {out_csv}")
    print(f"saved report:  {out_report}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
