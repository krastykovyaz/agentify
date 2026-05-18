#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm

TASK_PROMPTS = {
    "summarization": "Задача: summarization\nИнструкция: Сделай краткую фактическую выжимку текста.\n\nВход:\n{input}\n\nОтвет:\n",
    "telegram_post": "Задача: telegram_post\nИнструкция: Сделай пост для Telegram: заголовок и 1-3 абзаца, без выдумки фактов.\n\nВход:\n{input}\n\nОтвет:\n",
    "qa": "Задача: qa\nИнструкция: Ответь кратко и по фактам на вопрос по контексту.\n\nВход:\n{input}\n\nОтвет:\n",
    "extraction": "Задача: extraction\nИнструкция: Извлеки структуру в JSON с ключами title, category, location, date, price, contacts, summary.\n\nВход:\n{input}\n\nОтвет:\n",
    "dialog": "Задача: dialog\nИнструкция: Ответь как эмпатичный собеседник, кратко и уместно.\n\nВход:\n{input}\n\nОтвет:\n",
}

DEFAULT_SAMPLES = {
    "summarization": [
        "Сегодня в Ижевске ожидается сильный ветер, местами дождь и понижение температуры до +8 ночью. МЧС рекомендует ограничить поездки за город.",
        "На участке трассы М-7 временно ограничено движение из-за ремонта. Работы продлятся до конца недели, организован объезд.",
    ],
    "telegram_post": [
        "В субботу в центре города пройдет ярмарка вакансий. Участвуют более 40 компаний, вход свободный, начало в 12:00.",
        "В воскресенье в парке состоится семейный фестиваль, начало в 11:00, вход свободный.",
    ],
    "qa": [
        "Вопрос: Что пройдет в субботу? Контекст: В субботу в центре города пройдет ярмарка вакансий, начало в 12:00.",
        "Вопрос: До какого дня ремонт? Контекст: Ограничение движения на М-7 продлится до конца недели.",
    ],
    "extraction": [
        "Продам велосипед Trek, 25000 руб, Ижевск, звонить +79001234567.",
        "Сдам 1-комнатную квартиру, 18000 руб/мес, Мытищи, писать @owner_home.",
    ],
    "dialog": [
        "Пользователь: У меня тяжелый день, все валится из рук.\nАссистент:",
        "Пользователь: Я переживаю перед важной встречей.\nАссистент:",
    ],
}


def read_instances(csv_path: Path) -> List[Dict[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("name"):
                continue
            rows.append(r)
    return rows


def load_samples(samples_json: Path | None) -> Dict[str, List[str]]:
    if not samples_json:
        return DEFAULT_SAMPLES
    data = json.loads(samples_json.read_text(encoding="utf-8"))
    merged = {**DEFAULT_SAMPLES}
    for k, v in data.items():
        if isinstance(v, list) and v:
            merged[k] = [str(x) for x in v]
    return merged


def call_completion(base_url: str, prompt: str, max_tokens: int, temperature: float, top_p: float, timeout_s: int):
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": "local",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    started = time.time()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    elapsed = time.time() - started
    resp.raise_for_status()
    js = resp.json()
    text = js.get("choices", [{}])[0].get("text", "").strip()
    usage = js.get("usage", {})
    return text, elapsed, usage, js


def main():
    ap = argparse.ArgumentParser(description="Query multiple llama-server instances and save outputs")
    ap.add_argument("--instances", required=True, help="CSV: name,port,task (and optionally base_url)")
    ap.add_argument("--samples-json", default=None, help="Optional JSON overrides for samples per task")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    inst_rows = read_instances(Path(args.instances))
    samples = load_samples(Path(args.samples_json) if args.samples_json else None)

    jobs = []
    for inst in inst_rows:
        task = (inst.get("task") or "").strip()
        if task not in TASK_PROMPTS:
            continue
        for sample in samples.get(task, []):
            jobs.append((inst, task, sample))

    results = []
    for inst, task, sample in tqdm(jobs, desc="query-agents", unit="req"):
        base_url = (inst.get("base_url") or f"http://{args.host}:{inst['port']}").strip()
        prompt = TASK_PROMPTS[task].format(input=sample)

        item = {
            "name": inst.get("name"),
            "task": task,
            "port": inst.get("port"),
            "base_url": base_url,
            "input": sample,
        }

        try:
            text, elapsed, usage, raw = call_completion(
                base_url,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout_s=args.timeout,
            )
            item.update({
                "ok": 1,
                "output": text,
                "latency_s": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "raw": raw,
            })
        except Exception as e:
            item.update({"ok": 0, "output": "", "latency_s": None, "error": str(e), "raw": None})

        results.append(item)

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "name", "task", "port", "base_url", "ok", "latency_s",
            "prompt_tokens", "completion_tokens", "total_tokens", "input", "output", "error"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})

    ok_n = sum(1 for r in results if r.get("ok") == 1)
    print(f"saved json: {out_json}")
    print(f"saved csv:  {out_csv}")
    print(f"requests: {len(results)}, ok: {ok_n}, failed: {len(results) - ok_n}")


if __name__ == "__main__":
    main()
