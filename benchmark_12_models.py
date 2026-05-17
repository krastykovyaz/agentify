#!/usr/bin/env python3
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_SAMPLES = {
    "summarization": [
        "Сегодня в Ижевске ожидается сильный ветер, местами дождь и понижение температуры до +8 ночью. МЧС рекомендует ограничить поездки за город.",
    ],
    "telegram_post": [
        "В субботу в центре города пройдет ярмарка вакансий. Участвуют более 40 компаний, вход свободный, начало в 12:00.",
    ],
    "qa": [
        "Вопрос: Какая погода ожидается? Контекст: Сегодня в городе сильный ветер и дождь.",
    ],
    "extraction": [
        "Продам велосипед Trek, 25000 руб, Ижевск, звонить +79001234567.",
    ],
}

TASK_INSTRUCTIONS = {
    "summarization": "Сделай краткую фактическую выжимку текста.",
    "telegram_post": "Сделай пост для Telegram: заголовок и 1-3 абзаца, без выдумки фактов.",
    "qa": "Ответь кратко и по фактам.",
    "extraction": "Извлеки JSON с полями: title, category, location, date, price, contacts, summary.",
}


@dataclass
class ModelCfg:
    name: str
    family: str
    base_model: str
    lora_path: str | None
    tasks: List[str]
    system: str


def clean_output(text: str) -> str:
    t = text.strip()
    for m in ["\nПользователь:", "\nАссистент:", "<|im_start|>user", "<|im_start|>assistant"]:
        if m in t:
            t = t.split(m, 1)[0]
    return t.strip()


def load_cfg(path: Path) -> List[ModelCfg]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data["models"]:
        out.append(
            ModelCfg(
                name=x["name"],
                family=x["family"],
                base_model=x["base_model"],
                lora_path=x.get("lora_path"),
                tasks=x["tasks"],
                system=x.get("system", "Ты полезный ассистент."),
            )
        )
    return out


def run_one_model(cfg: ModelCfg, max_new_tokens: int, temperature: float, top_p: float):
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if cfg.lora_path:
        model = PeftModel.from_pretrained(model, cfg.lora_path)
    model.eval()

    rows = []
    for task in cfg.tasks:
        samples = TASK_SAMPLES.get(task, [])
        for s in samples:
            user_input = f"[{task}] {TASK_INSTRUCTIONS[task]}\n\n{s}"
            messages = [
                {"role": "system", "content": cfg.system},
                {"role": "user", "content": user_input},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )
            dt = time.time() - t0
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            gen = clean_output(gen)

            rows.append(
                {
                    "model": cfg.name,
                    "family": cfg.family,
                    "task": task,
                    "input": s,
                    "output": gen,
                    "latency_sec": f"{dt:.2f}",
                    "has_lora": bool(cfg.lora_path),
                }
            )
    return rows


def write_markdown(rows: List[Dict], md_path: Path):
    lines = ["# Benchmark Results", "", "| model | family | task | latency_sec | output |", "|---|---|---:|---:|---|"]
    for r in rows:
        out = r["output"].replace("\n", " ").replace("|", "\\|")
        if len(out) > 220:
            out = out[:220] + "..."
        lines.append(f"| {r['model']} | {r['family']} | {r['task']} | {r['latency_sec']} | {out} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config with 12 models")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.85)
    args = ap.parse_args()

    models = load_cfg(Path(args.config))
    all_rows = []
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m.name}")
        all_rows.extend(run_one_model(m, args.max_new_tokens, args.temperature, args.top_p))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "family", "task", "input", "output", "latency_sec", "has_lora"])
        w.writeheader()
        w.writerows(all_rows)

    write_markdown(all_rows, Path(args.out_md))
    print(f"saved csv: {out_csv}")
    print(f"saved md: {args.out_md}")


if __name__ == "__main__":
    main()
