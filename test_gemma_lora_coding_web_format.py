#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Ты senior web engineer. Пиши только в заданном формате без лишнего текста. "
    "Код должен быть рабочим, современным, адаптивным и безопасным."
)

FORMAT_INSTRUCTION = """
Верни ответ СТРОГО в формате:
[PLAN]
краткий план (3-7 пунктов)
[/PLAN]
[FILES]
file1
file2
...
[/FILES]
[CODE:relative/path/to/file]
<полный код файла>
[/CODE]
(повтори блок [CODE:...] для каждого файла)

Правила:
- Никакого текста вне блоков.
- Все пути в [FILES] должны иметь соответствующий [CODE:path].
- Для веб-задач обязателен index.html.
- Если есть JS, добавь отдельный .js файл.
- Для стилей используй styles.css.
""".strip()

TEST_CASES = [
    {"id": "landing_01", "task_type": "landing", "prompt": "Сделай лендинг для курса по Python: hero, программа курса, отзывы, FAQ, CTA. Нужна адаптивность mobile/desktop и аккуратная типографика.", "must_files": ["index.html", "styles.css"], "must_contain": ["hero", "faq", "@media"]},
    {"id": "site_01", "task_type": "site", "prompt": "Сделай многостраничный сайт для кофейни: главная, меню, контакты. Добавь навигацию между страницами и единый стиль.", "must_files": ["index.html", "menu.html", "contacts.html", "styles.css"], "must_contain": ["<nav", "menu.html", "contacts.html"]},
    {"id": "webapp_01", "task_type": "webapp", "prompt": "Сделай веб-приложение ToDo: добавление, удаление, отметка выполнения, фильтр все/активные/выполненные, сохранение в localStorage.", "must_files": ["index.html", "styles.css", "app.js"], "must_contain": ["localStorage", "addEventListener", "filter"]},
    {"id": "webapp_02", "task_type": "webapp", "prompt": "Сделай мини CRM страницу: таблица клиентов, поиск по имени, сортировка по дате, форма добавления клиента с валидацией email.", "must_files": ["index.html", "styles.css", "app.js"], "must_contain": ["email", "sort", "search"]},
    {"id": "landing_02", "task_type": "landing", "prompt": "Сделай промо-страницу мобильного приложения с секциями: преимущества, скриншоты, тарифы, блок скачивания. Добавь плавные анимации появления секций.", "must_files": ["index.html", "styles.css"], "must_contain": ["animation", "pricing", "download"]},
]


@dataclass
class ParseResult:
    ok: bool
    errors: List[str]
    files: List[str]
    code_map: Dict[str, str]


def extract_block(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\s*(.*?)\s*\[/{name}\]", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_response(text: str) -> ParseResult:
    errors = []
    plan = extract_block(text, "PLAN")
    files_raw = extract_block(text, "FILES")
    if not plan:
        errors.append("missing PLAN block")
    if not files_raw:
        errors.append("missing FILES block")

    files = [x.strip() for x in files_raw.splitlines() if x.strip()]
    code_blocks = re.findall(r"\[CODE:([^\]]+)\]\s*(.*?)\s*\[/CODE\]", text, flags=re.DOTALL)
    code_map = {path.strip(): code for path, code in code_blocks}

    stripped = text
    stripped = re.sub(r"\[PLAN\].*?\[/PLAN\]", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"\[FILES\].*?\[/FILES\]", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r"\[CODE:[^\]]+\].*?\[/CODE\]", "", stripped, flags=re.DOTALL)
    if stripped.strip():
        errors.append("text outside allowed blocks")

    for f in files:
        if f not in code_map:
            errors.append(f"missing CODE block for file: {f}")
    for f in code_map:
        if f not in files:
            errors.append(f"CODE block file not listed in FILES: {f}")

    if "index.html" not in files:
        errors.append("index.html is required")

    for f, c in code_map.items():
        if not c.strip():
            errors.append(f"empty code in file: {f}")

    return ParseResult(ok=(len(errors) == 0), errors=errors, files=files, code_map=code_map)


def score_case(case: dict, parsed: ParseResult) -> Tuple[int, List[str]]:
    errors = list(parsed.errors)
    for mf in case.get("must_files", []):
        if mf not in parsed.files:
            errors.append(f"missing required file: {mf}")

    combined = "\n".join(parsed.code_map.values()).lower()
    for needle in case.get("must_contain", []):
        if needle.lower() not in combined:
            errors.append(f"missing semantic token: {needle}")

    score = 100
    score -= min(60, 8 * len(parsed.errors))
    score -= min(40, 5 * (len(errors) - len(parsed.errors)))
    return max(0, score), errors


def make_prompt(task: str) -> str:
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n\n"
        f"[FORMAT]\n{FORMAT_INSTRUCTION}\n[/FORMAT]\n\n"
        f"[TASK]\n{task}\n[/TASK]\n"
    )


def main():
    ap = argparse.ArgumentParser(description="Test Gemma LoRA coding agent on web format tasks")
    ap.add_argument("--base-model", default="google/gemma-4-E2B-it")
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--do-sample", action="store_true", default=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    rows = []
    for case in tqdm(TEST_CASES, desc="gemma-coding-web", unit="case"):
        item = {"case_id": case["id"], "task_type": case["task_type"], "prompt": case["prompt"]}
        prompt = make_prompt(case["prompt"])
        ids = tok(prompt, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                repetition_penalty=1.05,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t0

        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        parsed = parse_response(text)
        score, errors = score_case(case, parsed)

        item.update(
            {
                "ok": 1 if score >= 70 and not parsed.errors else 0,
                "format_ok": 1 if parsed.ok else 0,
                "score": score,
                "latency_s": round(dt, 3),
                "files": parsed.files,
                "errors": errors,
                "response": text,
            }
        )
        rows.append(item)

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["case_id", "task_type", "ok", "format_ok", "score", "latency_s", "files", "errors"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in fields}
            rr["files"] = json.dumps(rr["files"], ensure_ascii=False)
            rr["errors"] = json.dumps(rr["errors"], ensure_ascii=False)
            w.writerow(rr)

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"] == 1)
    fmt = sum(1 for r in rows if r["format_ok"] == 1)
    avg = round(sum(r["score"] for r in rows) / total, 2) if total else 0

    print(f"saved json: {out_json}")
    print(f"saved csv:  {out_csv}")
    print(f"cases: {total}, passed: {passed}, format_ok: {fmt}, avg_score: {avg}")


if __name__ == "__main__":
    main()
