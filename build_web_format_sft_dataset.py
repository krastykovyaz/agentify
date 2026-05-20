#!/usr/bin/env python3
"""
Build web coding SFT dataset in strict block format:
[PLAN] ... [/PLAN]
[FILES] ... [/FILES]
[CODE:path] ... [/CODE]

Output CSV columns:
- task
- instruction
- input
- output
- source
- language

Can mix:
1) Local seed CSV (existing coding/web rows)
2) Synthetic templated examples (algorithmic, no LLM)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


LANDING_TOPICS = [
    "курс по Python", "мобильное приложение фитнеса", "онлайн-школа английского", "сервис доставки еды",
    "студия веб-дизайна", "курс по data science", "финтех-приложение", "агентство недвижимости",
]

WEBAPPS = [
    "ToDo менеджер", "трекер привычек", "мини CRM", "планировщик задач", "финансовый трекер",
    "каталог заметок", "трекер чтения", "система бронирования встреч",
]

SITES = [
    "кофейня", "ветклиника", "образовательный центр", "автосервис", "фотостудия", "туристическое агентство",
]


@dataclass
class Example:
    task: str
    instruction: str
    input: str
    output: str
    source: str
    language: str


BASE_CSS = """* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, system-ui, sans-serif; color: #111; background: #f7f7f8; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.card { background: white; border-radius: 14px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,.06); }
.btn { display: inline-block; padding: 10px 16px; border-radius: 10px; background: #0f62fe; color: #fff; text-decoration: none; }
@media (max-width: 768px) { .container { padding: 16px; } }
"""


def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def render_output(files: Dict[str, str], plan: List[str]) -> str:
    plan_txt = "\n".join(f"- {x}" for x in plan)
    file_list = "\n".join(files.keys())
    chunks = [
        "[PLAN]",
        plan_txt,
        "[/PLAN]",
        "[FILES]",
        file_list,
        "[/FILES]",
    ]
    for path, code in files.items():
        chunks += [f"[CODE:{path}]", code.rstrip(), "[/CODE]"]
    return "\n".join(chunks).strip()


def make_landing(topic: str) -> Example:
    title = topic.title()
    instruction = (
        f"Сделай адаптивный landing page для проекта '{topic}'. "
        "Нужны секции hero, преимущества, отзывы, FAQ и CTA. "
        "Используй семантический HTML и отдельный CSS файл."
    )
    index = f"""<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{title}</title>
  <link rel=\"stylesheet\" href=\"styles.css\" />
</head>
<body>
  <main class=\"container\">
    <section class=\"card hero\"><h1>{title}</h1><p>Современное решение для клиентов.</p><a class=\"btn\" href=\"#cta\">Начать</a></section>
    <section class=\"card\" id=\"features\"><h2>Преимущества</h2><ul><li>Быстро</li><li>Надежно</li><li>Прозрачно</li></ul></section>
    <section class=\"card\" id=\"reviews\"><h2>Отзывы</h2><p>Клиенты довольны качеством сервиса.</p></section>
    <section class=\"card\" id=\"faq\"><h2>FAQ</h2><details><summary>Сколько стоит?</summary><p>Есть бесплатный тариф.</p></details></section>
    <section class=\"card\" id=\"cta\"><h2>Готовы начать?</h2><a class=\"btn\" href=\"#\">Оставить заявку</a></section>
  </main>
</body>
</html>
"""
    files = {"index.html": index, "styles.css": BASE_CSS}
    out = render_output(files, [
        "Собрать структуру лендинга с ключевыми секциями.",
        "Добавить единый визуальный стиль и адаптивность.",
        "Подготовить CTA для конверсии.",
    ])
    return Example("coding_web", instruction, "", out, "templated_web", "html_css")


def make_site(topic: str) -> Example:
    instruction = (
        f"Сделай многостраничный сайт для '{topic}': главная, услуги/меню и контакты. "
        "Нужна общая навигация на всех страницах и единый CSS."
    )
    nav = '<nav><a href="index.html">Главная</a> | <a href="menu.html">Меню</a> | <a href="contacts.html">Контакты</a></nav>'
    files = {
        "index.html": f"<!doctype html><html lang=\"ru\"><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>{topic}</title><link rel=\"stylesheet\" href=\"styles.css\"></head><body><main class=\"container card\">{nav}<h1>{topic.title()}</h1><p>Добро пожаловать.</p></main></body></html>",
        "menu.html": f"<!doctype html><html lang=\"ru\"><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>Меню</title><link rel=\"stylesheet\" href=\"styles.css\"></head><body><main class=\"container card\">{nav}<h1>Меню</h1><ul><li>Позиция 1</li><li>Позиция 2</li></ul></main></body></html>",
        "contacts.html": f"<!doctype html><html lang=\"ru\"><head><meta charset=\"UTF-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"><title>Контакты</title><link rel=\"stylesheet\" href=\"styles.css\"></head><body><main class=\"container card\">{nav}<h1>Контакты</h1><p>Телефон: +7 (900) 000-00-00</p></main></body></html>",
        "styles.css": BASE_CSS,
    }
    out = render_output(files, [
        "Создать три страницы с общей навигацией.",
        "Вынести стили в единый файл.",
        "Сохранить консистентный UI на всех страницах.",
    ])
    return Example("coding_web", instruction, "", out, "templated_web", "html_css")


def make_webapp(topic: str) -> Example:
    instruction = (
        f"Сделай веб-приложение '{topic}' на HTML/CSS/JS. "
        "Добавь интерактивность через JavaScript и хранение состояния в localStorage."
    )
    index = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Web App</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main class="container card">
    <h1>Web App</h1>
    <form id="add-form"><input id="item-input" placeholder="Новая запись" /><button class="btn" type="submit">Добавить</button></form>
    <ul id="items"></ul>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""
    js = """const KEY = 'webapp_items_v1';
const form = document.getElementById('add-form');
const input = document.getElementById('item-input');
const itemsEl = document.getElementById('items');

let state = JSON.parse(localStorage.getItem(KEY) || '[]');

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function render(){
  itemsEl.innerHTML = '';
  state.forEach((x, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${x.text}</span> <button data-i="${i}">Удалить</button>`;
    itemsEl.appendChild(li);
  });
}

itemsEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-i]');
  if (!btn) return;
  state.splice(Number(btn.dataset.i), 1);
  save(); render();
});

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  state.push({ text });
  input.value = '';
  save(); render();
});

render();
"""
    files = {"index.html": index, "styles.css": BASE_CSS, "app.js": js}
    out = render_output(files, [
        "Собрать минимальный UI приложения.",
        "Реализовать интерактивность через JS обработчики.",
        "Добавить localStorage для сохранения данных.",
    ])
    return Example("coding_web", instruction, "", out, "templated_web", "html_css_js")


def load_seed_csv(path: Path) -> List[Example]:
    rows: List[Example] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ins = clean_text(str(r.get("instruction", "")))
            out = clean_text(str(r.get("output", "")))
            if not ins or not out:
                continue
            low = (ins + "\n" + out).lower()
            # only web-like rows
            if not any(k in low for k in ["html", "css", "javascript", "web", "landing", "site", "react", "frontend"]):
                continue
            rows.append(Example(
                task=str(r.get("task", "coding_web") or "coding_web"),
                instruction=ins,
                input=clean_text(str(r.get("input", ""))),
                output=out,
                source=str(r.get("source", "seed_csv") or "seed_csv"),
                language=str(r.get("language", "web") or "web"),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build web SFT dataset in strict output format")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed-csv", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--n-landing", type=int, default=1200)
    ap.add_argument("--n-site", type=int, default=1200)
    ap.add_argument("--n-webapp", type=int, default=1200)
    ap.add_argument("--max-seed-rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_rows: List[Example] = []

    if args.seed_csv:
        sp = Path(args.seed_csv)
        if sp.exists():
            seed_rows = load_seed_csv(sp)
            rng.shuffle(seed_rows)
            out_rows.extend(seed_rows[: args.max_seed_rows])

    for _ in tqdm(range(args.n_landing), desc="build-landing", unit="row"):
        out_rows.append(make_landing(rng.choice(LANDING_TOPICS)))
    for _ in tqdm(range(args.n_site), desc="build-site", unit="row"):
        out_rows.append(make_site(rng.choice(SITES)))
    for _ in tqdm(range(args.n_webapp), desc="build-webapp", unit="row"):
        out_rows.append(make_webapp(rng.choice(WEBAPPS)))

    rng.shuffle(out_rows)

    # dedupe by instruction+output
    seen = set()
    final: List[Example] = []
    for r in out_rows:
        k = (r.instruction, r.output)
        if k in seen:
            continue
        seen.add(k)
        final.append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "instruction", "input", "output", "source", "language"])
        w.writeheader()
        for r in final:
            w.writerow({
                "task": r.task,
                "instruction": r.instruction,
                "input": r.input,
                "output": r.output,
                "source": r.source,
                "language": r.language,
            })

    report = {
        "output": str(out),
        "rows": len(final),
        "seed_csv": args.seed_csv,
        "n_landing": args.n_landing,
        "n_site": args.n_site,
        "n_webapp": args.n_webapp,
        "max_seed_rows": args.max_seed_rows,
    }

    rp = Path(args.report) if args.report else out.with_suffix(".report.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {out}")
    print(f"report: {rp}")
    print(f"rows: {len(final)}")


if __name__ == "__main__":
    main()
