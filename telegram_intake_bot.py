#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_intake_bot")

TARGET = 1000
AGENT_STYLES = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]


def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def infer_style(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["продам", "сдам", "цена", "руб", "тел", "whatsapp", "@"]):
        return "extraction"
    if any(k in t for k in ["вопрос:", "ответь", "что", "когда", "где", "почему", "как"]):
        return "qa"
    if any(k in t for k in ["канал", "подпис", "анонс", "пост", "новость"]):
        return "telegram"
    if any(k in t for k in ["тяжелый день", "пережива", "поддерж", "поговорить", "устал"]):
        return "dialogue"
    if len(t) > 600:
        return "summary"
    return "universal"


def read_texts(path: Path) -> List[str]:
    ext = path.suffix.lower()
    if ext == ".txt":
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return [norm(x) for x in re.split(r"\n\s*\n", raw) if norm(x)]
    if ext == ".csv":
        out = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames or []
            col = "text" if "text" in cols else (cols[0] if cols else None)
            if not col:
                return []
            for row in r:
                t = norm(str(row.get(col, "")))
                if t:
                    out.append(t)
        return out
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        if isinstance(data, list):
            for x in data:
                if isinstance(x, str):
                    t = norm(x)
                elif isinstance(x, dict):
                    t = norm(str(x.get("text", "")))
                else:
                    t = ""
                if t:
                    out.append(t)
        return out
    return []


def build_report(texts: List[str]) -> str:
    n = len(texts)
    if n == 0:
        return "Не нашел валидных текстов в файле. Поддерживаются csv/json/txt."

    lens = [len(t) for t in texts]
    styles = [infer_style(t) for t in texts]
    dist = Counter(styles)

    dominant = dist.most_common(1)[0][0]
    dominant_share = dist[dominant] / n

    lines = []
    lines.append(f"Принято примеров: {n}")
    lines.append(f"Средняя длина: {int(sum(lens)/len(lens))} символов")
    lines.append(f"P95 длины: {sorted(lens)[int(0.95*(len(lens)-1))]}")
    lines.append(f"Доминирующий стиль: {dominant} ({dominant_share:.0%})")
    lines.append("Распределение по стилям:")
    for k in AGENT_STYLES:
        lines.append(f"- {k}: {dist.get(k,0)}")

    if n >= TARGET:
        lines.append("")
        lines.append(f"Данных достаточно (>= {TARGET}). Готов приступать к обучению.")
    else:
        need = TARGET - n
        lines.append("")
        lines.append(f"Данных недостаточно: нужно добрать {need} примеров до {TARGET}.")
        # proportional recommendation: fill weaker styles first
        weights = {k: 1.0/(dist.get(k,0)+1) for k in AGENT_STYLES}
        s = sum(weights.values())
        lines.append("Рекомендованная пропорция добора по стилям:")
        for k in AGENT_STYLES:
            p = weights[k]/s
            lines.append(f"- {k}: {p:.0%} (~{round(need*p)})")

    return "\n".join(lines)


def allowed_file(name: str) -> bool:
    return Path(name or "").suffix.lower() in {".csv", ".json", ".txt"}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Отправь файл с текстами (csv/json/txt). Я проанализирую объем, стиль и скажу, достаточно ли данных для обучения на 1000 примеров."
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пришли файл (csv/json/txt). Текстовые сообщения не обрабатываю в этом режиме.")


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return
    doc = update.message.document
    if not allowed_file(doc.file_name):
        await update.message.reply_text("Поддерживаются только файлы: .csv .json .txt")
        return

    root = Path(os.getenv("AGENTIFY_ROOT", "/home/aleksandr.koviazin/kovyaz/agentify")).resolve()
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    local_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{doc.file_name}"
    local_path = inbox / local_name

    tg_file = await context.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(str(local_path))

    await update.message.reply_text(f"Файл принят: {local_name}. Анализирую...")

    try:
        texts = read_texts(local_path)
        report = build_report(texts)
    except Exception as e:
        report = f"Ошибка обработки файла: {e}"

    await update.message.reply_text(report)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)


def main():
    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_BOT_TOKEN in .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    logger.info("Intake bot started")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
