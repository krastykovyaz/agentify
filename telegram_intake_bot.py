#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import os
import re
import asyncio
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_intake_bot")

TARGET = 1000
AGENT_STYLES = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
STYLE_MODEL = os.getenv("OLLAMA_MODEL_UNIVERSAL", "agentify-universal-q4_k_m")


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


def infer_style_via_universal(base_url: str, text: str) -> str:
    sys = (
        "Классифицируй текст строго в один стиль: summary|qa|extraction|dialogue|telegram|universal. "
        "Верни только одно слово из списка."
    )
    payload = {
        "model": STYLE_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": text[:3000]},
        ],
        "options": {"temperature": 0.0, "top_p": 0.9, "num_ctx": 4096},
    }
    try:
        r = requests.post(base_url.rstrip("/") + "/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        out = (r.json().get("message") or {}).get("content", "").strip().lower()
        for s in AGENT_STYLES:
            if s in out:
                return s
    except Exception:
        pass
    return infer_style(text)


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


def build_report(texts: List[str], styles: List[str]) -> str:
    n = len(texts)
    if n == 0:
        return "Не нашел валидных текстов в файле. Поддерживаются csv/json/txt."

    lens = [len(t) for t in texts]
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


def decision_keyboard(step: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("Да", callback_data=f"flow:{step}:yes"),
            InlineKeyboardButton("Нет", callback_data=f"flow:{step}:no"),
        ]]
    )


async def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    txt = (out.decode("utf-8", errors="ignore") + "\n" + err.decode("utf-8", errors="ignore")).strip()
    return proc.returncode, txt[:3500]


def allowed_file(name: str) -> bool:
    return Path(name or "").suffix.lower() in {".csv", ".json", ".txt"}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Отправь файл с текстами (csv/json/txt). Дальше пойдем пошагово: анализ -> подготовка датасета -> обучение -> финальный ответ со ссылкой."
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
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        # universal-style detection (with fallback) on full set up to 1000 items
        styles = [infer_style_via_universal(base_url, t) for t in texts[:1000]]
        if len(texts) > 1000:
            # extend remaining using heuristic for speed
            styles.extend(infer_style(t) for t in texts[1000:])
        report = build_report(texts, styles)

        context.chat_data["flow"] = {
            "input_file": str(local_path),
            "input_name": local_name,
            "run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        }
    except Exception as e:
        report = f"Ошибка обработки файла: {e}"
        context.chat_data.pop("flow", None)

    await update.message.reply_text(report)
    if context.chat_data.get("flow"):
        await update.message.reply_text("Запускаем подготовку датасета?", reply_markup=decision_keyboard("prepare"))


async def on_flow_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    if not data.startswith("flow:"):
        return
    _, step, decision = data.split(":", 2)

    flow = context.chat_data.get("flow")
    if not flow:
        await q.edit_message_text("Нет активной сессии. Отправь файл заново.")
        return

    root = Path(os.getenv("AGENTIFY_ROOT", "/home/aleksandr.koviazin/kovyaz/agentify")).resolve()
    run_dir = root / "runs" / flow["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ds_csv = run_dir / "pipeline_train_1000.csv"
    ds_report = run_dir / "pipeline_train_1000.report.json"

    if step == "prepare":
        if decision == "no":
            await q.edit_message_text("Ок, остановились на этапе анализа.")
            return
        await q.edit_message_text("Запускаю подготовку датасета...")
        code, log = await run_cmd([
            "python3", str(root / "pipeline" / "pipeline_runner.py"),
            "--input", flow["input_file"],
            "--config", str(root / "pipeline" / "pipeline_config.yaml"),
            "--output-csv", str(ds_csv),
            "--report-json", str(ds_report),
        ], root)
        if code != 0:
            await q.message.reply_text(f"Подготовка завершилась с ошибкой:\n{log}")
            return
        await q.message.reply_text("Подготовка датасета завершена. Запускаем обучение?", reply_markup=decision_keyboard("train"))
        return

    if step == "train":
        if decision == "no":
            await q.edit_message_text("Ок, датасет готов. Обучение не запускал.")
            if ds_csv.exists():
                await q.message.reply_document(document=ds_csv.open("rb"), filename=ds_csv.name)
            if ds_report.exists():
                await q.message.reply_document(document=ds_report.open("rb"), filename=ds_report.name)
            return
        await q.edit_message_text("Запускаю обучение... Это может занять время.")

        # Training command is configurable; keep simple default
        train_cmd = os.getenv("PIPELINE_TRAIN_CMD", "").strip()
        if not train_cmd:
            await q.message.reply_text(
                "Не задан PIPELINE_TRAIN_CMD. Укажи команду обучения в .env, например:\n"
                "PIPELINE_TRAIN_CMD=python3 /home/aleksandr.koviazin/kovyaz/agentify/sft_train_gemma_universal.py --csv-path {DATASET} --output-dir {OUTDIR}"
            )
            return

        outdir = run_dir / "model_out"
        cmd = train_cmd.replace("{DATASET}", str(ds_csv)).replace("{OUTDIR}", str(outdir))
        code, log = await run_cmd(shlex.split(cmd), root)
        if code != 0:
            await q.message.reply_text(f"Обучение завершилось с ошибкой:\n{log}")
            return

        hf_link = os.getenv("PIPELINE_LAST_HF_LINK", "").strip() or "(ссылка не указана)"
        test_hint = os.getenv("PIPELINE_TEST_HINT", "Отправь тестовый запрос в этого же бота.")
        await q.message.reply_text(
            "Готово!\n"
            f"Ссылка на агента: {hf_link}\n"
            f"Тестировать можно здесь: {test_hint}"
        )
        return


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)


def main():
    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_BOT_TOKEN in .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_flow_button, pattern=r"^flow:"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    logger.info("Intake bot started")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
