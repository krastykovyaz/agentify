#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_orchestrator_bot")


@dataclass(frozen=True)
class AgentCfg:
    key: str
    title: str
    model: str
    system: str


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def find_project_root() -> Path:
    env_root = os.getenv("AGENTIFY_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, here.parent.parent.parent]
    for candidate in candidates:
        if (candidate / "pipeline" / "pipeline_runner.py").exists() or (candidate / ".env").exists():
            return candidate
    return here.parent


def load_prompt(path: str, fallback: str) -> str:
    p = Path(path)
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t
    return fallback


def build_agents(root: Path) -> Dict[str, AgentCfg]:
    prompts_dir = root / "prompts"
    return {
        "summary": AgentCfg("summary", "Summary (Q3)", _env("OLLAMA_MODEL_SUMMARY", "agentify-summary-q3_k"), load_prompt(str(prompts_dir / "p_summary.txt"), "Краткое резюме без вводных слов.")),
        "qa": AgentCfg("qa", "QA (Q3)", _env("OLLAMA_MODEL_QA", "agentify-qa-q3_k"), load_prompt(str(prompts_dir / "p_qa.txt"), "Отвечай по фактам кратко и точно.")),
        "extraction": AgentCfg("extraction", "Extraction (Q3)", _env("OLLAMA_MODEL_EXTRACTION", "agentify-extraction-q3_k"), load_prompt(str(prompts_dir / "p_extraction.txt"), "Только валидный JSON строкой.")),
        "dialogue": AgentCfg("dialogue", "Dialogue (Q4)", _env("OLLAMA_MODEL_DIALOGUE", "agentify-dialogue-q4_k_m"), load_prompt(str(prompts_dir / "p_dialogue.txt"), "Ты добрый и харизматичный собеседник.")),
        "telegram": AgentCfg("telegram", "Telegram (Q4)", _env("OLLAMA_MODEL_TELEGRAM", "agentify-telegram-q4_k_m"), load_prompt(str(prompts_dir / "p_telegram.txt"), "Один готовый пост без вариантов.")),
        "universal": AgentCfg("universal", "Universal (Q4)", _env("OLLAMA_MODEL_UNIVERSAL", "agentify-universal-q4_k_m"), load_prompt(str(prompts_dir / "p_universal.txt"), "Универсальный ассистент.")),
    }


def keyboard(agents: Dict[str, AgentCfg]) -> InlineKeyboardMarkup:
    order = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
    rows = []
    for i in range(0, len(order), 2):
        pair = order[i:i+2]
        rows.append([InlineKeyboardButton(agents[k].title, callback_data=f"agent:{k}") for k in pair])
    return InlineKeyboardMarkup(rows)


def get_agent(context: ContextTypes.DEFAULT_TYPE, agents: Dict[str, AgentCfg]) -> AgentCfg:
    key = context.chat_data.get("agent_key", "universal")
    return agents.get(key, agents["universal"])


def split_text(text: str, limit: int = 3900):
    text = text or ""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut]
        text = text[cut:]
    if text:
        yield text


def ollama_chat(base_url: str, model: str, system: str, user_text: str, timeout: int = 300) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    context.chat_data["agent_key"] = "universal"
    await update.message.reply_text(
        "Готово. Выбери агента кнопкой ниже или отправь файл (csv/json/txt) для pipeline до 1000 примеров.",
        reply_markup=keyboard(agents),
    )


async def cmd_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    cur = get_agent(context, agents)
    await update.message.reply_text(
        f"Текущий агент: {cur.title}\nМодель: {cur.model}",
        reply_markup=keyboard(agents),
    )


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    lines = ["Доступные агенты:"]
    for k in ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]:
        a = agents[k]
        lines.append(f"- {a.title}: {a.model}")
    await update.message.reply_text("\n".join(lines))


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    q = update.callback_query
    try:
        await q.answer()
    except BadRequest:
        pass

    data = q.data or ""
    if not data.startswith("agent:"):
        return
    key = data.split(":", 1)[1]
    if key not in agents:
        await q.edit_message_text("Неизвестный агент")
        return

    context.chat_data["agent_key"] = key
    a = agents[key]
    await q.edit_message_text(f"Агент переключен: {a.title}\nМодель: {a.model}", reply_markup=keyboard(agents))


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    base_url: str = context.application.bot_data["ollama_base_url"]
    cfg = get_agent(context, agents)
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        ans = await asyncio.to_thread(ollama_chat, base_url, cfg.model, cfg.system, user_text, 300)
    except Exception as e:
        ans = f"Ошибка вызова модели {cfg.model}: {e}"

    chunks = list(split_text(f"[{cfg.title}]\n{ans}"))
    for c in chunks:
        await update.message.reply_text(c)


async def run_pipeline_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE, input_file: Path):
    root: Path = context.application.bot_data["root_dir"]
    pipeline_script: Path = context.application.bot_data["pipeline_script"]
    pipeline_cfg: Path = context.application.bot_data["pipeline_config"]

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_csv = run_dir / "pipeline_train_1000.csv"
    out_report = run_dir / "pipeline_train_1000.report.json"

    await update.message.reply_text("Pipeline запущен: intake -> normalize -> trim/augment -> materialize")

    cmd = [
        "python3",
        str(pipeline_script),
        "--input", str(input_file),
        "--config", str(pipeline_cfg),
        "--output-csv", str(out_csv),
        "--report-json", str(out_report),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(root),
    )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = (stderr.decode("utf-8", errors="ignore") or stdout.decode("utf-8", errors="ignore"))[:3500]
        await update.message.reply_text(f"Pipeline ошибка (code={proc.returncode}):\n{msg}")
        return

    await update.message.reply_text("Pipeline завершен. Отправляю датасет и отчет.")

    if out_csv.exists():
        await update.message.reply_document(document=out_csv.open("rb"), filename=out_csv.name)
    if out_report.exists():
        await update.message.reply_document(document=out_report.open("rb"), filename=out_report.name)


def allowed_file(name: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in {".csv", ".json", ".txt"}


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return
    doc = update.message.document
    if not allowed_file(doc.file_name or ""):
        await update.message.reply_text("Поддерживаются только .csv .json .txt")
        return

    root: Path = context.application.bot_data["root_dir"]
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    local_path = inbox / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{doc.file_name}"
    tg_file = await context.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(str(local_path))

    await update.message.reply_text(f"Файл получен: {local_path.name}")
    await run_pipeline_and_send(update, context, local_path)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


def main():
    load_dotenv()

    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_BOT_TOKEN in env/.env")

    root = find_project_root()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()

    pipeline_script = root / "pipeline" / "pipeline_runner.py"
    pipeline_cfg = root / "pipeline" / "pipeline_config.yaml"
    if not pipeline_script.exists() or not pipeline_cfg.exists():
        raise SystemExit("pipeline files not found")

    agents = build_agents(root)

    app = Application.builder().token(token).build()
    app.bot_data["root_dir"] = root
    app.bot_data["ollama_base_url"] = base_url
    app.bot_data["agents"] = agents
    app.bot_data["pipeline_script"] = pipeline_script
    app.bot_data["pipeline_config"] = pipeline_cfg

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("agent", cmd_agent))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^agent:"))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    logger.info("Bot started. root=%s ollama=%s", root, base_url)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
