#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agentify_test_session_bot")


@dataclass(frozen=True)
class AgentItem:
    key: str
    label: str
    hf_model: str


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def get_root() -> Path:
    env_root = os.getenv("AGENTIFY_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / ".env").exists() or (candidate / "pipeline").exists():
            return candidate
    return here.parent


def build_agents() -> Dict[str, AgentItem]:
    return {
        "summary": AgentItem("summary", "Summary", _env("HF_MODEL_SUMMARY", "Krasty/agentify-summary-q3_k")),
        "qa": AgentItem("qa", "QA", _env("HF_MODEL_QA", "Krasty/agentify-qa-q3_k")),
        "extraction": AgentItem("extraction", "Extraction", _env("HF_MODEL_EXTRACTION", "Krasty/agentify-extraction-q3_k")),
        "dialogue": AgentItem("dialogue", "Dialogue", _env("HF_MODEL_DIALOGUE", "Krasty/agentify-dialogue-q4_k_m")),
        "telegram": AgentItem("telegram", "Telegram", _env("HF_MODEL_TELEGRAM", "Krasty/agentify-telegram-q4_k_m")),
        "universal": AgentItem("universal", "Universal", _env("HF_MODEL_UNIVERSAL", "Krasty/agentify-universal-q4_k_m")),
    }


def keyboard(agents: Dict[str, AgentItem]) -> InlineKeyboardMarkup:
    rows = []
    order = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
    for i in range(0, len(order), 2):
        pair = order[i : i + 2]
        rows.append([InlineKeyboardButton(agents[k].label, callback_data=f"pick:{k}") for k in pair])
    return InlineKeyboardMarkup(rows)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents = context.application.bot_data["agents"]
    await update.message.reply_text(
        "Это тестовый бот для запуска временной сессии агента.\n"
        "Выбери агента кнопкой и отправь сообщение для теста.",
        reply_markup=keyboard(agents),
    )


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents = context.application.bot_data["agents"]
    await update.message.reply_text(
        "Выбери агента для тестовой сессии.",
        reply_markup=keyboard(agents),
    )


async def on_pick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try:
        await q.answer()
    except BadRequest:
        pass
    data = q.data or ""
    if not data.startswith("pick:"):
        return
    key = data.split(":", 1)[1]
    agents = context.application.bot_data["agents"]
    if key not in agents:
        await q.edit_message_text("Неизвестный агент")
        return
    context.chat_data["selected_agent"] = key
    a = agents[key]
    await q.edit_message_text(f"Агент выбран: {a.label}\nМодель: {a.hf_model}")
    await q.message.reply_text("Теперь отправь текст, и я открою тестовую сессию для этого агента.")


def create_gpu_session(api_url: str, agent_name: str, hf_model: str, user_id: int | None, chat_id: int | None, idle_timeout_sec: int) -> dict:
    payload = {
        "agent_name": agent_name,
        "hf_model": hf_model,
        "user_id": user_id,
        "chat_id": chat_id,
        "idle_timeout_sec": idle_timeout_sec,
    }
    r = requests.post(api_url.rstrip("/") + "/v1/sessions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    agents = context.application.bot_data["agents"]
    api_url = context.application.bot_data["gpu_api_url"]
    key = context.chat_data.get("selected_agent", "universal")
    agent = agents.get(key, agents["universal"])

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        session = await asyncio.to_thread(
            create_gpu_session,
            api_url,
            agent.key,
            agent.hf_model,
            update.effective_user.id if update.effective_user else None,
            update.effective_chat.id if update.effective_chat else None,
            int(context.application.bot_data["idle_timeout_sec"]),
        )
        link = session.get("runtime_url") or f"{api_url.rstrip('/')}/v1/sessions/{session['session_id']}"
        await update.message.reply_text(
            "Тестовая сессия создана.\n"
            f"Агент: {agent.label}\n"
            f"Модель: {agent.hf_model}\n"
            f"Session: {session['session_id']}\n"
            f"Ссылка: {link}\n\n"
            "Пока это заглушка API-контракта. Следующим шагом подключим реальный Docker runtime и скачивание модели с Hugging Face."
        )
    except Exception as e:
        await update.message.reply_text(f"Не удалось создать тестовую сессию: {e}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)


def main():
    load_dotenv()
    token = os.getenv("TG_TEST_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_TEST_BOT_TOKEN in .env")

    api_url = os.getenv("GPU_API_URL", "http://127.0.0.1:8000").strip()
    idle_timeout_sec = int(os.getenv("TEST_SESSION_IDLE_SEC", "900"))
    root = get_root()
    agents = build_agents()

    app = Application.builder().token(token).build()
    app.bot_data["agents"] = agents
    app.bot_data["gpu_api_url"] = api_url
    app.bot_data["idle_timeout_sec"] = idle_timeout_sec
    app.bot_data["root_dir"] = root

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("test", cmd_test))
    app.add_handler(CallbackQueryHandler(on_pick, pattern=r"^pick:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    logger.info("Test bot started. gpu_api=%s root=%s", api_url, root)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
