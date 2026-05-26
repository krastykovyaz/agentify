#!/usr/bin/env python3
"""
Telegram bot with buttons to route requests to different Ollama models.

Env vars:
  TG_BOT_TOKEN=...
  OLLAMA_BASE_URL=http://127.0.0.1:11434

Optional model overrides:
  OLLAMA_MODEL_SUMMARY=agentify:summary_q3_k
  OLLAMA_MODEL_QA=agentify:qa_q3_k
  OLLAMA_MODEL_EXTRACTION=agentify:extraction_q3_k
#   OLLAMA_MODEL_VALIDATOR=agentify:validator_q3_k
  OLLAMA_MODEL_DIALOGUE=agentify:dialogue_q4_k_m
  OLLAMA_MODEL_TELEGRAM=agentify:telegram_q4_k_m
  OLLAMA_MODEL_UNIVERSAL=agentify:universal_q4_k_m
  OLLAMA_MODEL_CODING_WEB=agentify:coding_web_q4_k_m
"""

from __future__ import annotations

import asyncio
import logging
import os, re
from dataclasses import dataclass
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
import re


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_multi_agent_bot")


@dataclass(frozen=True)
class AgentCfg:
    key: str
    title: str
    model: str
    system: str


def _env(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def load_coding_web_prompt() -> str:
    env_path = os.getenv("CODING_WEB_SYSTEM_PROMPT_FILE", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve().parent
    candidates.append(here / "prompts_coding_web_system.txt")
    for p in candidates:
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return "Ты ИИ-ассистент для создания веб-приложений. Выдавай только код."


def build_agents() -> Dict[str, AgentCfg]:
    coding_system = load_coding_web_prompt()
    return {
        "summary": AgentCfg(
            key="summary",
            title="Summary (Q3)",
            model=_env("OLLAMA_MODEL_SUMMARY", "agentify:summary_q3_k"),
            system="Краткое резюме ситуации. Сразу короткий текст по исходному тексту. Без вводных слов и без пояснений.",
        ),
        "qa": AgentCfg(
            key="qa",
            title="QA (Q3)",
            model=_env("OLLAMA_MODEL_QA", "agentify:qa_q3_k"),
            system="Отвечай по фактам кратко и точно.",
        ),
        "extraction": AgentCfg(
            key="extraction",
            title="Extraction (Q3)",
            model=_env("OLLAMA_MODEL_EXTRACTION", "agentify:extraction_q3_k"),
            system=(
                "Извлекай структурированные поля и возвращай строго JSON-строку. "
                "Без префикса 'json', без markdown, без пояснений, только валидный JSON."
            ),
        ),
        # "validator": AgentCfg(
        #     key="validator",
        #     title="Validator (Q3)",
        #     model=_env("OLLAMA_MODEL_VALIDATOR", "agentify:validator_q3_k"),
        #     system="Проверяй качество, находи ошибки и риски, давай четкие замечания.",
        # ),
        "dialogue": AgentCfg(
            key="dialogue",
            title="Dialogue (Q4)",
            model=_env("OLLAMA_MODEL_DIALOGUE", "agentify:dialogue_q4_k_m"),
            system="Ты добрый и харизматичный собеседник. Отвечай бережно, уместно, поддерживающе и по-человечески.",
        ),
        "telegram": AgentCfg(
            key="telegram",
            title="Telegram (Q4)",
            model=_env("OLLAMA_MODEL_TELEGRAM", "agentify:telegram_q4_k_m"),
            system=(
                "Сформируй готовый пост для Telegram-канала по исходному тексту. "
                "Верни только один финальный пост без вариантов, рекомендаций и саммари."
            ),
        ),
        "universal": AgentCfg(
            key="universal",
            title="Universal (Q4)",
            model=_env("OLLAMA_MODEL_UNIVERSAL", "agentify:universal_q4_k_m"),
            system=(
                "Ты универсальный ассистент. Следуй правилам: "
                "для постов — один финальный пост без вариантов/рекомендаций/саммари; "
                "для суммаризации — кратко и сразу по сути без вводных слов; "
                "для extraction — только валидный JSON-строкой без markdown и префиксов; "
                "для coding — давай только рабочие скрипты/код без лишних пояснений, минимально и качественно."
            ),
        ),
        "coding_web": AgentCfg(
            key="coding_web",
            title="Coding Web (Q4)",
            model=_env("OLLAMA_MODEL_CODING_WEB", "agentify:coding_web_q4_k_m"),
            system=coding_system,
        ),
    }


def keyboard(agents: Dict[str, AgentCfg]) -> InlineKeyboardMarkup:
    rows = []
    order = [
        "summary", "qa", "extraction", # "validator",
        "dialogue", "telegram", "universal", "coding_web",
    ]
    for i in range(0, len(order), 2):
        pair = order[i : i + 2]
        rows.append([
            InlineKeyboardButton(agents[k].title, callback_data=f"agent:{k}") for k in pair
        ])
    return InlineKeyboardMarkup(rows)


def get_selected_agent(context: ContextTypes.DEFAULT_TYPE, chat_id: int, agents: Dict[str, AgentCfg]) -> AgentCfg:
    _ = chat_id  # chat_id kept for signature compatibility
    chat_data = context.chat_data
    key = chat_data.get("agent_key", "universal")
    return agents.get(key, agents["universal"])


def set_selected_agent(context: ContextTypes.DEFAULT_TYPE, chat_id: int, key: str) -> None:
    _ = chat_id  # chat_id kept for signature compatibility
    chat_data = context.chat_data
    chat_data["agent_key"] = key


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

def telegram_postprocess(text: str) -> str:
    text = text.replace("###", "•")
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("```", "")
    text = text.replace("$", "")
    # collapse empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ollama_chat(base_url: str, model: str, system: str, user_text: str, timeout: int = 300) -> str:
    base = base_url.rstrip("/")

    # 1) Native Ollama API
    url_chat = base + "/api/chat"
    payload_chat = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }
    r = requests.post(url_chat, json=payload_chat, timeout=timeout)
    if r.status_code < 400:
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip()

    # 2) OpenAI-compatible chat endpoint
    url_oai_chat = base + "/v1/chat/completions"
    payload_oai_chat = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1200,
    }
    r2 = requests.post(url_oai_chat, json=payload_oai_chat, timeout=timeout)
    if r2.status_code < 400:
        data = r2.json()
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
        return ""

    # 3) OpenAI-compatible completions endpoint
    url_oai_comp = base + "/v1/completions"
    prompt = f"System: {system}\n\nUser: {user_text}\nAssistant:"
    payload_oai_comp = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1200,
    }
    r3 = requests.post(url_oai_comp, json=payload_oai_comp, timeout=timeout)
    if r3.status_code < 400:
        data = r3.json()
        choices = data.get("choices") or []
        if choices:
            return str(choices[0].get("text", "")).strip()
        return ""

    # If all failed, report compact diagnostics.
    msg = (
        f"All backends failed for {base_url}\n"
        f"/api/chat -> {r.status_code}: {r.text[:300]}\n"
        f"/v1/chat/completions -> {r2.status_code}: {r2.text[:300]}\n"
        f"/v1/completions -> {r3.status_code}: {r3.text[:300]}"
    )
    raise requests.HTTPError(msg)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    chat_id = update.effective_chat.id
    set_selected_agent(context, chat_id, "universal")
    await update.message.reply_text(
        "Выбери агента кнопкой ниже. Текущий: Universal (Q4)",
        reply_markup=keyboard(agents),
    )


async def cmd_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    chat_id = update.effective_chat.id
    current = get_selected_agent(context, chat_id, agents)
    safe_text = telegram_escape(ch)
    await update.message.reply_text(
        f"Текущий агент: {current.title}\nМодель: {current.model}",
        reply_markup=keyboard(agents),
        parse_mode="MarkdownV2",
    )
    


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest as e:
        msg = str(e).lower()
        if "query is too old" not in msg and "query id is invalid" not in msg:
            raise

    data = query.data or ""
    if not data.startswith("agent:"):
        return
    key = data.split(":", 1)[1]
    if key not in agents:
        await query.edit_message_text("Неизвестный агент.")
        return

    chat_id = query.message.chat_id
    set_selected_agent(context, chat_id, key)
    cfg = agents[key]
    await query.edit_message_text(
        f"Агент переключен: {cfg.title}\nМодель: {cfg.model}",
        reply_markup=keyboard(agents),
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    base_url: str = context.application.bot_data["ollama_base_url"]
    chat_id = update.effective_chat.id
    cfg = get_selected_agent(context, chat_id, agents)
    user_text = update.message.text
    if cfg.key == "coding_web":
        user_text = (
            "Верни результат как чистый текст кода/скрипта для сохранения в файл и запуска. "
            "Не используй markdown-блоки, не добавляй пояснения.\n\n"
            f"{user_text}"
        )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        answer = await asyncio.to_thread(
            ollama_chat,
            base_url,
            cfg.model,
            cfg.system,
            user_text,
            300,
        )
        if not answer:
            answer = "Пустой ответ от модели."
    except Exception as e:
        logger.exception("ollama error")
        answer = f"Ошибка вызова модели `{cfg.model}`: {e}"

    header = f"[{cfg.title}]\n"
    chunks = list(split_text(answer))
    if chunks:
        chunks[0] = header + chunks[0]
    else:
        chunks = [header + "(пустой ответ)"]

    for ch in chunks:
        await update.message.reply_text(telegram_postprocess(ch))


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents: Dict[str, AgentCfg] = context.application.bot_data["agents"]
    lines = ["Доступные агенты:"]
    for k in ["summary", "qa", "extraction", "dialogue", "telegram", "universal", "coding_web"]: # , "validator"
        a = agents[k]
        lines.append(f"- {a.title}: `{a.model}`")
    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled telegram error: %s", context.error)


def main():
    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_BOT_TOKEN in env")

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    agents = build_agents()

    app = Application.builder().token(token).build()
    app.bot_data["agents"] = agents
    app.bot_data["ollama_base_url"] = ollama_base_url

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("agent", cmd_agent))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^agent:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_error_handler(on_error)

    logger.info("Bot started. Ollama: %s", ollama_base_url)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
