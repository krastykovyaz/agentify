#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import logging
import os
from urllib.parse import quote
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


def bot_username() -> str:
    return _env("TG_TEST_BOT_USERNAME", "")


def deep_link(session_id: str) -> str:
    username = bot_username()
    if not username:
        return f"session:{session_id}"
    return f"https://t.me/{username}?start={quote(session_id)}"


def keyboard(agents: Dict[str, AgentItem]) -> InlineKeyboardMarkup:
    rows = []
    order = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
    for i in range(0, len(order), 2):
        pair = order[i : i + 2]
        rows.append([InlineKeyboardButton(agents[k].label, callback_data=f"pick:{k}") for k in pair])
    return InlineKeyboardMarkup(rows)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agents = context.application.bot_data["agents"]
    args = context.args or []
    if args:
        session_id = args[0].strip()
        if session_id:
            api_url = context.application.bot_data["gpu_api_url"]
            try:
                session = await asyncio.to_thread(get_gpu_session, api_url, session_id)
                context.chat_data["session_id"] = session_id
                context.chat_data["selected_agent"] = str(session.get("agent_name") or "universal")
                await update.message.reply_text(
                    "Тестовая сессия подключена.\n"
                    f"Session: {session_id}\n"
                    f"Агент: {session.get('agent_name')}\n"
                    f"Модель: {session.get('hf_model')}\n"
                    "Скачиваю модель с Hugging Face, подожди..."
                )
                ready, status = await ensure_session_ready(update, api_url, session_id)
                if not ready:
                    return
                runtime_model = str(status.get("runtime_model") or session.get("runtime_model") or "")
                await update.message.reply_text(
                    "Модель готова. Отправь текст, и я передам его в GPU API.\n"
                    f"Runtime: {runtime_model}"
                )
                return
            except Exception as e:
                await update.message.reply_text(f"Не удалось подключить сессию {session_id}: {e}")
                return
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
    context.chat_data.pop("session_id", None)
    a = agents[key]
    await q.edit_message_text(f"Агент выбран: {a.label}\nМодель: {a.hf_model}")
    await q.message.reply_text("Теперь отправь текст, и я открою тестовую сессию для этого агента.")


def create_gpu_session(api_url: str, agent_name: str, hf_model: str, user_id: int | None, chat_id: int | None, idle_timeout_sec: int) -> dict:
    payload = {
        "agent_name": agent_name,
        "hf_model": hf_model,
        "runtime_model": hf_model,
        "user_id": user_id,
        "chat_id": chat_id,
        "idle_timeout_sec": idle_timeout_sec,
    }
    r = requests.post(api_url.rstrip("/") + "/v1/sessions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def reply_gpu_session(api_url: str, session_id: str, text: str) -> dict:
    r = requests.post(
        api_url.rstrip("/") + f"/v1/sessions/{session_id}/reply",
        json={"text": text},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def get_gpu_session(api_url: str, session_id: str) -> dict:
    r = requests.get(api_url.rstrip("/") + f"/v1/sessions/{session_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def launch_gpu_session(api_url: str, session_id: str) -> dict:
    r = requests.post(api_url.rstrip("/") + f"/v1/sessions/{session_id}/launch", timeout=600)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return {"error": detail, "status_code": r.status_code}
    return r.json()


async def wait_for_session_ready(api_url: str, session_id: str, timeout_sec: int = 300) -> tuple[bool, dict]:
    deadline = asyncio.get_event_loop().time() + timeout_sec
    last_status: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        try:
            last_status = await asyncio.to_thread(get_gpu_session, api_url, session_id)
        except Exception:
            last_status = {}
        notes = str(last_status.get("notes") or "")
        runtime_model = str(last_status.get("runtime_model") or "")
        state = str(last_status.get("state") or "")
        if runtime_model.endswith(".gguf") or "downloaded" in notes:
            return True, last_status
        await asyncio.sleep(2)
    return False, last_status


async def ensure_session_ready(update: Update, api_url: str, session_id: str) -> tuple[bool, dict]:
    launch = await asyncio.to_thread(launch_gpu_session, api_url, session_id)
    if launch.get("error"):
        await update.message.reply_text(
            f"Не удалось запустить сессию: {launch.get('error')}\n"
            "Проверь, что на GPU-сервере обновлён bridge/gpu_session_api.py."
        )
        return False, launch

    ready, status = await wait_for_session_ready(api_url, session_id)
    if not ready:
        state = str(status.get("state") or launch.get("state") or "unknown")
        notes = str(status.get("notes") or launch.get("notes") or "")
        await update.message.reply_text(
            f"Модель ещё не готова (статус: {state}).\n{notes}\n"
            "Подожди минуту и отправь сообщение снова."
        )
        return False, status
    return True, status


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    agents = context.application.bot_data["agents"]
    api_url = context.application.bot_data["gpu_api_url"]
    key = context.chat_data.get("selected_agent", "universal")
    agent = agents.get(key, agents["universal"])
    session_id = context.chat_data.get("session_id")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        if not session_id:
            session = await asyncio.to_thread(
                create_gpu_session,
                api_url,
                agent.key,
                agent.hf_model,
                update.effective_user.id if update.effective_user else None,
                update.effective_chat.id if update.effective_chat else None,
                int(context.application.bot_data["idle_timeout_sec"]),
            )
            session_id = session["session_id"]
            context.chat_data["session_id"] = session_id

            # Try to launch and wait until model is downloaded (with timeout)
            launch = await asyncio.to_thread(launch_gpu_session, api_url, session_id)
            link = deep_link(session_id)
            launch_state = str(launch.get("state") or session.get("state") or "queued")
            launch_note = str(launch.get("notes") or launch.get("error") or session.get("notes") or "")

            # If queued or download in progress, poll status for up to 60s
            if launch_state != "running":
                await update.message.reply_text(
                    "Сессия создана и запуск в процессе. Ожидаю готовности модели (до 60s)..."
                )
                ready = False
                for _ in range(30):
                    await asyncio.sleep(2)
                    try:
                        status = await asyncio.to_thread(get_gpu_session, api_url, session_id)
                    except Exception:
                        status = {}
                    state = str(status.get("state") or "")
                    notes = str(status.get("notes") or "")
                    runtime_model = str(status.get("runtime_model") or "")
                    if "downloaded" in notes or runtime_model.endswith(".gguf"):
                        ready = True
                        launch_state = state or "running"
                        launch_note = notes
                        break
                if not ready:
                    await update.message.reply_text(
                        f"Модель не готова: статус {launch_state}. Примечание: {launch_note}"
                    )
                    return

            await update.message.reply_text(
                "Тестовая сессия создана и готова.\n"
                f"Агент: {agent.label}\n"
                f"Модель: {agent.hf_model}\n"
                f"Session: {session_id}\n"
                f"Ссылка на тест: {link}\n\n"
                f"Статус запуска: {launch_state}\n"
                f"{launch_note}\n\n"
                "Открой ссылку в Telegram, чтобы продолжить тестирование в отдельной сессии."
            )
            return

        ready, _ = await ensure_session_ready(update, api_url, session_id)
        if not ready:
            return

        reply = await asyncio.to_thread(reply_gpu_session, api_url, session_id, update.message.text)
        text = str(reply.get("reply") or "").strip()
        model = str(reply.get("model") or agent.hf_model)
        if text.startswith("[stub reply]"):
            await update.message.reply_text(
                f"Модель не ответила.\n{text}\n\n"
                "Перезапусти на GPU: python3 bridge/gpu_session_api.py"
            )
            return
        await update.message.reply_text(
            f"[{agent.label}]\nМодель: {model}\n\n{text}"
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка тестовой сессии: {e}")


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
