#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import os
import re
import asyncio
import subprocess
import shlex
import uuid
from urllib.parse import quote
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

TARGET = int(os.getenv("DATASET_TARGET_LIMIT", "1000"))
AGENT_STYLES = ["summary", "qa", "extraction", "dialogue", "telegram", "universal"]
STYLE_MODEL = os.getenv("OLLAMA_MODEL_UNIVERSAL", "agentify-universal-q4_k_m")
QA_MODEL = os.getenv("OLLAMA_MODEL_QA", "agentify:qa_q3_k")
STYLE_SAMPLE_LIMIT = int(os.getenv("STYLE_SAMPLE_LIMIT", "24"))


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


def ask_qa_style_ratio(base_url: str, texts: List[str]) -> dict | None:
    sample = "\n\n---\n\n".join(texts[:40])
    system = (
        "Определи доли стилей для датасета. Верни только JSON-объект с ключами "
        "summary, qa, extraction, dialogue, telegram, universal. "
        "Значения — доли от 0 до 1, сумма должна быть 1."
    )
    user = f"Тексты выборки:\n{sample}"
    payload = {
        "model": QA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.0, "top_p": 0.9, "num_ctx": 8192},
    }
    try:
        r = requests.post(base_url.rstrip("/") + "/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        out = (r.json().get("message") or {}).get("content", "").strip()
        m = re.search(r"\{.*\}", out, flags=re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(0))
        vals = {}
        for k in AGENT_STYLES:
            vals[k] = max(0.0, float(obj.get(k, 0.0)))
        s = sum(vals.values())
        if s <= 0:
            return None
        return {k: vals[k] / s for k in AGENT_STYLES}
    except Exception:
        return None


def infer_style_via_universal_raw(base_url: str, text: str) -> str | None:
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
    return None


def classify_style_with_consensus(base_url: str, text: str) -> tuple[str, str, str, bool]:
    """
    Returns:
      final_style, agent_style, heuristic_style, agreed
    Rules:
      - if agent and heuristic agree -> use that
      - if disagree -> use agent
      - if agent failed -> use heuristic
    """
    heuristic = infer_style(text)
    agent = infer_style_via_universal_raw(base_url, text)
    if agent is None:
        return heuristic, "failed", heuristic, False
    if agent == heuristic:
        return agent, agent, heuristic, True
    return agent, agent, heuristic, False


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


def build_report(texts: List[str], styles: List[str], qa_ratio: dict | None, agree_n: int, agent_failed_n: int) -> str:
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
    if n > 0:
        lines.append(f"Сходимость агент/эвристика: {agree_n}/{n} ({agree_n*100/n:.0f}%)")
        lines.append(f"Фолбэк на эвристики (агент не сработал): {agent_failed_n}")
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
        # Recommended ratios: prefer QA-agent output, fallback to inverse rarity.
        if qa_ratio:
            weights = qa_ratio
        else:
            inv = {k: 1.0 / (dist.get(k, 0) + 1) for k in AGENT_STYLES}
            s_inv = sum(inv.values())
            weights = {k: inv[k] / s_inv for k in AGENT_STYLES}
        s = sum(weights.values()) or 1.0
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


async def publish_to_hf(root: Path, run_id: str, outdir: Path, dataset: Path, report: Path) -> tuple[bool, str]:
    script = root / "pipeline" / "publish_run_to_hf.py"
    if not script.exists():
        return False, f"publish script not found: {script}"
    code, log = await run_cmd(
        [
            "python3",
            str(script),
            "--outdir",
            str(outdir),
            "--run-id",
            run_id,
            "--dataset",
            str(dataset),
            "--report",
            str(report),
        ],
        root,
    )
    if code != 0:
        return False, log
    link = ""
    for line in reversed(log.splitlines()):
        if line.strip().startswith("https://huggingface.co/"):
            link = line.strip()
            break
    return True, link or log.strip()


def free_disk_gb(path: Path) -> float:
    st = os.statvfs(str(path))
    return (st.f_bavail * st.f_frsize) / (1024**3)


def free_gpu_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return vals[0] if vals else None
    except Exception:
        return None


def resource_check(root: Path) -> tuple[bool, str]:
    min_disk = float(os.getenv("TRAIN_MIN_FREE_DISK_GB", "30"))
    min_gpu = int(os.getenv("TRAIN_MIN_FREE_GPU_MB", "20000"))
    d = free_disk_gb(root)
    g = free_gpu_mb()
    if d < min_disk:
        return False, f"disk {d:.1f}GB < {min_disk}GB"
    if g is not None and g < min_gpu:
        return False, f"gpu_free {g}MB < {min_gpu}MB"
    return True, f"disk {d:.1f}GB, gpu_free {g if g is not None else 'n/a'}MB"


def allowed_file(name: str) -> bool:
    return Path(name or "").suffix.lower() in {".csv", ".json", ".txt"}


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


def find_training_script(root: Path) -> Path:
    candidates = [
        root / "sft_train_gemma_universal.py",
        root / "sft_train_gemma_coding.py",
        root / "sft_train_gemma_coding_web_format.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_test_bot_link(session_id: str) -> str:
    username = os.getenv("TG_TEST_BOT_USERNAME", "").strip()
    if not username:
        return f"session:{session_id}"
    return f"https://t.me/{username}?start={quote(session_id)}"


def normalize_train_cmd(train_cmd: str, root: Path, dataset: Path, outdir: Path) -> str:
    cmd = train_cmd.replace("{DATASET}", str(dataset)).replace("{OUTDIR}", str(outdir))
    cmd = cmd.replace("{ROOT}", str(root))
    parts = shlex.split(cmd)
    if len(parts) >= 2 and parts[0].startswith("python"):
        script_path = Path(parts[1])
        if script_path.is_absolute() and not script_path.exists():
            local_candidate = root / script_path.name
            if local_candidate.exists():
                parts[1] = str(local_candidate)
    return shlex.join(parts)


def create_test_session(api_url: str, agent_name: str, hf_model: str, chat_id: int | None, idle_timeout_sec: int = 900) -> dict:
    payload = {
        "agent_name": agent_name,
        "hf_model": hf_model,
        "runtime_model": hf_model,
        "chat_id": chat_id,
        "idle_timeout_sec": idle_timeout_sec,
    }
    r = requests.post(api_url.rstrip("/") + "/v1/sessions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


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

    root = find_project_root()
    inbox = root / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    local_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{doc.file_name}"
    local_path = inbox / local_name

    tg_file = await context.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(str(local_path))

    await update.message.reply_text(f"Файл принят: {local_name}. Анализирую...")
    await update.message.reply_text("Анализирую стиль и распределение, это может занять немного времени...")

    try:
        texts = read_texts(local_path)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        styles = []
        agree_n = 0
        agent_failed_n = 0
        qa_ratio = ask_qa_style_ratio(base_url, texts[:STYLE_SAMPLE_LIMIT])
        # Use the universal agent only on a bounded sample to keep the bot responsive.
        sample_n = min(len(texts), STYLE_SAMPLE_LIMIT)
        for i, t in enumerate(texts):
            if i < sample_n:
                final_style, agent_style, heuristic_style, agreed = classify_style_with_consensus(base_url, t)
                _ = heuristic_style
                styles.append(final_style)
                if agreed:
                    agree_n += 1
                if agent_style == "failed":
                    agent_failed_n += 1
            else:
                styles.append(infer_style(t))
        report = build_report(texts, styles, qa_ratio, agree_n, agent_failed_n)

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

    root = find_project_root()
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
        msg = "Подготовка датасета завершена."
        try:
            rep = json.loads(ds_report.read_text(encoding="utf-8"))
            orig = int(rep.get("original_rows", 0))
            final = int(rep.get("final_rows", 0))
            src_dist = rep.get("source_distribution", {}) or {}
            synth = 0
            for k, v in src_dist.items():
                if str(k).startswith("synthetic:"):
                    synth += int(v)
            target = int(rep.get("target", 1000))
            status = "достигнут" if final >= target else "НЕ достигнут"
            msg = (
                f"Подготовка датасета завершена.\n"
                f"- Было real: {orig}\n"
                f"- Стало всего: {final}\n"
                f"- Добавлено synthetic: {synth}\n"
                f"- Target {target}: {status}"
            )
        except Exception:
            pass
        await q.message.reply_text(msg)
        await q.message.reply_text("Запускаем обучение?", reply_markup=decision_keyboard("train"))
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
            train_script = find_training_script(root)
            await q.message.reply_text(
                "Не задан PIPELINE_TRAIN_CMD. Укажи команду обучения в .env, например:\n"
                f"PIPELINE_TRAIN_CMD=python3 {train_script} --csv-path {{DATASET}} --output-dir {{OUTDIR}}"
            )
            return

        outdir = run_dir / "model_out"
        cmd = normalize_train_cmd(train_cmd, root, ds_csv, outdir)
        ok, why = resource_check(root)
        if ok:
            await q.message.reply_text(f"Ресурсы достаточны, запускаю сразу ({why})")
            code, log = await run_cmd(shlex.split(cmd), root)
            if code != 0:
                await q.message.reply_text(f"Обучение завершилось с ошибкой:\n{log}")
                return
            await q.message.reply_text("Обучение завершено. Загружаю веса на Hugging Face...")
            ok_pub, pub_result = await publish_to_hf(root, flow["run_id"], outdir, ds_csv, ds_report)
            hf_link = pub_result if ok_pub else f"публикация не удалась: {pub_result}"
            test_hint = os.getenv("PIPELINE_TEST_HINT", "Отправь тестовый запрос в этого же бота.")
            test_link = ""
            gpu_api = os.getenv("GPU_API_URL", "").strip()
            if ok_pub and gpu_api:
                try:
                    session = await asyncio.to_thread(
                        create_test_session,
                        gpu_api,
                        "universal",
                        hf_link,
                        q.message.chat_id,
                        int(os.getenv("TEST_SESSION_IDLE_SEC", "900")),
                    )
                    test_link = build_test_bot_link(str(session.get("session_id", "")))
                    if test_link.startswith("session:"):
                        test_link = ""
                except Exception as e:
                    logger.warning("test session creation failed: %s", e)
            lines = [
                "Готово!",
                f"Ссылка на агента: {hf_link}",
            ]
            if test_link:
                lines.append(f"Ссылка на тест-бота: {test_link}")
            lines.append(f"Тестировать можно здесь: {test_hint}")
            await q.message.reply_text(
                "\n".join(lines)
            )
            return

        # queue when not enough resources
        qdir = root / "queue" / "train"
        qdir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": str(uuid.uuid4()),
            "chat_id": q.message.chat_id,
            "cmd": cmd,
            "dataset": str(ds_csv),
            "outdir": str(outdir),
            "run_id": flow["run_id"],
            "report": str(ds_report),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "retries": 0,
        }
        jf = qdir / f"{flow['run_id']}_{job['id'][:8]}.json"
        jf.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        await q.message.reply_text(
            "Ресурсов сейчас недостаточно. Задача поставлена в очередь.\n"
            f"Причина: {why}\n"
            f"Файл очереди: {jf.name}\n"
            "Как только ресурсы освободятся, воркер автоматически запустит обучение и пришлет уведомление."
        )
        return


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)


def main():
    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TG_BOT_TOKEN in .env")

    global TARGET
    TARGET = int(os.getenv("DATASET_TARGET_LIMIT", "1000"))

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
