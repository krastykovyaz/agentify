#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


def find_project_root() -> Path:
    env_root = os.getenv("AGENTIFY_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    candidates = [here.parent.parent, here.parent.parent.parent]
    for candidate in candidates:
        if (candidate / "pipeline" / "pipeline_runner.py").exists() or (candidate / ".env").exists():
            return candidate
    return here.parent.parent


def send_tg(token: str, chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=30)


def free_disk_gb(path: Path) -> float:
    st = os.statvfs(str(path))
    return (st.f_bavail * st.f_frsize) / (1024**3)


def free_gpu_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if not out:
            return None
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return vals[0] if vals else None
    except Exception:
        return None


def can_run(root: Path) -> tuple[bool, str]:
    min_disk = float(os.getenv("TRAIN_MIN_FREE_DISK_GB", "30"))
    min_gpu = int(os.getenv("TRAIN_MIN_FREE_GPU_MB", "20000"))
    d = free_disk_gb(root)
    g = free_gpu_mb()
    if d < min_disk:
        return False, f"disk {d:.1f}GB < {min_disk}GB"
    if g is not None and g < min_gpu:
        return False, f"gpu_free {g}MB < {min_gpu}MB"
    return True, f"disk {d:.1f}GB, gpu_free {g if g is not None else 'n/a'}MB"


def run_job(root: Path, job: dict) -> tuple[int, str]:
    cmd = shlex.split(job["cmd"])
    p = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    log = ((out or "") + "\n" + (err or "")).strip()
    return p.returncode, log[-3500:]


def publish_run(root: Path, job: dict) -> tuple[int, str]:
    script = root / "pipeline" / "publish_run_to_hf.py"
    cmd = [
        "python3",
        str(script),
        "--outdir",
        str(job["outdir"]),
        "--run-id",
        str(job.get("run_id") or job.get("id") or "queued-run"),
        "--dataset",
        str(job.get("dataset", "")),
        "--report",
        str(job.get("report", "")),
    ]
    p = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    log = ((out or "") + "\n" + (err or "")).strip()
    return p.returncode, log[-3500:]


def extract_hf_link(text: str) -> str:
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("https://huggingface.co/"):
            return line
    return ""


def main():
    load_dotenv()
    root = find_project_root()
    qdir = root / "queue" / "train"
    qdir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("TG_BOT_TOKEN", "").strip()
    interval = int(os.getenv("TRAIN_QUEUE_POLL_SEC", "60"))

    while True:
        jobs = sorted(qdir.glob("*.json"))
        if not jobs:
            time.sleep(interval)
            continue

        ok, why = can_run(root)
        if not ok:
            time.sleep(interval)
            continue

        job_file = jobs[0]
        job = json.loads(job_file.read_text(encoding="utf-8"))
        chat_id = int(job.get("chat_id", 0))
        if token and chat_id:
            send_tg(token, chat_id, f"Очередь: запускаю обучение ({job_file.name})\n{why}")

        code, log = run_job(root, job)
        if code == 0:
            pub_code, pub_log = publish_run(root, job)
            if token and chat_id:
                if pub_code == 0:
                    link = extract_hf_link(pub_log) or pub_log
                    send_tg(token, chat_id, f"Обучение и публикация завершены.\nСсылка на агента: {link}")
                else:
                    send_tg(token, chat_id, f"Обучение завершено, но публикация не удалась:\n{pub_log}")
            job_file.unlink(missing_ok=True)
        else:
            if token and chat_id:
                send_tg(token, chat_id, f"Ошибка обучения (code={code})\n{log}")
            # leave file for retry unless max retries reached
            retries = int(job.get("retries", 0)) + 1
            job["retries"] = retries
            if retries >= int(os.getenv("TRAIN_QUEUE_MAX_RETRIES", "3")):
                failed = qdir / (job_file.stem + ".failed.json")
                failed.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
                job_file.unlink(missing_ok=True)
            else:
                job_file.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(2)


if __name__ == "__main__":
    main()
