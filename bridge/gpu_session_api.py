#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import tempfile
import uuid as uuidlib

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class SessionCreate(BaseModel):
    agent_name: str = Field(..., min_length=1)
    hf_model: str = Field(..., min_length=1)
    runtime_model: str | None = None
    user_id: int | None = None
    chat_id: int | None = None
    idle_timeout_sec: int = Field(default=900, ge=60)
    callback_url: str | None = None


class SessionStatus(BaseModel):
    session_id: str
    agent_name: str
    hf_model: str
    runtime_model: str | None = None
    state: str
    created_at: str
    idle_timeout_sec: int
    user_id: int | None = None
    chat_id: int | None = None
    callback_url: str | None = None
    runtime_url: str | None = None
    notes: str | None = None


class TrainJobCreate(BaseModel):
    run_id: str = Field(..., min_length=1)
    dataset_csv: str = Field(..., min_length=1)
    report_json: str = Field(..., min_length=1)
    train_cmd: str = Field(..., min_length=1)
    publish_cmd: str = Field(..., min_length=1)
    workdir: str | None = None
    user_id: int | None = None
    chat_id: int | None = None
    callback_url: str | None = None
    idle_timeout_sec: int = Field(default=900, ge=60)


class TrainJobStatus(BaseModel):
    job_id: str
    run_id: str
    state: str
    created_at: str
    dataset_csv: str
    report_json: str
    train_cmd: str
    publish_cmd: str
    workdir: str | None = None
    user_id: int | None = None
    chat_id: int | None = None
    callback_url: str | None = None
    hf_link: str | None = None
    notes: str | None = None


app = FastAPI(title="Agentify GPU Session API", version="0.1.0")
ROOT = Path(os.getenv("AGENTIFY_ROOT", Path(__file__).resolve().parent.parent)).resolve()
SESSIONS_DIR = Path(os.getenv("GPU_SESSION_DIR", str(ROOT / "runs" / "sessions"))).resolve()
JOBS_DIR = Path(os.getenv("GPU_TRAIN_JOBS_DIR", str(ROOT / "runs" / "train_jobs"))).resolve()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _gpu_free_mb() -> int | None:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip()
        vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
        return vals[0] if vals else None
    except Exception:
        return None


def _can_launch() -> tuple[bool, str]:
    min_gpu = int(os.getenv("GPU_MIN_FREE_MB", "20000"))
    g = _gpu_free_mb()
    if g is None:
        return False, "gpu memory unavailable"
    if g < min_gpu:
        return False, f"gpu_free {g}MB < {min_gpu}MB"
    return True, f"gpu_free {g}MB"


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_session(data: dict) -> None:
    _session_path(data["session_id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_job(data: dict) -> None:
    _job_path(data["job_id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_job(job_id: str) -> dict:
    p = _job_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_subprocess(cmd: str, cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd) if cwd else str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, (out or "")[-10000:]


def _gpu_ready() -> tuple[bool, str]:
    ok, why = _can_launch()
    return ok, why


@app.get("/health")
def health():
    return {"ok": True, "service": "gpu-session-api"}


@app.post("/v1/sessions", response_model=SessionStatus)
def create_session(payload: SessionCreate):
    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "session_id": session_id,
        "agent_name": payload.agent_name,
        "hf_model": payload.hf_model,
        "runtime_model": payload.runtime_model or payload.hf_model,
        "state": "queued",
        "created_at": now,
        "idle_timeout_sec": int(payload.idle_timeout_sec),
        "user_id": payload.user_id,
        "chat_id": payload.chat_id,
        "callback_url": payload.callback_url,
        "runtime_url": None,
        "notes": "session accepted; runtime launch not implemented yet",
    }
    _write_session(data)
    return data


@app.get("/v1/sessions/{session_id}", response_model=SessionStatus)
def get_session(session_id: str):
    return _read_session(session_id)


@app.post("/v1/sessions/{session_id}/launch", response_model=SessionStatus)
def launch_session(session_id: str):
    data = _read_session(session_id)
    ok, why = _can_launch()
    if not ok:
        data["state"] = "queued"
        data["notes"] = why
        _write_session(data)
        raise HTTPException(status_code=409, detail=why)
    data["state"] = "running"
    data["runtime_url"] = data.get("runtime_url") or f"http://localhost:8000/sessions/{session_id}"
    data["notes"] = "placeholder launch; integrate Docker/HF download next"
    _write_session(data)
    return data


@app.post("/v1/sessions/{session_id}/stop", response_model=SessionStatus)
def stop_session(session_id: str):
    data = _read_session(session_id)
    data["state"] = "stopped"
    data["notes"] = "session stopped"
    _write_session(data)
    return data


@app.post("/v1/sessions/{session_id}/reply")
def reply_session(session_id: str, payload: dict):
    data = _read_session(session_id)
    user_text = str(payload.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")

    if data.get("state") != "running":
        data["state"] = "running"
        _write_session(data)

    ollama_url = os.getenv("GPU_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = str(data.get("runtime_model") or data.get("hf_model") or "").strip()
    system = (
        f"Ты тестовый агент {data.get('agent_name')}. "
        "Отвечай по задаче пользователя кратко и по делу."
    )
    try:
        r = requests.post(
            ollama_url + "/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "options": {"temperature": 0.2, "top_p": 0.9, "num_ctx": 8192},
            },
            timeout=120,
        )
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("empty model response")
        data["notes"] = "reply served via ollama"
        _write_session(data)
        return {"session_id": session_id, "reply": content, "model": model, "state": data["state"]}
    except Exception as e:
        fallback = (
            f"[stub reply] Session {session_id} is ready, but model '{model}' could not be called: {e}"
        )
        return {"session_id": session_id, "reply": fallback, "model": model, "state": data["state"]}


@app.post("/v1/train-jobs", response_model=TrainJobStatus)
def create_train_job(payload: TrainJobCreate):
    job_id = uuidlib.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "job_id": job_id,
        "run_id": payload.run_id,
        "state": "queued",
        "created_at": now,
        "dataset_csv": payload.dataset_csv,
        "report_json": payload.report_json,
        "train_cmd": payload.train_cmd,
        "publish_cmd": payload.publish_cmd,
        "workdir": payload.workdir,
        "user_id": payload.user_id,
        "chat_id": payload.chat_id,
        "callback_url": payload.callback_url,
        "idle_timeout_sec": int(payload.idle_timeout_sec),
        "hf_link": None,
        "notes": "queued",
    }
    _write_job(data)
    return data


@app.get("/v1/train-jobs/{job_id}", response_model=TrainJobStatus)
def get_train_job(job_id: str):
    return _read_job(job_id)


@app.post("/v1/train-jobs/{job_id}/run", response_model=TrainJobStatus)
def run_train_job(job_id: str):
    load_dotenv()
    data = _read_job(job_id)
    ok, why = _gpu_ready()
    if not ok:
        data["state"] = "queued"
        data["notes"] = why
        _write_job(data)
        raise HTTPException(status_code=409, detail=why)

    workdir = Path(data.get("workdir") or ROOT).resolve()
    data["state"] = "running"
    data["notes"] = "training"
    _write_job(data)

    train_cmd = str(data["train_cmd"])
    publish_cmd = str(data["publish_cmd"])

    code, log = _run_subprocess(train_cmd, cwd=workdir)
    if code != 0:
        data["state"] = "failed"
        data["notes"] = log
        _write_job(data)
        raise HTTPException(status_code=500, detail=f"train failed: {log[-2000:]}")

    code, pub_log = _run_subprocess(publish_cmd, cwd=workdir)
    if code != 0:
        data["state"] = "failed"
        data["notes"] = pub_log
        _write_job(data)
        raise HTTPException(status_code=500, detail=f"publish failed: {pub_log[-2000:]}")

    hf_link = ""
    for line in reversed((pub_log or "").splitlines()):
        line = line.strip()
        if line.startswith("https://huggingface.co/"):
            hf_link = line
            break

    data["state"] = "done"
    data["notes"] = pub_log[-4000:]
    data["hf_link"] = hf_link or None
    _write_job(data)
    return data
