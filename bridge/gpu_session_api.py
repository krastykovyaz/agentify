#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    agent_name: str = Field(..., min_length=1)
    hf_model: str = Field(..., min_length=1)
    user_id: int | None = None
    chat_id: int | None = None
    idle_timeout_sec: int = Field(default=900, ge=60)
    callback_url: str | None = None


class SessionStatus(BaseModel):
    session_id: str
    agent_name: str
    hf_model: str
    state: str
    created_at: str
    idle_timeout_sec: int
    user_id: int | None = None
    chat_id: int | None = None
    callback_url: str | None = None
    runtime_url: str | None = None
    notes: str | None = None


app = FastAPI(title="Agentify GPU Session API", version="0.1.0")
ROOT = Path(os.getenv("AGENTIFY_ROOT", Path(__file__).resolve().parent.parent)).resolve()
SESSIONS_DIR = Path(os.getenv("GPU_SESSION_DIR", str(ROOT / "runs" / "sessions"))).resolve()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _write_session(data: dict) -> None:
    _session_path(data["session_id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(p.read_text(encoding="utf-8"))


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

