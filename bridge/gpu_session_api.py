#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import uuid
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import tempfile
import uuid as uuidlib
import re

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


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
    last_activity_at: str | None = None


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
MODEL_CACHE_DIR = Path(os.getenv("GPU_MODEL_CACHE_DIR", str(ROOT / "runs" / "hf_cache"))).resolve()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cleanup_stop = threading.Event()


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


def _job_artifacts_dir(job_id: str) -> Path:
    p = JOBS_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_artifacts_dir(session_id: str) -> Path:
    p = ROOT / "runs" / "session_artifacts" / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_model_dir(session_id: str) -> Path:
    p = _session_artifacts_dir(session_id) / "model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_session(data: dict) -> None:
    _session_path(data["session_id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_session(session_id: str) -> dict:
    p = _session_path(session_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_repo_id(hf_model: str) -> str:
    hf_model = (hf_model or "").strip()
    if hf_model.startswith("https://huggingface.co/"):
        hf_model = hf_model[len("https://huggingface.co/") :]
    hf_model = hf_model.strip("/")
    return hf_model


def _cleanup_session_artifacts(session_id: str) -> None:
    art = ROOT / "runs" / "session_artifacts" / session_id
    if art.exists():
        shutil.rmtree(art, ignore_errors=True)


def _cleanup_session_record(session_id: str) -> None:
    try:
        _session_path(session_id).unlink(missing_ok=True)
    except Exception:
        pass


def _cleanup_session_bundle(session_id: str) -> None:
    _cleanup_session_artifacts(session_id)
    _cleanup_session_record(session_id)


def _cleanup_previous_sessions(chat_id: int | None) -> None:
    if chat_id is None:
        return
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("chat_id") == chat_id:
            _cleanup_session_bundle(str(data.get("session_id") or p.stem))


def _touch_session(data: dict) -> None:
    data["last_activity_at"] = datetime.now(timezone.utc).isoformat()
    _write_session(data)


def _session_expired(data: dict) -> bool:
    last = str(data.get("last_activity_at") or data.get("created_at") or "").strip()
    if not last:
        return False
    try:
        dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except Exception:
        return False
    idle = int(data.get("idle_timeout_sec") or 900)
    return (datetime.now(timezone.utc) - dt).total_seconds() > idle


def _expire_session_if_needed(data: dict) -> bool:
    if not _session_expired(data):
        return False
    session_id = str(data.get("session_id") or "")
    if session_id:
        _cleanup_session_bundle(session_id)
    data["state"] = "stopped"
    data["notes"] = "session expired and artifacts cleaned"
    _write_session(data)
    return True


def _download_hf_model(repo_id_or_url: str, session_id: str) -> Path:
    repo_id = _parse_repo_id(repo_id_or_url)
    cache_dir = MODEL_CACHE_DIR / repo_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=["*.gguf", "*.json", "*.txt", "*.md"],
    )
    files = sorted(Path(path).rglob("*.gguf"))
    if not files:
        raise RuntimeError(f"no gguf found in {repo_id}")
    art = _session_model_dir(session_id)
    local = art / files[0].name
    shutil.copy2(files[0], local)
    return local


def _download_hf_artifacts(repo_id_or_url: str, session_id: str) -> dict:
    repo_id = _parse_repo_id(repo_id_or_url)
    cache_dir = MODEL_CACHE_DIR / repo_id.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = (os.getenv("HF_TOKEN") or "").strip() or None
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    model_files = sorted(Path(path).rglob("*.gguf"))
    if not model_files:
        raise RuntimeError(f"no gguf found in {repo_id}")
    session_dir = _session_model_dir(session_id)
    copied: list[str] = []
    for src in model_files:
        dst = session_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return {
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "session_dir": str(session_dir),
        "model_files": copied,
        "primary_model": copied[0] if copied else None,
    }


def _reply_via_local_gguf(model_path: Path, prompt: str) -> str:
    try:
        from llama_cpp import Llama  # type: ignore

        llm = Llama(model_path=str(model_path), n_ctx=8192, n_gpu_layers=-1, verbose=False)
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Отвечай кратко и по делу."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        return (out.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"[stub reply] local gguf model unavailable: {e}"


def _write_job(data: dict) -> None:
    _job_path(data["job_id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_job(job_id: str) -> dict:
    p = _job_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_subprocess(cmd: str, cwd: Path | None = None) -> tuple[int, str]:
    cmd = _normalize_cmd_paths(cmd)
    proc = subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd) if cwd else str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, (out or "")[-10000:]


def _normalize_cmd_paths(cmd: str) -> str:
    local_root = str(ROOT)
    remote_root = os.getenv("CPU_AGENTIFY_ROOT", "").strip()
    if remote_root:
        cmd = cmd.replace(remote_root, local_root)
    # replace common placeholders and stale absolute paths
    cmd = cmd.replace("{ROOT}", local_root)
    for stale in ["/home/alex/agentify", "/home/aleksandr.koviazin/kovyaz/agentify"]:
        cmd = cmd.replace(stale, local_root)
    return cmd


def _normalize_job_cmd(cmd: str, data: dict) -> str:
    cmd = _normalize_cmd_paths(cmd)
    source_dataset = str(data.get("source_dataset_csv") or "")
    source_report = str(data.get("source_report_json") or "")
    local_dataset = str(data.get("dataset_csv") or "")
    local_report = str(data.get("report_json") or "")
    local_outdir = str((JOBS_DIR / str(data.get("job_id") or "") / "model_out").resolve())
    if source_dataset and local_dataset:
        cmd = cmd.replace(source_dataset, local_dataset)
    if source_report and local_report:
        cmd = cmd.replace(source_report, local_report)
    cmd = cmd.replace("__GPU_DATASET__", local_dataset or source_dataset)
    cmd = cmd.replace("__GPU_REPORT__", local_report or source_report)
    cmd = cmd.replace("__GPU_OUTDIR__", local_outdir)
    return cmd


def _gpu_ready() -> tuple[bool, str]:
    ok, why = _can_launch()
    return ok, why


def _cleanup_expired_sessions_once() -> None:
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            if _session_expired(data):
                _cleanup_session_bundle(str(data.get("session_id") or p.stem))
                data["state"] = "stopped"
                data["notes"] = "session expired and artifacts cleaned"
                _write_session(data)
        except Exception:
            continue


def _cleanup_worker() -> None:
    while not _cleanup_stop.is_set():
        try:
            _cleanup_expired_sessions_once()
        except Exception:
            pass
        _cleanup_stop.wait(timeout=60)


@app.on_event("startup")
def _startup_cleanup_worker():
    if not getattr(app.state, "cleanup_worker_started", False):
        app.state.cleanup_worker_started = True
        t = threading.Thread(target=_cleanup_worker, daemon=True)
        t.start()


@app.get("/health")
def health():
    return {"ok": True, "service": "gpu-session-api"}


@app.post("/v1/sessions", response_model=SessionStatus)
def create_session(payload: SessionCreate):
    _cleanup_previous_sessions(payload.chat_id)
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
        "last_activity_at": now,
    }
    _write_session(data)
    return data


@app.get("/v1/sessions/{session_id}", response_model=SessionStatus)
def get_session(session_id: str):
    data = _read_session(session_id)
    if _expire_session_if_needed(data):
        return data
    return data


@app.post("/v1/sessions/{session_id}/launch", response_model=SessionStatus)
def launch_session(session_id: str):
    data = _read_session(session_id)
    if _expire_session_if_needed(data):
        raise HTTPException(status_code=409, detail="session expired")
    ok, why = _can_launch()
    if not ok:
        data["state"] = "queued"
        data["notes"] = why
        _write_session(data)
        raise HTTPException(status_code=409, detail=why)
    repo_or_url = str(data.get("hf_model") or "").strip()
    if repo_or_url.startswith("https://huggingface.co/") or "/" in repo_or_url:
        try:
            hf_info = _download_hf_artifacts(repo_or_url, session_id)
            data["runtime_model"] = str(hf_info["primary_model"])
            data["notes"] = f"downloaded {len(hf_info['model_files'])} model file(s)"
        except Exception as e:
            data["notes"] = f"hf download failed: {e}"
            _write_session(data)
            raise HTTPException(status_code=500, detail=f"hf download failed: {e}")
    data["state"] = "running"
    data["runtime_url"] = data.get("runtime_url") or f"local://{session_id}"
    data["last_activity_at"] = datetime.now(timezone.utc).isoformat()
    _write_session(data)
    return data


@app.post("/v1/sessions/{session_id}/stop", response_model=SessionStatus)
def stop_session(session_id: str):
    data = _read_session(session_id)
    data["state"] = "stopped"
    data["notes"] = "session stopped"
    _cleanup_session_bundle(session_id)
    _write_session(data)
    return data


@app.post("/v1/sessions/{session_id}/reply")
def reply_session(session_id: str, payload: dict):
    data = _read_session(session_id)
    user_text = str(payload.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="text is required")
    if _expire_session_if_needed(data):
        raise HTTPException(status_code=409, detail="session expired")

    if data.get("state") != "running":
        data["state"] = "running"
        _touch_session(data)

    model = str(data.get("runtime_model") or data.get("hf_model") or "").strip()
    try:
        if model.endswith(".gguf") and Path(model).exists():
            content = _reply_via_local_gguf(Path(model), user_text)
        else:
            ollama_url = os.getenv("GPU_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            system = (
                f"Ты тестовый агент {data.get('agent_name')}. "
                "Отвечай по задаче пользователя кратко и по делу."
            )
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
        _touch_session(data)
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
        "source_dataset_csv": payload.dataset_csv,
        "source_report_json": payload.report_json,
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


@app.post("/v1/train-jobs/{job_id}/artifacts", response_model=TrainJobStatus)
def upload_train_job_artifacts(
    job_id: str,
    dataset_csv: UploadFile | None = File(default=None),
    report_json: UploadFile | None = File(default=None),
):
    data = _read_job(job_id)
    art_dir = _job_artifacts_dir(job_id)
    if dataset_csv is None and report_json is None:
        raise HTTPException(status_code=400, detail="no artifacts uploaded")
    if dataset_csv is not None:
        (art_dir / "pipeline_train_1000.csv").write_bytes(dataset_csv.file.read())
        data["dataset_csv"] = str(art_dir / "pipeline_train_1000.csv")
    if report_json is not None:
        (art_dir / "pipeline_train_1000.report.json").write_bytes(report_json.file.read())
        data["report_json"] = str(art_dir / "pipeline_train_1000.report.json")
    data["notes"] = "artifacts uploaded"
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

    workdir_raw = str(data.get("workdir") or "").strip()
    workdir = Path(workdir_raw).resolve() if workdir_raw else ROOT
    if not workdir.exists():
        workdir = ROOT
    outdir = Path(str((JOBS_DIR / str(data.get("job_id") or "") / "model_out"))).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    data["state"] = "running"
    data["notes"] = "training"
    _write_job(data)

    train_cmd = str(data["train_cmd"])
    publish_cmd = str(data["publish_cmd"])
    train_cmd = _normalize_job_cmd(train_cmd, data)
    publish_cmd = _normalize_job_cmd(publish_cmd, data)
    train_cmd = train_cmd.replace("__GPU_OUTDIR__", str(outdir))
    publish_cmd = publish_cmd.replace("__GPU_OUTDIR__", str(outdir))

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
