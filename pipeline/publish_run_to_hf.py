#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder


def _lora_base_model(outdir: Path) -> str:
    try:
        cfg = json.loads((outdir / "adapter_config.json").read_text(encoding="utf-8"))
        base = str(cfg.get("base_model_name_or_path") or "").strip()
        if base:
            return base
    except Exception:
        pass
    return os.getenv("GPU_BASE_MODEL", "google/gemma-4-E2B-it").strip() or "google/gemma-4-E2B-it"


def maybe_build_gguf(outdir: Path, root: Path) -> Path | None:
    if os.getenv("SKIP_GGUF_BUILD", "0").strip() in {"1", "true", "yes"}:
        return None
    if not (outdir / "adapter_config.json").exists():
        return None
    if not (
        (outdir / "adapter_model.safetensors").exists() or (outdir / "adapter_model.bin").exists()
    ):
        return None
    script = root / "build_gemma_gguf.sh"
    if not script.exists():
        print(f"skip gguf build: {script} not found")
        return None
    quant = os.getenv("GPU_GGUF_QUANT", "Q4_K_M").strip() or "Q4_K_M"
    base_model = _lora_base_model(outdir)
    print(f"Building GGUF from LoRA: quant={quant} base={base_model}")
    subprocess.run(
        ["bash", str(script), str(outdir.resolve()), quant, base_model],
        check=True,
        cwd=str(root),
    )
    gguf = root / "artifacts" / f"{outdir.name}-{quant}.gguf"
    if not gguf.exists():
        raise RuntimeError(f"gguf build failed: {gguf}")
    dest = outdir / gguf.name
    shutil.copy2(gguf, dest)
    print(f"GGUF ready: {dest}")
    return dest


def write_readme(outdir: Path, repo_id: str, run_id: str, dataset: str, report: str) -> None:
    readme = outdir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        f"""---
language:
- ru
license: mit
tags:
- agentify
- gemma
- lora
- sft
pipeline_tag: text-generation
---

# {repo_id}

Agentify fine-tuned model artifacts for run `{run_id}`.

## Training Inputs
- Dataset: `{dataset}`
- Report: `{report}`

## Notes
This repository contains the trained adapter/artifacts produced by the Telegram pipeline.
""",
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Publish one pipeline training run to Hugging Face")
    ap.add_argument("--outdir", required=True, help="Trained model output directory")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--namespace", default=os.getenv("HF_NAMESPACE", "Krasty"))
    ap.add_argument("--repo-prefix", default=os.getenv("HF_REPO_PREFIX", "agentify-pipeline"))
    ap.add_argument("--private", action="store_true", default=os.getenv("HF_PRIVATE", "0") == "1")
    args = ap.parse_args()

    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit("HF_TOKEN not found")

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise SystemExit(f"outdir not found: {outdir}")
    if not any(outdir.iterdir()):
        raise SystemExit(f"outdir is empty: {outdir}")

    safe_run = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in args.run_id)
    repo_id = f"{args.namespace}/{args.repo_prefix}-{safe_run}"

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    write_readme(outdir, repo_id, safe_run, args.dataset, args.report)

    root = Path(__file__).resolve().parent.parent
    maybe_build_gguf(outdir, root)

    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(outdir),
        token=token,
        commit_message=f"Upload pipeline run {safe_run}",
    )

    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
