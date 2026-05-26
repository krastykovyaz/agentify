#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from huggingface_hub import HfApi

AGENTS: List[Tuple[str, str, str, str]] = [
    # key, gguf filename, quant, short description
    ("summary", "models_gemma_summarization-Q3_K.gguf", "Q3_K", "Краткое резюме текста без вводных слов."),
    ("qa", "models_universal_gemma_v1-Q3_K.gguf", "Q3_K", "Фактические ответы по контексту."),
    ("extraction", "models_gemma_extraction_v1-Q3_K.gguf", "Q3_K", "Извлечение структурированных полей в JSON."),
    ("validator", "models_gemma_validator_v1-Q3_K.gguf", "Q3_K", "Проверка качества и рисков контента/ответов."),
    ("dialogue", "models_gemma_ak_dialog-Q4_K_M.gguf", "Q4_K_M", "Добрый и харизматичный диалоговый агент."),
    ("telegram", "models_gemma_telegram_post_v1-Q4_K_M.gguf", "Q4_K_M", "Один готовый пост для Telegram без вариантов."),
    ("universal", "models_universal_gemma_v1-Q4_K_M.gguf", "Q4_K_M", "Универсальный агент для mixed-задач."),
    ("coding_web", "models_gemma_coding_web_format_v2-Q4_K_M.gguf", "Q4_K_M", "Генерация веб-кода и скриптов под выполнение."),
]

SYSTEM_HINTS: Dict[str, str] = {
    "summary": "Краткое резюме ситуации. Сразу короткий текст по исходному тексту.",
    "qa": "Отвечай по фактам кратко и точно.",
    "extraction": "Верни только валидный JSON-объект строкой, без markdown и пояснений.",
    "validator": "Дай проверку качества и список замечаний по приоритету.",
    "dialogue": "Будь добрым и харизматичным собеседником.",
    "telegram": "Верни один финальный пост для Telegram без вариантов и саммари.",
    "universal": "Подстройся под задачу: summary/qa/extraction/telegram/coding.",
    "coding_web": "Верни чистый код/скрипт без markdown-пояснений.",
}


def make_model_card(repo_id: str, agent_key: str, quant: str, desc: str, gguf_name: str) -> str:
    hint = SYSTEM_HINTS.get(agent_key, "")
    return f"""---
language:
- ru
license: mit
tags:
- gguf
- gemma
- agentify
- {agent_key}
library_name: llama-cpp
pipeline_tag: text-generation
---

# {repo_id}

{desc}

## Quantization
- {quant}

## Files
- `{gguf_name}`

## Recommended system behavior
{hint}

## Usage (Ollama)
```bash
ollama create {agent_key} -f Modelfile
```

Where `Modelfile`:
```text
FROM ./{gguf_name}
```
"""


def upload_agents(api: HfApi, namespace: str, artifacts_dir: Path, private: bool, create_pr: bool) -> List[str]:
    created = []
    for key, gguf_name, quant, desc in AGENTS:
        gguf_path = artifacts_dir / gguf_name
        if not gguf_path.exists():
            print(f"skip {key}: missing {gguf_path}")
            continue

        repo_id = f"{namespace}/agentify-{key}-{quant.lower()}"
        print(f"\n=== upload {repo_id} ===")
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        card = make_model_card(repo_id, key, quant, desc, gguf_name)
        readme_tmp = artifacts_dir / f"README.{key}.md"
        readme_tmp.write_text(card, encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_name,
            repo_id=repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        api.upload_file(
            path_or_fileobj=str(readme_tmp),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            create_pr=create_pr,
        )
        created.append(repo_id)
        print(f"ok: {repo_id}")
    return created


def create_space(api: HfApi, namespace: str, private: bool, create_pr: bool):
    space_id = f"{namespace}/agentify-multi-agent-chat"
    print(f"\n=== setup space {space_id} ===")
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)

    here = Path(__file__).resolve().parent
    space_dir = here / "hf_space_multi_agent"
    if not space_dir.exists():
        raise SystemExit(f"missing {space_dir}")

    for name in ["app.py", "README.md", "requirements.txt"]:
        p = space_dir / name
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=name,
            repo_id=space_id,
            repo_type="space",
            create_pr=create_pr,
        )
    print(f"ok space: {space_id}")
    print("Set Space secret: OLLAMA_BASE_URL=https://<your-host>:11434")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Publish GGUF agents to Hugging Face and create chat Space")
    ap.add_argument("--namespace", required=True, help="HF user/org name")
    ap.add_argument("--artifacts-dir", default="/home/aleksandr.koviazin/kovyaz/agentify/artifacts")
    ap.add_argument("--private", action="store_true", default=False)
    ap.add_argument("--create-pr", action="store_true", default=False)
    ap.add_argument("--skip-space", action="store_true", default=False)
    args = ap.parse_args()

    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit("HF_TOKEN not found in env/.env")

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"artifacts dir not found: {artifacts_dir}")

    api = HfApi(token=token)
    who = api.whoami()
    print(f"HF auth ok: {who.get('name')}")

    repos = upload_agents(api, args.namespace, artifacts_dir, args.private, args.create_pr)
    print("\nUploaded repos:")
    for r in repos:
        print(f"- https://huggingface.co/{r}")

    if not args.skip_space:
        create_space(api, args.namespace, args.private, args.create_pr)


if __name__ == "__main__":
    main()
