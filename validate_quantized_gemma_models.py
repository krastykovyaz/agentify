#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TASK_SAMPLES = {
    "summarization": [
        "Сегодня в Ижевске ожидается сильный ветер, местами дождь и понижение температуры до +8 ночью. МЧС рекомендует ограничить поездки за город.",
        "На участке трассы М-7 временно ограничено движение из-за ремонта. Работы продлятся до конца недели, организован объезд.",
    ],
    "telegram_post": [
        "В субботу в центре города пройдет ярмарка вакансий. Участвуют более 40 компаний, вход свободный, начало в 12:00.",
        "В парке Горького в воскресенье состоится семейный фестиваль, начало в 11:00, вход свободный.",
    ],
    "qa": [
        "Вопрос: Что будет в субботу? Контекст: В субботу в центре города пройдет ярмарка вакансий, начало в 12:00.",
        "Вопрос: До какого дня ремонт на трассе? Контекст: Ограничение на участке М-7 продлится до конца недели.",
    ],
    "extraction": [
        "Продам велосипед Trek, 25000 руб, Ижевск, звонить +79001234567.",
        "Сдам 1-комнатную квартиру, 18000 руб/мес, Мытищи, писать @owner_home.",
    ],
}

TASK_INSTR = {
    "summarization": "Сделай краткую фактическую выжимку текста.",
    "telegram_post": "Сделай пост для Telegram: заголовок и 1-3 абзаца, без выдумки фактов.",
    "qa": "Ответь кратко и по фактам.",
    "extraction": "Извлеки JSON с полями title, category, location, date, price, contacts, summary.",
}


@dataclass
class ModelCfg:
    name: str
    family: str
    base_model: str
    lora_path: str | None
    tasks: List[str]
    system: str


def load_benchmark_cfg(path: Path) -> Dict[str, ModelCfg]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for x in data.get("models", []):
        out[x["name"]] = ModelCfg(
            name=x["name"],
            family=x.get("family", "gemma"),
            base_model=x["base_model"],
            lora_path=x.get("lora_path"),
            tasks=x.get("tasks", ["qa"]),
            system=x.get("system", "Ты полезный ассистент."),
        )
    return out


def clean_output(text: str) -> str:
    t = text.strip()
    for m in ["\nПользователь:", "\nАссистент:", "<|im_start|>user", "<|im_start|>assistant"]:
        if m in t:
            t = t.split(m, 1)[0]
    return t.strip()


def hf_answer(base_model: str, lora_path: str, system: str, task: str, text: str) -> str:
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"[{task}] {TASK_INSTR[task]}\n\n{text}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=180,
            temperature=0.4,
            top_p=0.85,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_output(ans)


def gguf_answer(gguf_path: str, task: str, text: str, n_gpu_layers: int) -> str:
    prompt = (
        f"Задача: {task}\n"
        f"Инструкция: {TASK_INSTR[task]}\n\n"
        f"Вход:\n{text}\n\n"
        "Ответ:\n"
    )
    cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", gguf_path,
        "-p", prompt,
        "-n", "180",
        "--temp", "0.4",
        "--top-p", "0.85",
        "--repeat-penalty", "1.1",
        "--no-display-prompt",
    ]
    if n_gpu_layers > 0:
        cmd.extend(["-ngl", str(n_gpu_layers)])

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "") + "\n" + (r.stdout or ""))
    return clean_output(r.stdout)


def task_valid(task: str, out: str) -> bool:
    out = out.strip()
    if not out:
        return False
    if task == "summarization":
        return len(out) >= 40 and len(out) <= 500 and not out.startswith("{")
    if task == "telegram_post":
        return len(out) >= 80 and ("\n" in out)
    if task == "qa":
        return len(out) >= 8 and len(out) <= 400
    if task == "extraction":
        return out.startswith("{") and ("summary" in out or "contacts" in out)
    return True


def main():
    ap = argparse.ArgumentParser(description="Validate quantized Gemma models against LoRA references")
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--quants-csv", required=True, help="selected_gemma_quants.csv")
    ap.add_argument("--out-detailed-csv", required=True)
    ap.add_argument("--out-summary-csv", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--cwd", default=".", help="Run cwd for llama-cli path resolution")
    args = ap.parse_args()

    cfg = load_benchmark_cfg(Path(args.benchmark_config))

    quant_rows = []
    with Path(args.quants_csv).open("r", encoding="utf-8-sig", newline="") as f:
        quant_rows = list(csv.DictReader(f))

    reference_cache: Dict[Tuple[str, str, str], str] = {}
    detailed = []

    # ensure llama-cli relative path works
    old_cwd = Path.cwd()
    run_cwd = Path(args.cwd)

    for qrow in quant_rows:
        model_name = qrow["model"]
        gguf_path = qrow["gguf_path"]
        quant = qrow["quant"]
        size_bytes = int(qrow.get("gguf_size_bytes") or 0)

        if model_name not in cfg:
            print(f"skip unknown model in config: {model_name}")
            continue
        mcfg = cfg[model_name]
        if not mcfg.lora_path:
            print(f"skip no lora_path for model: {model_name}")
            continue

        for task in mcfg.tasks:
            for sample in TASK_SAMPLES.get(task, []):
                rkey = (mcfg.lora_path, task, sample)
                if rkey not in reference_cache:
                    reference_cache[rkey] = hf_answer(
                        base_model=mcfg.base_model,
                        lora_path=mcfg.lora_path,
                        system=mcfg.system,
                        task=task,
                        text=sample,
                    )
                ref = reference_cache[rkey]

                try:
                    # run gguf from requested cwd so ./llama.cpp path stays valid
                    import os
                    os.chdir(run_cwd)
                    gg = gguf_answer(gguf_path=gguf_path, task=task, text=sample, n_gpu_layers=args.n_gpu_layers)
                finally:
                    os.chdir(old_cwd)

                sim = SequenceMatcher(None, ref, gg).ratio()
                valid = task_valid(task, gg)
                detailed.append(
                    {
                        "model": model_name,
                        "task": task,
                        "quant": quant,
                        "gguf_path": gguf_path,
                        "size_bytes": size_bytes,
                        "sample": sample,
                        "reference": ref,
                        "gguf_output": gg,
                        "similarity": f"{sim:.4f}",
                        "task_valid": int(valid),
                    }
                )

    # write detailed
    dpath = Path(args.out_detailed_csv)
    dpath.parent.mkdir(parents=True, exist_ok=True)
    with dpath.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task",
                "quant",
                "gguf_path",
                "size_bytes",
                "sample",
                "reference",
                "gguf_output",
                "similarity",
                "task_valid",
            ],
        )
        w.writeheader()
        w.writerows(detailed)

    # summary by model+quant
    group = {}
    for r in detailed:
        key = (r["model"], r["quant"], r["gguf_path"], r["size_bytes"])
        group.setdefault(key, {"sims": [], "valid": []})
        group[key]["sims"].append(float(r["similarity"]))
        group[key]["valid"].append(int(r["task_valid"]))

    summary_rows = []
    for (model, quant, gguf_path, size_bytes), vals in group.items():
        avg_sim = statistics.mean(vals["sims"]) if vals["sims"] else 0.0
        valid_rate = statistics.mean(vals["valid"]) if vals["valid"] else 0.0
        summary_rows.append(
            {
                "model": model,
                "quant": quant,
                "gguf_path": gguf_path,
                "size_bytes": size_bytes,
                "size_gb": f"{int(size_bytes)/(1024**3):.3f}",
                "avg_similarity": f"{avg_sim:.4f}",
                "valid_rate": f"{valid_rate:.4f}",
                "quality_size_score": f"{(avg_sim * 0.7 + valid_rate * 0.3):.4f}",
            }
        )

    # better first: higher quality score, then smaller size
    summary_rows.sort(key=lambda x: (-float(x["quality_size_score"]), int(x["size_bytes"])))

    spath = Path(args.out_summary_csv)
    spath.parent.mkdir(parents=True, exist_ok=True)
    with spath.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "quant",
                "gguf_path",
                "size_bytes",
                "size_gb",
                "avg_similarity",
                "valid_rate",
                "quality_size_score",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"saved detailed: {dpath}")
    print(f"saved summary: {spath}")


if __name__ == "__main__":
    main()
