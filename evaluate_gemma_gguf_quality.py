#!/usr/bin/env python3
import argparse
import json
import subprocess
from difflib import SequenceMatcher

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SAMPLES = {
    "summarization": [
        "Сегодня в регионе ожидаются сильный ветер и осадки, МЧС рекомендует ограничить поездки за город."
    ],
    "telegram_post": [
        "В субботу в 12:00 в центре города пройдет ярмарка вакансий, вход свободный, участвуют более 40 компаний."
    ],
    "qa": [
        "Вопрос: Что за событие будет в субботу? Контекст: В субботу в 12:00 пройдет ярмарка вакансий в центре города."
    ],
    "extraction": [
        "Продам велосипед Trek, 25000 руб, Ижевск, звонить +79001234567."
    ],
}

INSTR = {
    "summarization": "Сделай краткую фактическую выжимку текста.",
    "telegram_post": "Сделай пост для Telegram: заголовок и 1-3 абзаца.",
    "qa": "Ответь по фактам кратко.",
    "extraction": "Извлеки JSON с полями title, category, location, date, price, contacts, summary.",
}


def hf_answer(base_model, lora_path, task):
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

    text = SAMPLES[task][0]
    messages = [
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": f"[{task}] {INSTR[task]}\n\n{text}"},
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
    ans = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return ans


def gguf_answer(gguf_path, task, n_gpu_layers=0):
    text = SAMPLES[task][0]
    prompt = (
        f"Задача: {task}\n"
        f"Инструкция: {INSTR[task]}\n\n"
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
    if n_gpu_layers and n_gpu_layers > 0:
        cmd.extend(["-ngl", str(n_gpu_layers)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "") + "\n" + (r.stdout or ""))
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["summarization", "telegram_post", "qa", "extraction"])
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--gguf-path", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    ref = hf_answer(args.base_model, args.lora_path, args.task)
    gg = gguf_answer(args.gguf_path, args.task, n_gpu_layers=args.n_gpu_layers)

    score = SequenceMatcher(None, ref, gg).ratio()
    payload = {"task": args.task, "score": score, "reference": ref, "gguf": gg}

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{score:.4f}")


if __name__ == "__main__":
    main()
