#!/usr/bin/env python3
import argparse
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Test Gemma LoRA dialog model")
    p.add_argument("--base-model", default="google/gemma-4-E2B-it")
    p.add_argument("--lora-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=36)
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.12)
    p.add_argument(
        "--system",
        default=(
            "Отвечай как нейтральный эмпатичный собеседник. "
            "Только одна реплика ассистента, 1-2 предложения, "
            "не пиши за пользователя."
        ),
    )
    return p.parse_args()


def clean_reply(text: str) -> str:
    t = text.strip()
    markers = ["\nПользователь:", "\nАссистент:", "<|im_start|>user", "<|im_start|>assistant"]
    cuts = [t.find(m) for m in markers if t.find(m) != -1]
    if cuts:
        t = t[: min(cuts)]
    t = re.sub(r"^\s*(Ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def main():
    args = parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    tests = [
        "Пользователь: У меня тяжелый день, все валится из рук.\nАссистент:",
        "Пользователь: Привет, есть минутка поговорить?\nАссистент:",
        "Пользователь: Спасибо, ты мне очень помогла.\nАссистент:",
        "Пользователь: Я переживаю перед важной встречей.\nАссистент:",
    ]

    for i, t in enumerate(tests, 1):
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": t},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = clean_reply(gen)

        print(f"\n=== TEST {i} ===")
        print(t)
        print(f"Ответ: {gen}\n")


if __name__ == "__main__":
    main()
