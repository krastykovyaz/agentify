#!/usr/bin/env python3
"""
Run two LoRA personas in a dialogue loop on top of one base model.
Optionally mirror messages to Telegram via bot API.

Example:
python3 agent_duo_chat.py \
  --agent1-path /home/aleksandr.koviazin/kovyaz/agentify/models_ak \
  --agent2-path /home/aleksandr.koviazin/kovyaz/agentify/models_ir \
  --turns 20 \
  --seed-text "Привет, как проходит день?"

Telegram (optional):
python3 agent_duo_chat.py ... \
  --telegram-token "$TG_BOT_TOKEN" \
  --telegram-chat-id "123456789"
"""

import argparse
import os
import re
import time
from typing import Optional

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dialogue between two LoRA agents")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--agent1-path", required=True, help="Path to LoRA adapter #1")
    p.add_argument("--agent2-path", required=True, help="Path to LoRA adapter #2")
    p.add_argument("--agent1-name", default="Александр")
    p.add_argument("--agent2-name", default="Ира")
    p.add_argument("--turns", type=int, default=16)
    p.add_argument("--context-turns", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.65)
    p.add_argument("--top-p", type=float, default=0.85)
    p.add_argument("--repetition-penalty", type=float, default=1.12)
    p.add_argument("--seed-text", default="Привет. Как ты сегодня?")
    p.add_argument(
        "--system",
        default=(
            "Ты эмпатичный собеседник. Отвечай только одной репликой ассистента, "
            "кратко и по делу. Не пиши за пользователя и не продолжай обе роли."
        ),
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument("--telegram-token", default=os.getenv("TG_BOT_TOKEN"))
    p.add_argument("--telegram-chat-id", default=os.getenv("TG_CHAT_ID"))
    p.add_argument("--telegram-prefix", default="[duo-chat]")
    p.add_argument("--sleep-sec", type=float, default=0.0)
    return p.parse_args()


def tg_send(token: Optional[str], chat_id: Optional[str], text: str) -> None:
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=20)
    except Exception as e:
        print(f"[telegram] send failed: {e}")


def clean_reply(text: str) -> str:
    # Keep first assistant reply only.
    t = text.strip()
    markers = ["\nПользователь:", "\nАссистент:", "<|im_start|>user", "<|im_start|>assistant"]
    cut = [t.find(m) for m in markers if t.find(m) != -1]
    if cut:
        t = t[: min(cut)]
    t = re.sub(r"^\s*(Ассистент|assistant)\s*:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def build_prompt(system: str, dialog_lines: list[str], speaker_name: str) -> str:
    history = "\n".join(dialog_lines)
    user_text = f"{history}\n{speaker_name}:"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_once(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_reply(gen)


def main() -> None:
    args = parse_args()

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    print("Loading adapter #1...")
    model = PeftModel.from_pretrained(base_model, args.agent1_path, adapter_name="agent1")
    print("Loading adapter #2...")
    model.load_adapter(args.agent2_path, adapter_name="agent2")
    model.eval()

    dialog: list[str] = [f"{args.agent1_name}: {args.seed_text}"]

    start_msg = (
        f"{args.telegram_prefix} start\n"
        f"agent1={args.agent1_name}\n"
        f"agent2={args.agent2_name}\n"
        f"turns={args.turns}"
    )
    tg_send(args.telegram_token, args.telegram_chat_id, start_msg)

    print("\n=== Dialogue ===")
    print(dialog[0])
    tg_send(args.telegram_token, args.telegram_chat_id, dialog[0])

    for turn in range(args.turns):
        speaker_is_1 = (turn % 2 == 1)
        adapter_name = "agent1" if speaker_is_1 else "agent2"
        speaker_name = args.agent1_name if speaker_is_1 else args.agent2_name

        model.set_adapter(adapter_name)

        ctx = dialog[-args.context_turns:]
        prompt = build_prompt(args.system, ctx, speaker_name)
        reply = generate_once(model, tokenizer, prompt, args)
        if not reply:
            reply = "..."

        line = f"{speaker_name}: {reply}"
        dialog.append(line)
        print(line)
        tg_send(args.telegram_token, args.telegram_chat_id, line)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    tg_send(args.telegram_token, args.telegram_chat_id, f"{args.telegram_prefix} done")


if __name__ == "__main__":
    main()
