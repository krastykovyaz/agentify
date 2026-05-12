#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    # remove most emoji/symbol noise to stabilize matching
    name = re.sub(r"[^\w\sа-яА-ЯёЁ-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_text(msg: Dict[str, Any]) -> str:
    text = msg.get("text", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        parts: List[str] = []
        for chunk in text:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                parts.append(str(chunk.get("text", "")))
        return "".join(parts).strip()
    return ""


def load_messages(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    return [m for m in messages if m.get("type") == "message"]


def build_rows(
    messages: List[Dict[str, Any]],
    assistant_name: str,
    user_name: str,
    context_turns: int,
    system_prompt: str,
) -> List[Dict[str, str]]:
    assistant_norm = normalize_name(assistant_name)
    user_norm = normalize_name(user_name)

    timeline: List[Dict[str, str]] = []
    rows: List[Dict[str, str]] = []

    for m in messages:
        speaker = str(m.get("from", "")).strip()
        speaker_norm = normalize_name(speaker)
        text = extract_text(m)
        if not text:
            continue

        # Keep only two target speakers to avoid noisy examples
        if speaker_norm not in {assistant_norm, user_norm}:
            continue

        timeline.append({"speaker": speaker, "speaker_norm": speaker_norm, "text": text})

        if speaker_norm != assistant_norm:
            continue

        # find nearest previous user message index
        prev_user_idx = None
        for i in range(len(timeline) - 2, -1, -1):
            if timeline[i]["speaker_norm"] == user_norm:
                prev_user_idx = i
                break
        if prev_user_idx is None:
            continue

        # build context ending at the previous user message
        start_idx = max(0, prev_user_idx - context_turns)
        context = timeline[start_idx : prev_user_idx + 1]
        raw_text = "\n".join([f"{item['speaker']}: {item['text']}" for item in context]).strip()
        ready_text = text

        if raw_text and ready_text:
            rows.append(
                {
                    "raw_text": raw_text,
                    "ready_text": ready_text,
                    "system": system_prompt,
                }
            )

    return rows


def save_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_text", "ready_text", "system"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT dialogue CSV from Telegram-like result.json"
    )
    parser.add_argument(
        "--input",
        default="datasets/chats/result.json",
        help="Path to source JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/chats",
        help="Directory to store generated CSV files",
    )
    parser.add_argument(
        "--ira-name",
        default="Иришка",
        help="Display name fragment for Ira in source chat",
    )
    parser.add_argument(
        "--alex-name",
        default="Kovyaz",
        help="Display name fragment for Alexander in source chat",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=6,
        help="How many prior turns to include before user message",
    )
    parser.add_argument(
        "--system",
        default="Отвечай естественно, кратко и по делу, сохраняя стиль диалога.",
        help="System prompt for each training row",
    )

    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    messages = load_messages(src)

    # Dataset 1: assistant is Ira, user is Alexander
    rows_ira = build_rows(
        messages=messages,
        assistant_name=args.ira_name,
        user_name=args.alex_name,
        context_turns=args.context_turns,
        system_prompt=args.system,
    )
    save_csv(rows_ira, out_dir / "dialog_sft_ira_assistant.csv")

    # Dataset 2: assistant is Alexander, user is Ira
    rows_alex = build_rows(
        messages=messages,
        assistant_name=args.alex_name,
        user_name=args.ira_name,
        context_turns=args.context_turns,
        system_prompt=args.system,
    )
    save_csv(rows_alex, out_dir / "dialog_sft_alex_assistant.csv")

    print("Done.")
    print(f"Ira assistant rows: {len(rows_ira)} -> {out_dir / 'dialog_sft_ira_assistant.csv'}")
    print(f"Alexander assistant rows: {len(rows_alex)} -> {out_dir / 'dialog_sft_alex_assistant.csv'}")


if __name__ == "__main__":
    main()
