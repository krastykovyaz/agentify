#!/usr/bin/env python3
"""Collect real Telegram channel posts to CSV via Telethon.

Requires env vars (or CLI):
- TG_API_ID
- TG_API_HASH
- TG_SESSION (optional, default: tg_session)

Example:
python collect_telegram_posts.py \
  --channels meduzalive bbcrussian riafan_everywhere \
  --limit 1500 \
  --output datasets/telegram_real_posts.csv
"""

import argparse
import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path
from telethon import TelegramClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", nargs="+", required=True, help="@channel or username list")
    p.add_argument("--limit", type=int, default=1000, help="Messages per channel")
    p.add_argument("--output", required=True)
    p.add_argument("--min-chars", type=int, default=120)
    p.add_argument("--max-chars", type=int, default=2200)
    p.add_argument("--api-id", default=os.getenv("TG_API_ID"))
    p.add_argument("--api-hash", default=os.getenv("TG_API_HASH"))
    p.add_argument("--session", default=os.getenv("TG_SESSION", "tg_session"))
    return p.parse_args()


def looks_like_post(text: str) -> bool:
    t = text.lower()
    if text.count("\n") < 1:
        return False
    bad = ["подписаться", "реклама", "промокод", "донат", "скидка", "конкурс"]
    return not any(b in t for b in bad)


async def main_async(args):
    if not args.api_id or not args.api_hash:
        raise SystemExit("Set TG_API_ID and TG_API_HASH (or pass --api-id/--api-hash)")

    client = TelegramClient(args.session, int(args.api_id), args.api_hash)
    await client.start()

    rows = []
    for ch in args.channels:
        entity = await client.get_entity(ch)
        count = 0
        async for m in client.iter_messages(entity, limit=args.limit):
            if not m or not m.message:
                continue
            text = m.message.strip()
            if len(text) < args.min_chars or len(text) > args.max_chars:
                continue
            if not looks_like_post(text):
                continue
            rows.append({
                "channel": ch,
                "message_id": m.id,
                "date": m.date.isoformat() if m.date else "",
                "text": text,
            })
            count += 1
        print(f"{ch}: kept {count}")

    await client.disconnect()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "message_id", "date", "text"])
        w.writeheader()
        w.writerows(rows)

    print(f"saved: {out}")
    print(f"rows: {len(rows)}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
