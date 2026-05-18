#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path

TARGET_MODELS = [
    "gemma_summarization",
    "gemma_telegram_post_v1",
    "gemma_universal_v1",
    "gemma_extraction_v1",
    "gemma_ak_dialog",
]

QUANTS = ["Q2_K", "Q3_K", "Q4_K_M", "Q5_K_M", "Q6_K"]


def run(cmd: list[str], cwd: Path):
    p = subprocess.run(cmd, cwd=str(cwd), text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="Quantize selected gemma models from benchmark config")
    ap.add_argument("--config", required=True, help="benchmark_12_models.config.json")
    ap.add_argument("--root", required=True, help="project root, e.g. /home/aleksandr.koviazin/kovyaz/agentify")
    ap.add_argument("--base-model", default="google/gemma-4-E2B-it")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    root = Path(args.root)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    selected = []
    by_name = {m["name"]: m for m in cfg.get("models", [])}
    for name in TARGET_MODELS:
        m = by_name.get(name)
        if not m:
            print(f"skip: model '{name}' not found in config")
            continue
        lora_path = m.get("lora_path")
        if not lora_path:
            print(f"skip: model '{name}' has no lora_path")
            continue
        selected.append((name, lora_path))

    if not selected:
        raise SystemExit("No target models found with lora_path")

    rows = []
    for name, lora_path in selected:
        for q in QUANTS:
            print(f"\n==> {name} | {q}")
            run([
                "bash",
                str(root / "build_gemma_gguf.sh"),
                lora_path,
                q,
                args.base_model,
            ], cwd=root)

            gguf = root / "artifacts" / f"{Path(lora_path).name}-{q}.gguf"
            f16 = root / "artifacts" / f"{Path(lora_path).name}-f16.gguf"
            size = gguf.stat().st_size if gguf.exists() else 0
            rows.append({
                "model": name,
                "lora_path": lora_path,
                "quant": q,
                "gguf_path": str(gguf),
                "gguf_size_bytes": size,
                "f16_path": str(f16),
            })

    out_csv = Path(args.out_csv) if args.out_csv else (root / "artifacts" / "selected_gemma_quants.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "lora_path", "quant", "gguf_path", "gguf_size_bytes", "f16_path"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. Saved: {out_csv}")


if __name__ == "__main__":
    main()
