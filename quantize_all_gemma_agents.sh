#!/usr/bin/env bash
set -euo pipefail

# Batch quantization + quality eval for Gemma agent set.
# Usage:
#   ./quantize_all_gemma_agents.sh /home/aleksandr.koviazin/kovyaz/agentify

ROOT="${1:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENTS=(
  "gemma_summarization:${ROOT}/models_gemma_summarization:summarization"
  "gemma_telegram_post_v1:${ROOT}/models_gemma_telegram_post_v1:telegram_post"
  "gemma_universal_v1:${ROOT}/models_universal_gemma_v1:qa"
  "gemma_extraction_v1:${ROOT}/models_gemma_extraction_v1:extraction"
  "gemma_ak_dialog:${ROOT}/models_gemma_ak_dialog:qa"
)

QUANTS=(Q2_K Q3_K Q4_K_M Q5_K_M Q6_K)

RESULT_CSV="${ROOT}/artifacts/gemma_quant_quality.csv"
mkdir -p "${ROOT}/artifacts"

echo "model,task,quant,gguf_path,size_bytes,quality_score" > "${RESULT_CSV}"

for a in "${AGENTS[@]}"; do
  IFS=':' read -r name lora task <<< "$a"
  if [[ ! -d "$lora" ]]; then
    echo "skip missing $lora"
    continue
  fi

  for q in "${QUANTS[@]}"; do
    "${SCRIPT_DIR}/build_gemma_gguf.sh" "$lora" "$q"
    gguf="${ROOT}/artifacts/$(basename "$lora")-${q}.gguf"
    if [[ ! -f "$gguf" ]]; then
      echo "missing gguf $gguf" >&2
      continue
    fi

    score=$(python3 "${SCRIPT_DIR}/evaluate_gemma_gguf_quality.py" \
      --task "$task" \
      --base-model "google/gemma-4-E2B-it" \
      --lora-path "$lora" \
      --gguf-path "$gguf" \
      --json-out /tmp/eval_${name}_${q}.json | tail -n1)

    size=$(stat -c%s "$gguf" 2>/dev/null || stat -f%z "$gguf")
    echo "${name},${task},${q},${gguf},${size},${score}" >> "${RESULT_CSV}"
  done

done

echo "Saved: ${RESULT_CSV}"
