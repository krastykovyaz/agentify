#!/usr/bin/env bash
set -euo pipefail

# Build quantized GGUF for Gemma LoRA adapters.
# Usage:
#   ./build_gemma_gguf.sh /path/to/lora_adapter Q4_K_M
#   ./build_gemma_gguf.sh /path/to/lora_adapter Q5_K_M google/gemma-4-E2B-it

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/artifacts"
LLAMA_DIR="${ROOT_DIR}/llama.cpp"

DEFAULT_BASE_MODEL="google/gemma-4-E2B-it"
DEFAULT_QUANT="Q4_K_M"

LORA_PATH="${1:-}"
QUANT_TYPE="${2:-$DEFAULT_QUANT}"
BASE_MODEL="${3:-$DEFAULT_BASE_MODEL}"

if [[ -z "${LORA_PATH}" ]]; then
  echo "Usage: $0 <lora_path> [quant_type] [base_model]" >&2
  exit 1
fi
if [[ ! -d "${LORA_PATH}" ]]; then
  echo "LoRA path not found: ${LORA_PATH}" >&2
  exit 1
fi

VALID_QUANTS=(Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0 Q4_K_M Q5_K_M Q2_K_M Q3_K_M Q4_K_S Q5_K_S)
valid=0
for q in "${VALID_QUANTS[@]}"; do
  if [[ "$q" == "$QUANT_TYPE" ]]; then valid=1; break; fi
done
if [[ $valid -eq 0 ]]; then
  echo "Unknown quant '${QUANT_TYPE}', fallback to ${DEFAULT_QUANT}" >&2
  QUANT_TYPE="${DEFAULT_QUANT}"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing dependency: $cmd" >&2
    exit 1
  fi
}

require_cmd git
require_cmd cmake
require_cmd python3

mkdir -p "${OUT_DIR}"

python3 -c "import torch, transformers, peft" >/dev/null 2>&1 || {
  echo "Need python deps: torch transformers peft" >&2
  exit 1
}

if [[ ! -d "${LLAMA_DIR}" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
fi

if [[ ! -f "${LLAMA_DIR}/build/bin/llama-quantize" ]]; then
  rm -rf "${LLAMA_DIR}/build"
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  cmake --build "${LLAMA_DIR}/build" -j
fi

ADAPTER_NAME="$(basename "${LORA_PATH}")"
TMP_MERGED="${OUT_DIR}/tmp_merged_${ADAPTER_NAME}"
F16_GGUF="${OUT_DIR}/${ADAPTER_NAME}-f16.gguf"
QUANT_GGUF="${OUT_DIR}/${ADAPTER_NAME}-${QUANT_TYPE}.gguf"

rm -rf "${TMP_MERGED}"

echo "Merging LoRA '${LORA_PATH}' into base '${BASE_MODEL}' ..."
python3 <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "${BASE_MODEL}"
lora_path = "${LORA_PATH}"
out_dir = "${TMP_MERGED}"

m = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
pm = PeftModel.from_pretrained(m, lora_path)
merged = pm.merge_and_unload()
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("merged saved", out_dir)
PY

[[ -f "${TMP_MERGED}/config.json" ]] || { echo "Merge failed" >&2; exit 1; }

echo "Converting to f16 gguf ..."
python3 "${LLAMA_DIR}/convert_hf_to_gguf.py" "${TMP_MERGED}" --outfile "${F16_GGUF}" --outtype f16
[[ -f "${F16_GGUF}" ]] || { echo "f16 conversion failed" >&2; exit 1; }

echo "Quantizing ${QUANT_TYPE} ..."
"${LLAMA_DIR}/build/bin/llama-quantize" "${F16_GGUF}" "${QUANT_GGUF}" "${QUANT_TYPE}"
[[ -f "${QUANT_GGUF}" ]] || { echo "Quantization failed" >&2; exit 1; }

rm -rf "${TMP_MERGED}"

echo "Built:"
echo "  ${F16_GGUF}"
echo "  ${QUANT_GGUF}"
du -h "${F16_GGUF}" "${QUANT_GGUF}" || true
