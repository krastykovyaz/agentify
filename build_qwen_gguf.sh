#!/usr/bin/env bash
set -euo pipefail

# Build quantized GGUF from Qwen LoRA model (without unsloth).
# Usage:
#   ./build_qwen_gguf.sh /path/to/lora/adapter Q4_K_M
#   ./build_qwen_gguf.sh ./qwen35_4b_finetuned Q2_K_M

# Default values
DEFAULT_MODEL_PATH="./qwen35_4b_finetuned_no_reasoning"
DEFAULT_QUANT_TYPE="Q4_K_M"

# Get parameters from command line
LORA_PATH="${1:-$DEFAULT_MODEL_PATH}"
QUANT_TYPE="${2:-$DEFAULT_QUANT_TYPE}"
BASE_MODEL="Qwen/Qwen3.5-4B"

# Validate input
if [[ ! -d "${LORA_PATH}" ]]; then
    echo "❌ LoRA path not found: ${LORA_PATH}" >&2
    echo "Usage: $0 [path_to_lora] [quantization_type]" >&2
    echo "Example: $0 ./qwen35_4b_finetuned Q2_K_M" >&2
    exit 1
fi

# Validate quantization type
case "${QUANT_TYPE}" in
    Q2_K|Q3_K|Q4_K|Q5_K|Q6_K|Q8_0|Q4_K_M|Q5_K_M|Q2_K_M|Q3_K_M|Q4_K_S|Q5_K_S)
        echo "✅ Quantization type: ${QUANT_TYPE}"
        ;;
    *)
        echo "⚠️  Unknown quantization type: ${QUANT_TYPE}" >&2
        echo "Available types: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_K_M, Q5_K_M, Q2_K_M, Q3_K_M, Q4_K_S, Q5_K_S" >&2
        echo "Using default: ${DEFAULT_QUANT_TYPE}" >&2
        QUANT_TYPE="${DEFAULT_QUANT_TYPE}"
        ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/artifacts"
LLAMA_DIR="${ROOT_DIR}/llama.cpp"
TEMP_MERGED="${OUT_DIR}/temp_merged"
F16_GGUF="${OUT_DIR}/qwen-f16.gguf"
QUANT_GGUF="${OUT_DIR}/qwen-${QUANT_TYPE}.gguf"

mkdir -p "${OUT_DIR}"

# Check if torch is available (activate venv first)
python3 -c "import torch" 2>/dev/null || {
  echo "❌ torch not found. Please activate your venv first" >&2
  echo "Run: source /path/to/venv/bin/activate" >&2
  exit 1
}

# Check if transformers are available
python3 -c "from transformers import AutoModelForCausalLM" 2>/dev/null || {
  echo "❌ transformers not found. Please install: pip install transformers peft" >&2
  exit 1
}

# Build llama.cpp if needed
if [[ ! -d "${LLAMA_DIR}" ]]; then
  echo "📦 Cloning llama.cpp..."
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
fi

BUILD_ARGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  BUILD_ARGS+=(-DGGML_METAL=ON)
fi

if [[ ! -f "${LLAMA_DIR}/build/bin/llama-quantize" ]]; then
  echo "🔨 Building llama.cpp..."
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" "${BUILD_ARGS[@]}"
  cmake --build "${LLAMA_DIR}/build" -j
fi

# Step 1: Merge LoRA with base model
echo "🔄 Merging LoRA from ${LORA_PATH} with base model ${BASE_MODEL}..."

python3 << PYTHON_SCRIPT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys

lora_path = "${LORA_PATH}"
temp_merged = "${TEMP_MERGED}"
base_model_name = "${BASE_MODEL}"

print(f"Loading base model: {base_model_name}")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
except Exception as e:
    print(f"❌ Failed to load base model: {e}")
    sys.exit(1)

print(f"Loading LoRA adapter from: {lora_path}")
try:
    model = PeftModel.from_pretrained(base_model, lora_path)
except Exception as e:
    print(f"❌ Failed to load LoRA adapter: {e}")
    sys.exit(1)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {temp_merged}")
merged_model.save_pretrained(temp_merged)
tokenizer.save_pretrained(temp_merged)

print("✅ Merge completed!")
PYTHON_SCRIPT

# Check if merge was successful
if [[ ! -f "${TEMP_MERGED}/config.json" ]]; then
    echo "❌ Merge failed: ${TEMP_MERGED}/config.json not found" >&2
    exit 1
fi

# Step 2: Convert merged model to FP16 GGUF
echo "🔄 Converting to FP16 GGUF..."
python3 "${LLAMA_DIR}/convert_hf_to_gguf.py" \
  "${TEMP_MERGED}" \
  --outfile "${F16_GGUF}" \
  --outtype f16

if [[ ! -f "${F16_GGUF}" ]]; then
    echo "❌ Conversion to FP16 GGUF failed" >&2
    exit 1
fi

# Step 3: Quantize to target format
echo "🔄 Quantizing to ${QUANT_TYPE}..."
"${LLAMA_DIR}/build/bin/llama-quantize" "${F16_GGUF}" "${QUANT_GGUF}" "${QUANT_TYPE}"

if [[ ! -f "${QUANT_GGUF}" ]]; then
    echo "❌ Quantization failed" >&2
    exit 1
fi

# Cleanup
echo "🧹 Cleaning up temporary files..."
rm -rf "${TEMP_MERGED}"

echo ""
echo "✅ Build completed successfully!"
echo "📁 Output files:"
echo "   - FP16: ${F16_GGUF}"
echo "   - Quantized (${QUANT_TYPE}): ${QUANT_GGUF}"
echo ""
echo "📊 File sizes:"
du -h "${F16_GGUF}" "${QUANT_GGUF}" 2>/dev/null || true