#!/usr/bin/env bash
set -euo pipefail

# Batch build GGUF + register Ollama models for selected agents.
#
# Usage:
#   ./batch_build_and_register_ollama.sh
#   AGENTIFY_ROOT=/home/aleksandr.koviazin/kovyaz/agentify ./batch_build_and_register_ollama.sh
#
# Optional env overrides:
#   AGENTIFY_ROOT=...           project root
#   BASE_MODEL=google/gemma-4-E2B-it
#   OLLAMA_NS=agentify          model name prefix in Ollama
#   SKIP_BUILD=1                only run ollama create for existing gguf
#   SKIP_OLLAMA=1               only build gguf files

ROOT_DIR="${AGENTIFY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
BUILD_SCRIPT="${ROOT_DIR}/build_gemma_gguf.sh"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-E2B-it}"
OLLAMA_NS="${OLLAMA_NS:-agentify}"

if [[ ! -x "${BUILD_SCRIPT}" ]]; then
  echo "Missing executable build script: ${BUILD_SCRIPT}" >&2
  exit 1
fi

if [[ "${SKIP_OLLAMA:-0}" != "1" ]] && ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found in PATH. Install ollama or run with SKIP_OLLAMA=1" >&2
  exit 1
fi

mkdir -p "${ARTIFACTS_DIR}" "${ARTIFACTS_DIR}/modelfiles"

# Quant mapping (requested):
# - Q3 agents -> Q3_K
# - Q4 agents -> Q4_K_M

# Agent label | LoRA dir name | quant
# NOTE: validator path is a placeholder; override if needed.
AGENTS=$(cat <<'EOF'
summary|models_gemma_summarization|Q3_K
qa|models_universal_gemma_v1|Q3_K
extraction|models_gemma_extraction_v1|Q3_K
validator|models_gemma_validator_v1|Q3_K
dialogue|models_gemma_ak_dialog|Q4_K_M
telegram|models_gemma_telegram_post_v1|Q4_K_M
universal|models_universal_gemma_v1|Q4_K_M
coding_web|models_gemma_coding_v2|Q4_K_M
EOF
)

build_one() {
  local label="$1"
  local lora_dir="$2"
  local quant="$3"

  local lora_path="${ROOT_DIR}/${lora_dir}"
  local gguf_path="${ARTIFACTS_DIR}/$(basename "${lora_dir}")-${quant}.gguf"

  echo
  echo "=== ${label} (${quant}) ==="
  echo "lora: ${lora_path}"

  if [[ ! -d "${lora_path}" ]]; then
    echo "skip ${label}: missing lora dir ${lora_path}" >&2
    return
  fi

  if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    "${BUILD_SCRIPT}" "${lora_path}" "${quant}" "${BASE_MODEL}"
  fi

  if [[ ! -f "${gguf_path}" ]]; then
    echo "skip ${label}: gguf not found after build: ${gguf_path}" >&2
    return
  fi

  if [[ "${SKIP_OLLAMA:-0}" == "1" ]]; then
    echo "built gguf: ${gguf_path}"
    return
  fi

  local ollama_name="${OLLAMA_NS}:${label,,}_${quant,,}"
  local modelfile="${ARTIFACTS_DIR}/modelfiles/${label}_${quant}.Modelfile"
  local system_prompt=""

  case "${label}" in
    summary)
      system_prompt='Краткое резюме ситуации. Сразу короткий текст по исходному тексту. Без вводных слов и без пояснений.'
      ;;
    extraction)
      system_prompt='Извлекай структурированные поля и возвращай строго JSON-строку. Без префикса "json", без markdown, без пояснений, только валидный JSON.'
      ;;
    telegram)
      system_prompt='Сформируй готовый пост для Telegram-канала по исходному тексту. Верни только один финальный пост без вариантов, рекомендаций и саммари.'
      ;;
    dialogue)
      system_prompt='Ты добрый и харизматичный собеседник. Отвечай бережно, уместно, поддерживающе и по-человечески.'
      ;;
    coding_web)
      system_prompt='Ты ИИ-ассистент для создания веб-приложений. Выдавай только код/скрипты без лишних пояснений. По умолчанию HTML+CSS+JS и FastAPI, минимальные изменения, качество выше скорости.'
      ;;
    universal)
      system_prompt='Универсальный ассистент: для постов выдавай один финальный текст без вариантов; для summary сразу кратко по сути; для extraction только валидный JSON-строкой; для coding только рабочий код без лишних пояснений.'
      ;;
    qa)
      system_prompt='Отвечай по фактам кратко и точно.'
      ;;
    validator)
      system_prompt='Проверяй качество, находи ошибки и риски, давай четкие замечания.'
      ;;
    *)
      system_prompt='Ты полезный ассистент.'
      ;;
  esac

  cat > "${modelfile}" <<EOF
FROM ${gguf_path}
SYSTEM ${system_prompt}
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
EOF

  echo "register ollama model: ${ollama_name}"
  ollama create "${ollama_name}" -f "${modelfile}"
  echo "ok: ${ollama_name}"
}

while IFS='|' read -r label lora_dir quant; do
  [[ -z "${label}" ]] && continue
  build_one "${label}" "${lora_dir}" "${quant}"
done <<< "${AGENTS}"

echo
echo "Done."
echo "Artifacts: ${ARTIFACTS_DIR}"
if [[ "${SKIP_OLLAMA:-0}" != "1" ]]; then
  echo "Ollama models list:"
  ollama list || true
fi
