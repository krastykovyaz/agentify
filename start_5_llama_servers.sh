#!/usr/bin/env bash
set -euo pipefail

# Starts multiple llama-server instances.
# Usage:
#   ./start_5_llama_servers.sh --config ./llama_5_instances.example.csv
#   ./start_5_llama_servers.sh --preset gemma_q4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DEFAULT="${ROOT_DIR}/llama.cpp/build/bin/llama-server"
CONFIG=""
PRESET=""
BIN="${BIN_DEFAULT}"
HOST="127.0.0.1"
LOG_DIR="${ROOT_DIR}/logs/llama_servers"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --preset)
      PRESET="$2"; shift 2 ;;
    --bin)
      BIN="$2"; shift 2 ;;
    --host)
      HOST="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --artifacts-dir)
      ARTIFACTS_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ ! -x "${BIN}" ]]; then
  echo "llama-server not found/executable: ${BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

# Keep track of started pids and cleanup on Ctrl+C
PIDS=()
cleanup() {
  echo
  echo "Stopping ${#PIDS[@]} llama-server process(es)..."
  for p in "${PIDS[@]}"; do
    kill "$p" >/dev/null 2>&1 || true
  done
  wait || true
}
trap cleanup INT TERM

start_one() {
  local name="$1"
  local port="$2"
  local gguf_path="$3"
  local ctx="$4"
  local predict="$5"
  local gpu_layers="$6"
  local temp="$7"
  local top_p="$8"

  gguf_path="${gguf_path//\r/}"
  if [[ ! -f "${gguf_path}" ]]; then
    echo "skip ${name}: gguf not found: ${gguf_path}" >&2
    return
  fi

  local log_file="${LOG_DIR}/${name}.log"
  echo "Starting ${name} on ${HOST}:${port}"

  "${BIN}" \
    -m "${gguf_path}" \
    --host "${HOST}" \
    --port "${port}" \
    -c "${ctx}" \
    -n "${predict}" \
    --temp "${temp}" \
    --top-p "${top_p}" \
    --reasoning-budget 0 \
    >"${log_file}" 2>&1 &

  PIDS+=("$!")
  sleep 1
}

if [[ -n "${PRESET}" ]]; then
  case "${PRESET}" in
    gemma_q4)
      start_one "gemma_summarization_q4" "8081" "${ARTIFACTS_DIR}/models_gemma_summarization-Q4_K_M.gguf" "2048" "220" "0" "0.2" "0.9"
      start_one "gemma_telegram_q4" "8082" "${ARTIFACTS_DIR}/models_gemma_telegram_post_v1-Q4_K_M.gguf" "3072" "260" "0" "0.25" "0.9"
      start_one "gemma_universal_q4" "8083" "${ARTIFACTS_DIR}/models_universal_gemma_v1-Q4_K_M.gguf" "3072" "240" "0" "0.2" "0.9"
      start_one "gemma_extraction_q4" "8084" "${ARTIFACTS_DIR}/models_gemma_extraction_v1-Q4_K_M.gguf" "3072" "260" "0" "0.15" "0.85"
      start_one "gemma_dialog_q4" "8085" "${ARTIFACTS_DIR}/models_gemma_ak_dialog-Q4_K_M.gguf" "2048" "180" "0" "0.45" "0.85"
      ;;
    *)
      echo "Unknown preset: ${PRESET}" >&2
      echo "Supported presets: gemma_q4" >&2
      exit 1
      ;;
  esac
else
  if [[ -z "${CONFIG}" ]]; then
    echo "Pass either --config <csv> or --preset gemma_q4" >&2
    exit 1
  fi

  if [[ ! -f "${CONFIG}" ]]; then
    echo "config not found: ${CONFIG}" >&2
    exit 1
  fi

  while IFS=, read -r name port gguf_path ctx predict gpu_layers temp top_p; do
    [[ -z "${name}" ]] && continue
    [[ "${name}" =~ ^# ]] && continue
    [[ "${name}" == "name" ]] && continue

    ctx="${ctx:-2048}"
    predict="${predict:-256}"
    gpu_layers="${gpu_layers:-0}"
    temp="${temp:-0.2}"
    top_p="${top_p:-0.9}"

    start_one "${name}" "${port}" "${gguf_path}" "${ctx}" "${predict}" "${gpu_layers}" "${temp}" "${top_p}"
  done < "${CONFIG}"
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No instances started. Check model paths." >&2
  exit 1
fi

echo
echo "Started ${#PIDS[@]} instance(s). Logs: ${LOG_DIR}"
echo "Press Ctrl+C to stop all."
wait
