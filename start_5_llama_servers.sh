#!/usr/bin/env bash
set -euo pipefail

# Starts multiple llama-server instances from one config file.
# Usage:
#   ./start_5_llama_servers.sh --config ./llama_5_instances.example.csv
#
# Config CSV format (with header):
# name,port,gguf_path,ctx,predict,gpu_layers,temp,top_p

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DEFAULT="${ROOT_DIR}/llama.cpp/build/bin/llama-server"
CONFIG=""
BIN="${BIN_DEFAULT}"
HOST="127.0.0.1"
LOG_DIR="${ROOT_DIR}/logs/llama_servers"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --bin)
      BIN="$2"; shift 2 ;;
    --host)
      HOST="$2"; shift 2 ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Pass --config <csv>" >&2
  exit 1
fi

if [[ ! -x "${BIN}" ]]; then
  echo "llama-server not found/executable: ${BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "config not found: ${CONFIG}" >&2
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

# skip header
TAIL_CMD=(tail -n +2 "${CONFIG}")

while IFS=, read -r name port gguf_path ctx predict gpu_layers temp top_p; do
  [[ -z "${name}" ]] && continue
  [[ "${name}" =~ ^# ]] && continue

  gguf_path="${gguf_path//\r/}"
  if [[ ! -f "${gguf_path}" ]]; then
    echo "skip ${name}: gguf not found: ${gguf_path}" >&2
    continue
  fi

  ctx="${ctx:-2048}"
  predict="${predict:-256}"
  gpu_layers="${gpu_layers:-0}"
  temp="${temp:-0.2}"
  top_p="${top_p:-0.9}"

  log_file="${LOG_DIR}/${name}.log"
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

done < <("${TAIL_CMD[@]}")

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "No instances started. Check config paths." >&2
  exit 1
fi

echo
echo "Started ${#PIDS[@]} instance(s). Logs: ${LOG_DIR}"
echo "Press Ctrl+C to stop all."
wait
