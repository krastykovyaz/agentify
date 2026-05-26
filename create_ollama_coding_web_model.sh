#!/usr/bin/env bash
set -euo pipefail

# Create/update Ollama coding_web model with strict system prompt.
# Usage:
#   ./create_ollama_coding_web_model.sh
#   ./create_ollama_coding_web_model.sh --from agentify:coding_web_q4_k_m --to agentify:coding_web_q4_k_m

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM_MODEL="agentify:coding_web_q4_k_m"
TO_MODEL="agentify:coding_web_q4_k_m"
PROMPT_FILE="${ROOT_DIR}/prompts_coding_web_system.txt"
MOD_DIR="${ROOT_DIR}/artifacts/modelfiles"
MOD_FILE="${MOD_DIR}/coding_web_runtime.Modelfile"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_MODEL="$2"; shift 2 ;;
    --to)
      TO_MODEL="$2"; shift 2 ;;
    --prompt-file)
      PROMPT_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

command -v ollama >/dev/null 2>&1 || { echo "ollama not found" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "prompt file not found: $PROMPT_FILE" >&2; exit 1; }

mkdir -p "$MOD_DIR"

{
  echo "FROM ${FROM_MODEL}"
  echo "SYSTEM \"\"\""
  cat "$PROMPT_FILE"
  echo "\"\"\""
  echo "PARAMETER temperature 0.15"
  echo "PARAMETER top_p 0.9"
  echo "PARAMETER num_ctx 8192"
} > "$MOD_FILE"

echo "Creating Ollama model: $TO_MODEL"
ollama create "$TO_MODEL" -f "$MOD_FILE"

echo "Done: $TO_MODEL"
echo "Modelfile: $MOD_FILE"
