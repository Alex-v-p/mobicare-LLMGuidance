#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

OLLAMA_SERVICE="${OLLAMA_SERVICE:-ollama}"
OLLAMA_START_TIMEOUT_SECONDS="${OLLAMA_START_TIMEOUT_SECONDS:-180}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"

models=("$OLLAMA_EMBEDDING_MODEL" "$OLLAMA_MODEL")
if (( $# > 0 )); then
  models+=("$@")
fi

# De-duplicate while preserving order.
unique_models=()
for model in "${models[@]}"; do
  [[ -z "$model" ]] && continue
  skip=false
  for seen in "${unique_models[@]:-}"; do
    if [[ "$seen" == "$model" ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == false ]]; then
    unique_models+=("$model")
  fi
done

if (( ${#unique_models[@]} == 0 )); then
  echo "No Ollama models configured."
  exit 0
fi

echo "Waiting for the '$OLLAMA_SERVICE' container to accept Ollama commands..."
deadline=$((SECONDS + OLLAMA_START_TIMEOUT_SECONDS))
until docker compose exec -T "$OLLAMA_SERVICE" ollama list >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out while waiting for Ollama after ${OLLAMA_START_TIMEOUT_SECONDS}s." >&2
    exit 1
  fi
  sleep 2
done

for model in "${unique_models[@]}"; do
  echo "Pulling Ollama model: $model"
  docker compose exec -T "$OLLAMA_SERVICE" ollama pull "$model"
done

echo "Ollama model pull complete."
