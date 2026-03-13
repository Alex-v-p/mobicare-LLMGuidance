#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building and starting the Docker Compose stack..."
docker compose up -d --build "$@"

echo "Installing Ollama models..."
"$ROOT_DIR/scripts/pull-ollama-models.sh"

echo "Startup complete."
