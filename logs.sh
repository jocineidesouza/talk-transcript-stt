#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
SERVICE="stt"
TAIL_LINES="${1:-200}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: docker nao encontrado no PATH." >&2
  exit 1
fi

docker compose -f "${COMPOSE_FILE}" logs -f --tail "${TAIL_LINES}" "${SERVICE}"
