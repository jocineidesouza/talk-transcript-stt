#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
SERVICE="stt"

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: docker nao encontrado no PATH." >&2
  exit 1
fi

echo "[1/3] Atualizando repositorio..."
git -C "${ROOT_DIR}" pull --rebase --autostash

echo "[2/3] Build e deploy do servico ${SERVICE}..."
docker compose -f "${COMPOSE_FILE}" up --build -d "${SERVICE}"

echo "[3/3] Status do servico ${SERVICE}:"
docker compose -f "${COMPOSE_FILE}" ps "${SERVICE}"

echo "Concluido."
