#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"
SERVICE="stt"
DB_PATH="${ROOT_DIR}/stt/data/queue.db"
RETENTION_DAYS="30"
DELETE="0"

usage() {
  cat <<USAGE
Uso: ./cleanup_old_sessions.sh [--days N] [--delete]

Limpa sessoes finalizadas antigas do SQLite.

Opcoes:
  --days N     Mantem sessoes finalizadas nos ultimos N dias. Default: 30.
  --delete     Executa a limpeza. Sem esta opcao, apenas mostra o que faria.
  -h, --help   Mostra esta ajuda.

Exemplos:
  ./cleanup_old_sessions.sh
  ./cleanup_old_sessions.sh --days 60
  ./cleanup_old_sessions.sh --days 30 --delete
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days)
      if [[ $# -lt 2 ]]; then
        echo "Erro: --days exige um numero." >&2
        exit 1
      fi
      RETENTION_DAYS="$2"
      shift 2
      ;;
    --delete)
      DELETE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Erro: opcao desconhecida: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]] || [[ "${RETENTION_DAYS}" -lt 1 ]]; then
  echo "Erro: --days deve ser um inteiro maior que zero." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: docker nao encontrado no PATH." >&2
  exit 1
fi

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "Erro: sqlite3 nao encontrado no PATH." >&2
  echo "Instale com: sudo apt-get install sqlite3" >&2
  exit 1
fi

if [[ ! -f "${DB_PATH}" ]]; then
  echo "Erro: banco SQLite nao encontrado em ${DB_PATH}" >&2
  exit 1
fi

CUTOFF="$(date -u -d "-${RETENTION_DAYS} days" "+%Y-%m-%dT%H:%M:%S+00:00")"
SESSION_COUNT="$(
  sqlite3 "${DB_PATH}" "
    SELECT COUNT(*)
    FROM sessions
    WHERE finalized_at IS NOT NULL
      AND finalized_at <> ''
      AND finalized_at < '${CUTOFF}';
  "
)"

echo "Retencao: ${RETENTION_DAYS} dias. Use --days N para informar outro numero de dias."
echo "Cutoff UTC: ${CUTOFF}"
echo "Sessoes finalizadas antigas encontradas: ${SESSION_COUNT}"

if [[ "${DELETE}" != "1" ]]; then
  sqlite3 -header -column "${DB_PATH}" "
    SELECT room_name, session_id, finalized_at
    FROM sessions
    WHERE finalized_at IS NOT NULL
      AND finalized_at <> ''
      AND finalized_at < '${CUTOFF}'
    ORDER BY finalized_at
    LIMIT 50;
  "
  echo "Simulacao concluida. Nenhum dado foi apagado."
  echo "Para executar a limpeza, rode: ./cleanup_old_sessions.sh --days ${RETENTION_DAYS} --delete"
  exit 0
fi

if [[ "${SESSION_COUNT}" == "0" ]]; then
  echo "Nada para limpar."
  exit 0
fi

BACKUP_PATH="${DB_PATH}.bak-$(date -u "+%Y%m%d-%H%M%S")"

echo "[1/5] Parando servico ${SERVICE}..."
docker compose -f "${COMPOSE_FILE}" stop "${SERVICE}"

echo "[2/5] Criando backup em ${BACKUP_PATH}..."
cp "${DB_PATH}" "${BACKUP_PATH}"

echo "[3/5] Limpando sessoes finalizadas antigas..."
sqlite3 "${DB_PATH}" <<SQL
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE;

CREATE TEMP TABLE cleanup_sessions AS
SELECT room_name, session_id
FROM sessions
WHERE finalized_at IS NOT NULL
  AND finalized_at <> ''
  AND finalized_at < '${CUTOFF}';

DELETE FROM summary_tasks
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = summary_tasks.room_name
    AND c.session_id = summary_tasks.session_id
);

DELETE FROM minute_exports
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = minute_exports.room_name
    AND c.session_id = minute_exports.session_id
);

DELETE FROM transcripts
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = transcripts.room_name
    AND c.session_id = transcripts.session_id
);

DELETE FROM chunks
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = chunks.room_name
    AND c.session_id = chunks.session_id
);

DELETE FROM participants
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = participants.room_name
    AND c.session_id = participants.session_id
);

DELETE FROM sessions
WHERE EXISTS (
  SELECT 1 FROM cleanup_sessions c
  WHERE c.room_name = sessions.room_name
    AND c.session_id = sessions.session_id
);

COMMIT;
VACUUM;
SQL

echo "[4/5] Subindo servico ${SERVICE}..."
docker compose -f "${COMPOSE_FILE}" up -d "${SERVICE}"

echo "[5/5] Status do servico ${SERVICE}:"
docker compose -f "${COMPOSE_FILE}" ps "${SERVICE}"

echo "Concluido. Backup mantido em: ${BACKUP_PATH}"
