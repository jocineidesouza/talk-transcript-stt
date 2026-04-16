#!/usr/bin/env sh
set -eu

HOST_URL="${1:-http://localhost:3001}"
WAV_PATH="${2:-/opt/stack/stt/models/pt_br.wav}"

curl -sS -X POST "${HOST_URL}/transcribe" -F "file=@${WAV_PATH}"
echo
