#!/usr/bin/env sh
set -eu

MODELS_ROOT="${MODELS_ROOT:-/models}"
MODEL_NAME="${MODEL_NAME:-sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12}"
MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}/${MODEL_NAME}}"
MODEL_TAR="${MODELS_ROOT}/${MODEL_NAME}.tar.bz2"
MODEL_TAR_TMP="${MODEL_TAR}.part"
MODEL_URL="${MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2}"
PT_TEST_WAV="${PT_TEST_WAV:-${MODELS_ROOT}/pt_br.wav}"
PT_TEST_WAV_URL="${PT_TEST_WAV_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/pt_br.wav}"
AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-1}"
MODEL_TYPE="${MODEL_TYPE:-nemo_ctc_offline_vad_streaming}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
COHERE_ENCODER_DATA_MIN_BYTES="${COHERE_ENCODER_DATA_MIN_BYTES:-2500000000}"

file_nonempty() {
  [ -f "$1" ] && [ -s "$1" ]
}

file_size_at_least() {
  path="$1"
  min_bytes="$2"
  if [ ! -f "$path" ]; then
    return 1
  fi
  size="$(wc -c < "$path" | tr -d ' ')"
  [ "${size:-0}" -ge "$min_bytes" ]
}

download_model_with_progress() {
  url="$1"
  output="$2"
  attempts="$3"
  attempt=1
  while [ "$attempt" -le "$attempts" ]; do
    echo "Baixando modelo tentativa ${attempt}/${attempts}."
    if python3 - "$url" "$output" <<'PY'
import os
import sys
import urllib.request

url = sys.argv[1]
out_path = sys.argv[2]
tmp_path = f"{out_path}.tmp"
milestones = list(range(0, 101, 10))
printed = set()

def emit(percent: int) -> None:
    if percent in printed:
        return
    printed.add(percent)
    print(f"baixando modelo {percent}%")
    sys.stdout.flush()

emit(0)

with urllib.request.urlopen(url, timeout=120) as response, open(tmp_path, "wb") as out_file:
    total = response.headers.get("Content-Length")
    total_size = int(total) if total and total.isdigit() else 0
    downloaded = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        out_file.write(chunk)
        downloaded += len(chunk)
        if total_size > 0:
            pct = int((downloaded * 100) / total_size)
            for milestone in milestones:
                if milestone <= pct:
                    emit(milestone)
    out_file.flush()
    os.fsync(out_file.fileno())

os.replace(tmp_path, out_path)
emit(100)
PY
    then
      return 0
    fi

    echo "Falha ao baixar modelo na tentativa ${attempt}."
    rm -f "${output}.tmp" "$output"
    attempt=$((attempt + 1))
    sleep 2
  done
  return 1
}

is_model_ready() {
  if [ "${MODEL_TYPE#cohere_transcribe}" != "$MODEL_TYPE" ]; then
    if file_nonempty "${MODEL_DIR}/encoder.int8.onnx" && \
      file_nonempty "${MODEL_DIR}/decoder.int8.onnx" && \
      file_size_at_least "${MODEL_DIR}/encoder.int8.onnx.data" "$COHERE_ENCODER_DATA_MIN_BYTES"; then
      return 0
    fi
    if file_nonempty "${MODEL_DIR}/encoder.onnx" && \
      file_nonempty "${MODEL_DIR}/decoder.onnx" && \
      file_size_at_least "${MODEL_DIR}/encoder.onnx.data" "$COHERE_ENCODER_DATA_MIN_BYTES"; then
      return 0
    fi
    return 1
  fi

  if { file_nonempty "${MODEL_DIR}/model.int8.onnx" || file_nonempty "${MODEL_DIR}/model.onnx"; } && \
    file_nonempty "${MODEL_DIR}/tokens.txt"; then
    return 0
  fi
  return 1
}

if [ "$AUTO_DOWNLOAD_MODEL" != "1" ]; then
  echo "AUTO_DOWNLOAD_MODEL=${AUTO_DOWNLOAD_MODEL}; pulando bootstrap de modelos."
  exit 0
fi

mkdir -p "$MODELS_ROOT"

if ! is_model_ready; then
  echo "Modelo nao encontrado em ${MODEL_DIR}."
  rm -rf "$MODEL_DIR"
  rm -f "$MODEL_TAR_TMP"

  download_model_with_progress "$MODEL_URL" "$MODEL_TAR_TMP" "$DOWNLOAD_RETRIES"
  echo "Extraindo modelo em ${MODELS_ROOT}."
  python3 - "$MODEL_TAR_TMP" "$MODELS_ROOT" <<'PY'
import sys
import tarfile

archive = sys.argv[1]
models_root = sys.argv[2]

with tarfile.open(archive) as tf:
    tf.extractall(models_root)
PY
  mv -f "$MODEL_TAR_TMP" "$MODEL_TAR"
fi

if ! is_model_ready; then
  echo "Falha no bootstrap: arquivos do modelo ausentes ou invalidos em ${MODEL_DIR}."
  exit 1
fi

if [ ! -f "$PT_TEST_WAV" ]; then
  echo "Audio de teste PT-BR nao encontrado; baixando em ${PT_TEST_WAV}."
  curl -fL -o "$PT_TEST_WAV" "$PT_TEST_WAV_URL"
fi

echo "Bootstrap de modelos concluido."
