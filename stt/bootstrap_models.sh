#!/usr/bin/env sh
set -eu

MODELS_ROOT="${MODELS_ROOT:-/models}"
MODEL_NAME="${MODEL_NAME:-sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc-int8}"
MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}/${MODEL_NAME}}"
MODEL_TAR="${MODELS_ROOT}/${MODEL_NAME}.tar.bz2"
MODEL_URL="${MODEL_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2}"
VAD_MODEL="${VAD_MODEL:-${MODELS_ROOT}/silero_vad.onnx}"
VAD_URL="${VAD_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx}"
PT_TEST_WAV="${PT_TEST_WAV:-${MODELS_ROOT}/pt_br.wav}"
PT_TEST_WAV_URL="${PT_TEST_WAV_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/pt_br.wav}"
AUTO_DOWNLOAD_MODEL="${AUTO_DOWNLOAD_MODEL:-1}"

if [ "$AUTO_DOWNLOAD_MODEL" != "1" ]; then
  echo "AUTO_DOWNLOAD_MODEL=${AUTO_DOWNLOAD_MODEL}; pulando bootstrap de modelos."
  exit 0
fi

mkdir -p "$MODELS_ROOT"

if [ ! -f "${MODEL_DIR}/model.int8.onnx" ] || [ ! -f "${MODEL_DIR}/tokens.txt" ]; then
  echo "Modelo PT-BR nao encontrado em ${MODEL_DIR}; iniciando download."
  if [ ! -f "$MODEL_TAR" ]; then
    curl -fL -o "$MODEL_TAR" "$MODEL_URL"
  fi

  python3 -c "import tarfile; tarfile.open('${MODEL_TAR}').extractall('${MODELS_ROOT}')"
fi

if [ ! -f "$VAD_MODEL" ]; then
  echo "VAD nao encontrado em ${VAD_MODEL}; iniciando download."
  curl -fL -o "$VAD_MODEL" "$VAD_URL"
fi

if [ ! -f "$PT_TEST_WAV" ]; then
  echo "Audio de teste PT-BR nao encontrado; baixando em ${PT_TEST_WAV}."
  curl -fL -o "$PT_TEST_WAV" "$PT_TEST_WAV_URL"
fi

echo "Bootstrap de modelos concluido."
