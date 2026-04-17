FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01 \
    VAD_MODEL=/models/silero_vad.onnx \
    SHERPA_NUM_THREADS=3 \
    DECODE_MAX_CONCURRENCY=1 \
    UVICORN_WORKERS=1 \
    SAMPLE_RATE=16000 \
    FEATURE_DIM=80 \
    MODEL_LANGUAGE=pt \
    MODEL_TYPE=cohere_transcribe_offline_vad_streaming \
    USE_INTERNAL_VAD=true \
    AUTO_DOWNLOAD_MODEL=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY bootstrap_models.sh .
COPY ws_client_test.py .
RUN chmod +x /app/bootstrap_models.sh

EXPOSE 3001

CMD ["sh", "-c", "/app/bootstrap_models.sh && uvicorn app:app --host 0.0.0.0 --port 3001 --workers ${UVICORN_WORKERS:-1}"]
