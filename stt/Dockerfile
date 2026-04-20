FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01 \
    SHERPA_NUM_THREADS=3 \
    DECODE_MAX_CONCURRENCY=1 \
    UVICORN_WORKERS=1 \
    SAMPLE_RATE=16000 \
    FEATURE_DIM=80 \
    MODEL_LANGUAGE=pt \
    MODEL_TYPE=cohere_transcribe_offline_vad_streaming \
    SQLITE_PATH=/data/queue.db \
    SPOOL_DIR=/data/spool \
    QUEUE_MAX_PENDING=2000 \
    HMAC_WINDOW_SECONDS=300 \
    STT_HMAC_KEY_ID=default \
    STT_HMAC_SECRET=replace_with_long_random_secret \
    FIREBASE_ENABLED=false \
    FIREBASE_FLUSH_INTERVAL_SECONDS=30 \
    STORAGE_MINUTE_WINDOW_SECONDS=60 \
    OPENAI_SUMMARY_ENABLED=false \
    OPENAI_APIKEY_FILE=/secrets/openai_apikey.json \
    OPENAI_MODEL_MINUTE_SUMMARY=gpt-4.1-mini \
    OPENAI_MODEL_ACCUMULATED_SUMMARY=gpt-4.1 \
    OPENAI_MODEL_FINAL_SUMMARY=gpt-4.1 \
    OPENAI_REQUEST_TIMEOUT_SECONDS=20 \
    OPENAI_MAX_RETRIES=3 \
    AUTO_DOWNLOAD_MODEL=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY bootstrap_models.sh .
RUN chmod +x /app/bootstrap_models.sh

EXPOSE 3001

CMD ["sh", "-c", "/app/bootstrap_models.sh && uvicorn app:app --host 0.0.0.0 --port 3001 --workers ${UVICORN_WORKERS:-1}"]
