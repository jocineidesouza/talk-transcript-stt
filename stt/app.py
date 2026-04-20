from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import firebase_admin
import numpy as np
import sherpa_onnx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from firebase_admin import credentials, firestore, storage
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.datastructures import UploadFile


MODEL_DIR = Path(
    os.environ.get(
        "MODEL_DIR",
        "/models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    )
)
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "3"))
DECODE_MAX_CONCURRENCY = int(os.environ.get("DECODE_MAX_CONCURRENCY", "1"))
TAIL_PADDING_SECONDS = float(os.environ.get("TAIL_PADDING_SECONDS", "0.35"))
MODEL_LANGUAGE = os.environ.get("MODEL_LANGUAGE", "pt")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "cohere_transcribe_offline_vad_streaming")
FEATURE_DIM = int(os.environ.get("FEATURE_DIM", "80"))

SQLITE_PATH = Path(os.environ.get("SQLITE_PATH", "/data/queue.db"))
SPOOL_DIR = Path(os.environ.get("SPOOL_DIR", "/data/spool"))
QUEUE_MAX_PENDING = int(os.environ.get("QUEUE_MAX_PENDING", "2000"))
WORKER_POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "0.4"))
ROOM_INACTIVITY_TIMEOUT_SECONDS = int(
    os.environ.get("ROOM_INACTIVITY_TIMEOUT_SECONDS", "1800")
)

HMAC_WINDOW_SECONDS = int(os.environ.get("HMAC_WINDOW_SECONDS", "300"))
HMAC_KEYS_RAW = os.environ.get("STT_HMAC_KEYS", "")
HMAC_KEY_ID = os.environ.get("STT_HMAC_KEY_ID", "default")
HMAC_SECRET = os.environ.get("STT_HMAC_SECRET", "")

FIREBASE_ENABLED = os.environ.get("FIREBASE_ENABLED", "false").lower() == "true"
FIREBASE_NAMESPACE_CONFIG_JSON = os.environ.get("FIREBASE_NAMESPACE_CONFIG_JSON", "")
FIREBASE_FLUSH_INTERVAL_SECONDS = max(
    5, int(os.environ.get("FIREBASE_FLUSH_INTERVAL_SECONDS", "30"))
)
STORAGE_MINUTE_WINDOW_SECONDS = max(
    10, int(os.environ.get("STORAGE_MINUTE_WINDOW_SECONDS", "60"))
)

OPENAI_SUMMARY_ENABLED = os.environ.get("OPENAI_SUMMARY_ENABLED", "false").lower() == "true"
OPENAI_APIKEY_FILE = Path(os.environ.get("OPENAI_APIKEY_FILE", "/secrets/openai_apikey.json"))
OPENAI_MODEL_MINUTE_SUMMARY = os.environ.get(
    "OPENAI_MODEL_MINUTE_SUMMARY", "gpt-4.1-mini"
).strip()
OPENAI_MODEL_ACCUMULATED_SUMMARY = os.environ.get(
    "OPENAI_MODEL_ACCUMULATED_SUMMARY", "gpt-4.1"
).strip()
OPENAI_MODEL_FINAL_SUMMARY = os.environ.get(
    "OPENAI_MODEL_FINAL_SUMMARY", "gpt-4.1"
).strip()
OPENAI_REQUEST_TIMEOUT_SECONDS = max(
    5, int(os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "20"))
)
OPENAI_MAX_RETRIES = max(1, int(os.environ.get("OPENAI_MAX_RETRIES", "3")))

ALLOWED_LIVEKIT_NAMESPACES = frozenset(
    (
        "talk__dev",
        "talk__stg",
        "talk__prd",
        "ellevo-connect__dev",
        "ellevo-connect__stg",
        "ellevo-connect__prd",
    )
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stt")


class StartRequest(BaseModel):
    room_name: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    participant_identity: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    participant_name: str | None = None
    track_sid: str | None = None
    metadata: dict | None = None


class ChunkMeta(BaseModel):
    room_name: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    participant_identity: str = Field(min_length=1)
    seq: int = Field(ge=1)
    chunk_started_at: str = Field(min_length=1)
    chunk_ended_at: str = Field(min_length=1)
    sample_rate: int
    channels: int
    encoding: str
    participant_name: str | None = None
    track_sid: str | None = None

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value != SAMPLE_RATE:
            raise ValueError(f"sample_rate deve ser {SAMPLE_RATE}")
        return value

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: int) -> int:
        if value != 1:
            raise ValueError("channels deve ser 1")
        return value

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        if value != "pcm_s16le":
            raise ValueError("encoding deve ser pcm_s16le")
        return value


class EndRequest(BaseModel):
    room_name: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    participant_identity: str | None = None
    ended_at: str | None = None
    metadata: dict | None = None

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        if value not in ("participant", "room"):
            raise ValueError("scope deve ser participant ou room")
        return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


@dataclass(frozen=True)
class RoomNamespaceInfo:
    namespace: str
    room_id: str


@dataclass(frozen=True)
class FirebaseNamespaceConfig:
    project_id: str
    storage_bucket: str
    credentials_file: str


@dataclass(frozen=True)
class MinuteShardPayload:
    room_name: str
    session_id: str
    call_key: str
    minute_index: int
    minute_started_at: str
    minute_ended_at: str
    transcript_json_path: str
    summary_json_path: str
    lines: list[dict]
    line_count: int
    transcript_hash: str
    finalized: bool


def extract_room_namespace(room_name: str) -> RoomNamespaceInfo | None:
    for namespace in sorted(ALLOWED_LIVEKIT_NAMESPACES, key=len, reverse=True):
        prefix = f"{namespace}__"
        if room_name.startswith(prefix):
            room_id = room_name[len(prefix) :].strip()
            if room_id:
                return RoomNamespaceInfo(namespace=namespace, room_id=room_id)
            return None
    return None


def parse_firebase_namespace_configs() -> dict[str, FirebaseNamespaceConfig]:
    if not FIREBASE_NAMESPACE_CONFIG_JSON.strip():
        if FIREBASE_ENABLED:
            logger.warning(
                "FIREBASE_ENABLED=true, mas FIREBASE_NAMESPACE_CONFIG_JSON nao foi configurado"
            )
        return {}

    try:
        raw = json.loads(FIREBASE_NAMESPACE_CONFIG_JSON)
    except json.JSONDecodeError as exc:
        raise ValueError("FIREBASE_NAMESPACE_CONFIG_JSON invalido: JSON malformado") from exc

    if not isinstance(raw, dict):
        raise ValueError("FIREBASE_NAMESPACE_CONFIG_JSON deve ser um objeto JSON")

    configs: dict[str, FirebaseNamespaceConfig] = {}
    for namespace, conf in raw.items():
        if not isinstance(namespace, str):
            raise ValueError("FIREBASE_NAMESPACE_CONFIG_JSON contem namespace invalido")
        if namespace not in ALLOWED_LIVEKIT_NAMESPACES:
            raise ValueError(
                f"FIREBASE_NAMESPACE_CONFIG_JSON contem namespace nao permitido: {namespace}"
            )
        if not isinstance(conf, dict):
            raise ValueError(f"configuracao do namespace {namespace} deve ser objeto JSON")

        credentials_file = str(conf.get("credentials_file", "")).strip()
        if not credentials_file:
            raise ValueError(
                f"namespace {namespace}: campo credentials_file obrigatorio e nao vazio"
            )
        project_id = str(conf.get("project_id", "")).strip()
        storage_bucket = str(conf.get("storage_bucket", "")).strip()
        configs[namespace] = FirebaseNamespaceConfig(
            project_id=project_id,
            storage_bucket=storage_bucket,
            credentials_file=credentials_file,
        )

    return configs


def ignored_namespace_response(route: str, room_name: str, reason: str) -> JSONResponse:
    logger.info(
        "ingest ignored route=%s room=%s reason=%s",
        route,
        room_name,
        reason,
    )
    return JSONResponse(
        {"status": "ignored", "reason": reason, "room_name": room_name},
        status_code=202,
    )


def read_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)

    conn = read_db_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                metadata_json TEXT,
                state TEXT NOT NULL DEFAULT 'active',
                room_end_received INTEGER NOT NULL DEFAULT 0,
                last_chunk_at TEXT,
                last_firebase_flush_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                finalized_at TEXT,
                PRIMARY KEY(room_name, session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                participant_identity TEXT NOT NULL,
                participant_name TEXT,
                track_sid TEXT,
                started_at TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                last_seq INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ended_at TEXT,
                finalized_at TEXT,
                PRIMARY KEY(room_name, session_id, participant_identity)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                participant_identity TEXT NOT NULL,
                seq INTEGER NOT NULL,
                track_sid TEXT,
                chunk_started_at TEXT NOT NULL,
                chunk_ended_at TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                encoding TEXT NOT NULL,
                spool_path TEXT NOT NULL,
                status TEXT NOT NULL,
                transcript TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                processed_at TEXT,
                PRIMARY KEY(room_name, session_id, participant_identity, seq)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                participant_identity TEXT NOT NULL,
                seq INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(room_name, session_id, participant_identity, seq)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hmac_nonces (
                key_id TEXT NOT NULL,
                nonce TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(key_id, nonce)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_status_created_at ON chunks(status, created_at)"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_call_status
            ON chunks(room_name, session_id, status, created_at)
            """
        )
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "last_chunk_at" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_chunk_at TEXT")
        if "last_firebase_flush_at" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN last_firebase_flush_at TEXT")
        conn.execute(
            "UPDATE sessions SET last_chunk_at = started_at WHERE last_chunk_at IS NULL"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_inactivity
            ON sessions(room_end_received, finalized_at, last_chunk_at)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS minute_exports (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                minute_index INTEGER NOT NULL,
                transcript_json_path TEXT NOT NULL,
                summary_json_path TEXT,
                content_hash TEXT NOT NULL,
                minute_started_at TEXT NOT NULL,
                minute_ended_at TEXT NOT NULL,
                finalized INTEGER NOT NULL DEFAULT 0,
                exported_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(room_name, session_id, minute_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_tasks (
                room_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                minute_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                retries INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(room_name, session_id, minute_index)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_summary_tasks_poll
            ON summary_tasks(status, next_attempt_at, updated_at)
            """
        )
    finally:
        conn.close()


def parse_hmac_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    if HMAC_KEYS_RAW.strip():
        for entry in HMAC_KEYS_RAW.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError("STT_HMAC_KEYS invalido. Use formato keyId:secret,keyId2:secret2")
            key_id, secret = entry.split(":", 1)
            key_id = key_id.strip()
            secret = secret.strip()
            if not key_id or not secret:
                raise ValueError("STT_HMAC_KEYS contem chave ou segredo vazio")
            keys[key_id] = secret

    if not keys and HMAC_SECRET:
        keys[HMAC_KEY_ID] = HMAC_SECRET

    if not keys:
        raise ValueError("Configure STT_HMAC_SECRET ou STT_HMAC_KEYS")

    return keys


def require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo ausente: {path}")
    return str(path)


def require_one(paths: list[Path]) -> str:
    for path in paths:
        if path.is_file():
            return str(path)
    options = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Nenhum arquivo encontrado entre: {options}")


def build_offline_recognizer() -> sherpa_onnx.OfflineRecognizer:
    if MODEL_TYPE.startswith("cohere_transcribe"):
        tokens_path = MODEL_DIR / "tokens.txt"
        return sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=require_one([MODEL_DIR / "encoder.int8.onnx", MODEL_DIR / "encoder.onnx"]),
            decoder=require_one([MODEL_DIR / "decoder.int8.onnx", MODEL_DIR / "decoder.onnx"]),
            tokens=str(tokens_path) if tokens_path.is_file() else "",
            num_threads=NUM_THREADS,
            language=MODEL_LANGUAGE,
            use_punct=True,
            use_itn=True,
            provider="cpu",
            decoding_method="greedy_search",
            debug=False,
        )

    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=require_one([MODEL_DIR / "model.int8.onnx", MODEL_DIR / "model.onnx"]),
        tokens=require_file(MODEL_DIR / "tokens.txt"),
        num_threads=NUM_THREADS,
        provider="cpu",
        decoding_method="greedy_search",
        debug=False,
    )


def decode_samples(recognizer: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def decode_with_tail_padding(recognizer: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    padded = np.concatenate(
        [samples, np.zeros(int(SAMPLE_RATE * TAIL_PADDING_SECONDS), dtype=np.float32)]
    )
    return decode_samples(recognizer, padded)


def verify_hmac_or_raise(
    request: Request,
    body_bytes: bytes,
    hmac_keys: dict[str, str],
) -> None:
    timestamp_raw = request.headers.get("X-TS-Timestamp", "")
    nonce = request.headers.get("X-TS-Nonce", "")
    signature = request.headers.get("X-TS-Signature", "")
    key_id = request.headers.get("X-TS-Key-Id", "")

    if not (timestamp_raw and nonce and signature and key_id):
        raise HTTPException(status_code=401, detail="cabecalhos HMAC ausentes")

    secret = hmac_keys.get(key_id)
    if not secret:
        raise HTTPException(status_code=401, detail="key id invalido")

    try:
        timestamp = int(timestamp_raw)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="timestamp invalido") from exc

    now_ts = int(datetime.now(timezone.utc).timestamp())
    if abs(now_ts - timestamp) > HMAC_WINDOW_SECONDS:
        raise HTTPException(status_code=401, detail="timestamp fora da janela")

    body_hash = hashlib.sha256(body_bytes).hexdigest()
    canonical = (
        f"{request.method.upper()}\n"
        f"{request.url.path}\n"
        f"{timestamp_raw}\n"
        f"{nonce}\n"
        f"{body_hash}"
    )
    expected = hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="assinatura invalida")

    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM hmac_nonces WHERE timestamp < ?",
            (now_ts - HMAC_WINDOW_SECONDS,),
        )
        try:
            conn.execute(
                """
                INSERT INTO hmac_nonces(key_id, nonce, timestamp, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key_id, nonce, timestamp, utc_now_iso()),
            )
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise HTTPException(status_code=401, detail="nonce repetido") from exc
        conn.execute("COMMIT")
    finally:
        conn.close()


@dataclass
class FirebaseSink:
    namespace: str
    enabled: bool
    firestore_client: firestore.Client | None
    storage_bucket: storage.bucket.Bucket | None

    def publish_call_index(
        self,
        room_name: str,
        session_id: str,
        room_id: str,
        status: str,
        last_minute_index: int,
        finalized: bool,
        minute_window_seconds: int,
        flush_interval_seconds: int,
        final_summary_path: str | None = None,
        summary_accumulated_path: str | None = None,
    ) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        call_key = f"{room_name}__{session_id}"
        call_ref = self.firestore_client.collection("calls").document(call_key)
        call_ref.set(
            {
                "room_name": room_name,
                "room_id": room_id,
                "session_id": session_id,
                "call_key": call_key,
                "namespace": self.namespace,
                "status": status,
                "finalized": finalized,
                "last_minute_index": last_minute_index,
                "minute_window_seconds": minute_window_seconds,
                "flush_interval_seconds": flush_interval_seconds,
                "storage_base": f"calls/{safe_key(call_key)}/",
                "summary_accumulated_path": summary_accumulated_path,
                "final_summary_path": final_summary_path,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def upsert_minute_shard(self, payload: MinuteShardPayload) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        shard_ref = (
            self.firestore_client.collection("calls")
            .document(payload.call_key)
            .collection("minute_shards")
            .document(str(payload.minute_index))
        )
        shard_ref.set(
            {
                "minute_index": payload.minute_index,
                "minute_started_at": payload.minute_started_at,
                "minute_ended_at": payload.minute_ended_at,
                "transcript_json_path": payload.transcript_json_path,
                "summary_json_path": payload.summary_json_path,
                "line_count": payload.line_count,
                "transcript_hash": payload.transcript_hash,
                "finalized": payload.finalized,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def update_minute_summary_link(
        self, call_key: str, minute_index: int, summary_json_path: str
    ) -> None:
        if not self.enabled or self.firestore_client is None:
            return
        shard_ref = (
            self.firestore_client.collection("calls")
            .document(call_key)
            .collection("minute_shards")
            .document(str(minute_index))
        )
        shard_ref.set(
            {
                "summary_json_path": summary_json_path,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def upload_json(self, object_path: str, body: dict) -> None:
        if not self.enabled or self.storage_bucket is None:
            return

        json_blob = self.storage_bucket.blob(object_path)
        json_blob.upload_from_string(
            json.dumps(body, ensure_ascii=False),
            content_type="application/json",
        )

    def fetch_json(self, object_path: str) -> dict | None:
        if not self.enabled or self.storage_bucket is None:
            return None
        blob = self.storage_bucket.blob(object_path)
        if not blob.exists():
            return None
        payload = blob.download_as_text()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def upload_minute_transcript(self, payload: MinuteShardPayload) -> None:
        self.upload_json(
            payload.transcript_json_path,
            {
                "room_name": payload.room_name,
                "session_id": payload.session_id,
                "call_key": payload.call_key,
                "namespace": self.namespace,
                "minute_index": payload.minute_index,
                "minute_started_at": payload.minute_started_at,
                "minute_ended_at": payload.minute_ended_at,
                "line_count": payload.line_count,
                "lines": payload.lines,
            },
        )


class FirebaseRouter:
    def __init__(
        self,
        enabled: bool,
        configs_by_namespace: dict[str, FirebaseNamespaceConfig] | None = None,
    ) -> None:
        self.enabled = enabled
        self.configs_by_namespace = configs_by_namespace or {}
        self._sinks: dict[str, FirebaseSink] = {}
        self._lock = threading.Lock()
        self._disabled_sink = FirebaseSink(
            namespace="disabled",
            enabled=False,
            firestore_client=None,
            storage_bucket=None,
        )

    @staticmethod
    def create() -> "FirebaseRouter":
        if not FIREBASE_ENABLED:
            logger.info("Firebase desabilitado (FIREBASE_ENABLED=false)")
            return FirebaseRouter(False, {})

        configs = parse_firebase_namespace_configs()
        logger.info(
            "Firebase namespace router habilitado: %s namespace(s) configurado(s)",
            len(configs),
        )
        return FirebaseRouter(True, configs)

    def _get_or_create_sink(self, namespace: str) -> FirebaseSink:
        if not self.enabled:
            return self._disabled_sink
        cached = self._sinks.get(namespace)
        if cached is not None:
            return cached

        conf = self.configs_by_namespace.get(namespace)
        if conf is None:
            logger.error(
                "firebase namespace sem configuracao namespace=%s; escrita descartada",
                namespace,
            )
            return self._disabled_sink

        with self._lock:
            cached = self._sinks.get(namespace)
            if cached is not None:
                return cached

            app_name = f"stt-{safe_key(namespace)}"
            try:
                app_instance = firebase_admin.get_app(app_name)
            except ValueError:
                app_kwargs: dict[str, str] = {}
                if conf.project_id:
                    app_kwargs["projectId"] = conf.project_id
                if conf.storage_bucket:
                    app_kwargs["storageBucket"] = conf.storage_bucket
                cred = credentials.Certificate(conf.credentials_file)
                app_instance = firebase_admin.initialize_app(
                    cred,
                    app_kwargs or None,
                    name=app_name,
                )

            fs_client = firestore.client(app=app_instance)
            bucket = None
            if conf.storage_bucket:
                bucket = storage.bucket(conf.storage_bucket, app=app_instance)

            sink = FirebaseSink(
                namespace=namespace,
                enabled=True,
                firestore_client=fs_client,
                storage_bucket=bucket,
            )
            self._sinks[namespace] = sink
            return sink

    def sink_for_room(self, room_name: str) -> FirebaseSink:
        info = extract_room_namespace(room_name)
        if info is None:
            logger.error("nao foi possivel resolver namespace para room=%s", room_name)
            return self._disabled_sink
        return self._get_or_create_sink(info.namespace)

    def publish_call_index(
        self,
        room_name: str,
        session_id: str,
        status: str,
        last_minute_index: int,
        finalized: bool,
        final_summary_path: str | None = None,
        summary_accumulated_path: str | None = None,
    ) -> None:
        info = extract_room_namespace(room_name)
        room_id = info.room_id if info is not None else room_name
        self.sink_for_room(room_name).publish_call_index(
            room_name,
            session_id,
            room_id,
            status,
            last_minute_index,
            finalized,
            STORAGE_MINUTE_WINDOW_SECONDS,
            FIREBASE_FLUSH_INTERVAL_SECONDS,
            final_summary_path=final_summary_path,
            summary_accumulated_path=summary_accumulated_path,
        )

    def upsert_minute_shard(self, room_name: str, payload: MinuteShardPayload) -> None:
        self.sink_for_room(room_name).upsert_minute_shard(payload)

    def update_minute_summary_link(
        self, room_name: str, session_id: str, minute_index: int, summary_json_path: str
    ) -> None:
        call_key = f"{room_name}__{session_id}"
        self.sink_for_room(room_name).update_minute_summary_link(
            call_key, minute_index, summary_json_path
        )

    def upload_minute_transcript(self, room_name: str, payload: MinuteShardPayload) -> None:
        self.sink_for_room(room_name).upload_minute_transcript(payload)

    def upload_json(self, room_name: str, object_path: str, body: dict) -> None:
        self.sink_for_room(room_name).upload_json(object_path, body)

    def fetch_json(self, room_name: str, object_path: str) -> dict | None:
        return self.sink_for_room(room_name).fetch_json(object_path)


def db_pending_count() -> int:
    conn = read_db_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE status IN ('queued','processing')"
        ).fetchone()
        return int(row["total"] if row else 0)
    finally:
        conn.close()


def db_upsert_start(req: StartRequest) -> None:
    now = utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO sessions(
                room_name, session_id, started_at, metadata_json,
                state, room_end_received, last_chunk_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', 0, ?, ?, ?)
            ON CONFLICT(room_name, session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                metadata_json=COALESCE(excluded.metadata_json, sessions.metadata_json),
                last_chunk_at=COALESCE(sessions.last_chunk_at, excluded.last_chunk_at)
            """,
            (
                req.room_name,
                req.session_id,
                req.started_at,
                json.dumps(req.metadata, ensure_ascii=False) if req.metadata is not None else None,
                req.started_at,
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO participants(
                room_name, session_id, participant_identity, participant_name, track_sid,
                started_at, state, last_seq, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'active', 0, ?, ?)
            ON CONFLICT(room_name, session_id, participant_identity) DO UPDATE SET
                participant_name=COALESCE(excluded.participant_name, participants.participant_name),
                track_sid=COALESCE(excluded.track_sid, participants.track_sid),
                updated_at=excluded.updated_at
            """,
            (
                req.room_name,
                req.session_id,
                req.participant_identity,
                req.participant_name,
                req.track_sid,
                req.started_at,
                now,
                now,
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def write_spool_file(meta: ChunkMeta, audio_bytes: bytes) -> Path:
    call_dir = SPOOL_DIR / safe_key(f"{meta.room_name}__{meta.session_id}") / safe_key(
        meta.participant_identity
    )
    call_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{meta.seq:08d}-{uuid.uuid4().hex}.pcm"
    path = call_dir / filename
    path.write_bytes(audio_bytes)
    return path


def db_enqueue_chunk(meta: ChunkMeta, spool_path: Path) -> str:
    now = utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        pending = conn.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE status IN ('queued','processing')"
        ).fetchone()
        if int(pending["total"]) >= QUEUE_MAX_PENDING:
            conn.execute("ROLLBACK")
            return "queue_full"

        session = conn.execute(
            "SELECT 1 FROM sessions WHERE room_name = ? AND session_id = ?",
            (meta.room_name, meta.session_id),
        ).fetchone()
        if not session:
            conn.execute("ROLLBACK")
            return "session_not_found"

        participant = conn.execute(
            """
            SELECT last_seq FROM participants
            WHERE room_name = ? AND session_id = ? AND participant_identity = ?
            """,
            (meta.room_name, meta.session_id, meta.participant_identity),
        ).fetchone()
        if not participant:
            conn.execute(
                """
                INSERT INTO participants(
                    room_name, session_id, participant_identity, participant_name, track_sid,
                    started_at, state, last_seq, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', 0, ?, ?)
                """,
                (
                    meta.room_name,
                    meta.session_id,
                    meta.participant_identity,
                    meta.participant_name,
                    meta.track_sid,
                    meta.chunk_started_at,
                    now,
                    now,
                ),
            )
            last_seq = 0
        else:
            last_seq = int(participant["last_seq"])

        if meta.seq <= last_seq:
            conn.execute("ROLLBACK")
            return "sequence_conflict"

        try:
            conn.execute(
                """
                INSERT INTO chunks(
                    room_name, session_id, participant_identity, seq, track_sid,
                    chunk_started_at, chunk_ended_at, sample_rate, channels, encoding,
                    spool_path, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    meta.room_name,
                    meta.session_id,
                    meta.participant_identity,
                    meta.seq,
                    meta.track_sid,
                    meta.chunk_started_at,
                    meta.chunk_ended_at,
                    meta.sample_rate,
                    meta.channels,
                    meta.encoding,
                    str(spool_path),
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            conn.execute("ROLLBACK")
            return "sequence_conflict"

        conn.execute(
            """
            UPDATE participants
            SET last_seq = ?, participant_name = COALESCE(?, participant_name),
                track_sid = COALESCE(?, track_sid), updated_at = ?
            WHERE room_name = ? AND session_id = ? AND participant_identity = ?
            """,
            (
                meta.seq,
                meta.participant_name,
                meta.track_sid,
                now,
                meta.room_name,
                meta.session_id,
                meta.participant_identity,
            ),
        )
        conn.execute(
            """
            UPDATE sessions
            SET last_chunk_at = ?, updated_at = ?
            WHERE room_name = ? AND session_id = ?
            """,
            (now, now, meta.room_name, meta.session_id),
        )
        conn.execute("COMMIT")
        return "accepted"
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_participant_end(req: EndRequest) -> None:
    now = req.ended_at or utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE participants
            SET state='ended', ended_at=COALESCE(ended_at, ?), updated_at=?
            WHERE room_name = ? AND session_id = ? AND participant_identity = ?
            """,
            (now, utc_now_iso(), req.room_name, req.session_id, req.participant_identity),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_room_end(req: EndRequest) -> None:
    now = req.ended_at or utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE sessions
            SET room_end_received=1, state='room_ended',
                ended_at=COALESCE(ended_at, ?), updated_at=?
            WHERE room_name = ? AND session_id = ?
            """,
            (now, utc_now_iso(), req.room_name, req.session_id),
        )
        conn.execute(
            """
            UPDATE participants
            SET state='ended', ended_at=COALESCE(ended_at, ?), updated_at=?
            WHERE room_name = ? AND session_id = ? AND state='active'
            """,
            (now, utc_now_iso(), req.room_name, req.session_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_claim_next_chunk() -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM chunks
            WHERE status='queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            conn.execute("COMMIT")
            return None

        conn.execute(
            """
            UPDATE chunks
            SET status='processing', updated_at=?
            WHERE room_name = ? AND session_id = ? AND participant_identity = ? AND seq = ?
            """,
            (
                utc_now_iso(),
                row["room_name"],
                row["session_id"],
                row["participant_identity"],
                row["seq"],
            ),
        )
        conn.execute("COMMIT")
        return row
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_chunk_done(chunk_row: sqlite3.Row, text: str) -> None:
    now = utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE chunks
            SET status='done', transcript=?, processed_at=?, updated_at=?
            WHERE room_name = ? AND session_id = ? AND participant_identity = ? AND seq = ?
            """,
            (
                text,
                now,
                now,
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
                chunk_row["seq"],
            ),
        )
        conn.execute(
            """
            INSERT INTO transcripts(room_name, session_id, participant_identity, seq, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_name, session_id, participant_identity, seq) DO UPDATE SET text=excluded.text
            """,
            (
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
                chunk_row["seq"],
                text,
                now,
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_chunk_error(chunk_row: sqlite3.Row, error_message: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE chunks
            SET status='error', error_message=?, updated_at=?
            WHERE room_name = ? AND session_id = ? AND participant_identity = ? AND seq = ?
            """,
            (
                error_message[:1000],
                utc_now_iso(),
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
                chunk_row["seq"],
            ),
        )
    finally:
        conn.close()


def db_get_participant_aggregate(
    room_name: str, session_id: str, participant_identity: str
) -> tuple[str, list[dict]]:
    conn = read_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT seq, transcript, chunk_started_at, chunk_ended_at
            FROM chunks
            WHERE room_name=? AND session_id=? AND participant_identity=? AND status='done'
            ORDER BY seq ASC
            """,
            (room_name, session_id, participant_identity),
        ).fetchall()
        segments = [
            {
                "seq": int(row["seq"]),
                "text": row["transcript"] or "",
                "chunk_started_at": row["chunk_started_at"],
                "chunk_ended_at": row["chunk_ended_at"],
            }
            for row in rows
        ]
        transcript = "\n".join(segment["text"] for segment in segments if segment["text"])
        return transcript, segments
    finally:
        conn.close()


def db_get_room_aggregate(room_name: str, session_id: str) -> tuple[str, list[dict]]:
    conn = read_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                c.participant_identity,
                p.participant_name,
                c.seq,
                c.transcript,
                c.chunk_started_at,
                c.chunk_ended_at
            FROM chunks c
            LEFT JOIN participants p
              ON p.room_name = c.room_name
             AND p.session_id = c.session_id
             AND p.participant_identity = c.participant_identity
            WHERE c.room_name=? AND c.session_id=? AND c.status='done'
            ORDER BY c.chunk_started_at ASC, c.participant_identity ASC, c.seq ASC
            """,
            (room_name, session_id),
        ).fetchall()
        lines = []
        for row in rows:
            text = (row["transcript"] or "").strip()
            if not text:
                continue
            participant_name = (row["participant_name"] or "").strip()
            display_name = participant_name or row["participant_identity"]
            lines.append(
                {
                    "participant_identity": row["participant_identity"],
                    "participant_name": participant_name or None,
                    "speaker": display_name,
                    "seq": int(row["seq"]),
                    "text": text,
                    "chunk_started_at": row["chunk_started_at"],
                    "chunk_ended_at": row["chunk_ended_at"],
                }
            )

        transcript = "\n".join(f"[{line['speaker']}] {line['text']}" for line in lines)
        return transcript, lines
    finally:
        conn.close()


def compute_minute_index(
    session_started_at: str, chunk_ended_at: str, minute_window_seconds: int
) -> int:
    session_start = parse_iso_datetime(session_started_at)
    chunk_end = parse_iso_datetime(chunk_ended_at)
    delta_seconds = max(0.0, (chunk_end - session_start).total_seconds())
    return int(delta_seconds // minute_window_seconds)


def db_get_sessions_due_for_flush(now_iso: str, interval_seconds: int) -> list[sqlite3.Row]:
    cutoff_iso = (parse_iso_datetime(now_iso) - timedelta(seconds=interval_seconds)).isoformat()
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT room_name, session_id, started_at, room_end_received
            FROM sessions
            WHERE finalized_at IS NULL
              AND (
                last_firebase_flush_at IS NULL
                OR last_firebase_flush_at <= ?
              )
            """,
            (cutoff_iso,),
        ).fetchall()
    finally:
        conn.close()


def db_mark_session_flushed(room_name: str, session_id: str, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE sessions
            SET last_firebase_flush_at=?, updated_at=?
            WHERE room_name=? AND session_id=?
            """,
            (now_iso, now_iso, room_name, session_id),
        )
    finally:
        conn.close()


def db_get_session_row(room_name: str, session_id: str) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT room_name, session_id, started_at, room_end_received, finalized_at
            FROM sessions
            WHERE room_name=? AND session_id=?
            """,
            (room_name, session_id),
        ).fetchone()
    finally:
        conn.close()


def db_get_done_chunks_for_session(room_name: str, session_id: str) -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT
                c.participant_identity,
                p.participant_name,
                c.seq,
                c.transcript,
                c.chunk_started_at,
                c.chunk_ended_at
            FROM chunks c
            LEFT JOIN participants p
              ON p.room_name = c.room_name
             AND p.session_id = c.session_id
             AND p.participant_identity = c.participant_identity
            WHERE c.room_name=? AND c.session_id=? AND c.status='done'
            ORDER BY c.chunk_ended_at ASC, c.participant_identity ASC, c.seq ASC
            """,
            (room_name, session_id),
        ).fetchall()
    finally:
        conn.close()


def build_minute_shards(
    room_name: str,
    session_id: str,
    started_at: str,
    done_chunks: list[sqlite3.Row],
    minute_window_seconds: int,
    finalized: bool,
) -> list[MinuteShardPayload]:
    call_key = f"{room_name}__{session_id}"
    grouped: dict[int, list[dict]] = {}
    for row in done_chunks:
        text = (row["transcript"] or "").strip()
        if not text:
            continue
        minute_index = compute_minute_index(
            started_at, row["chunk_ended_at"], minute_window_seconds
        )
        participant_name = (row["participant_name"] or "").strip()
        display_name = participant_name or row["participant_identity"]
        grouped.setdefault(minute_index, []).append(
            {
                "speaker": display_name,
                "participant_identity": row["participant_identity"],
                "participant_name": participant_name or None,
                "seq": int(row["seq"]),
                "text": text,
                "chunk_started_at": row["chunk_started_at"],
                "chunk_ended_at": row["chunk_ended_at"],
            }
        )

    shards: list[MinuteShardPayload] = []
    for minute_index in sorted(grouped):
        minute_start_dt = parse_iso_datetime(started_at) + timedelta(
            seconds=minute_index * minute_window_seconds
        )
        minute_end_dt = minute_start_dt + timedelta(seconds=minute_window_seconds)
        call_base = f"calls/{safe_key(call_key)}/minutes/{minute_index:04d}"
        transcript_path = f"{call_base}/transcript.json"
        summary_path = f"{call_base}/summary.json"
        minute_lines = grouped[minute_index]
        raw_hash = hashlib.sha256(
            json.dumps(minute_lines, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        shards.append(
            MinuteShardPayload(
                room_name=room_name,
                session_id=session_id,
                call_key=call_key,
                minute_index=minute_index,
                minute_started_at=minute_start_dt.isoformat(),
                minute_ended_at=minute_end_dt.isoformat(),
                transcript_json_path=transcript_path,
                summary_json_path=summary_path,
                lines=minute_lines,
                line_count=len(minute_lines),
                transcript_hash=raw_hash,
                finalized=finalized,
            )
        )
    return shards


def db_get_minute_export(
    room_name: str, session_id: str, minute_index: int
) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT *
            FROM minute_exports
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (room_name, session_id, minute_index),
        ).fetchone()
    finally:
        conn.close()


def db_upsert_minute_export(payload: MinuteShardPayload, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO minute_exports(
                room_name, session_id, minute_index,
                transcript_json_path, summary_json_path, content_hash,
                minute_started_at, minute_ended_at, finalized,
                exported_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_name, session_id, minute_index) DO UPDATE SET
                transcript_json_path=excluded.transcript_json_path,
                summary_json_path=excluded.summary_json_path,
                content_hash=excluded.content_hash,
                minute_started_at=excluded.minute_started_at,
                minute_ended_at=excluded.minute_ended_at,
                finalized=excluded.finalized,
                exported_at=excluded.exported_at,
                updated_at=excluded.updated_at
            """,
            (
                payload.room_name,
                payload.session_id,
                payload.minute_index,
                payload.transcript_json_path,
                payload.summary_json_path,
                payload.transcript_hash,
                payload.minute_started_at,
                payload.minute_ended_at,
                1 if payload.finalized else 0,
                now_iso,
                now_iso,
            ),
        )
    finally:
        conn.close()


def db_upsert_summary_task(room_name: str, session_id: str, minute_index: int, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO summary_tasks(
                room_name, session_id, minute_index,
                status, retries, next_attempt_at,
                error_message, created_at, updated_at
            ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, ?, ?)
            ON CONFLICT(room_name, session_id, minute_index) DO UPDATE SET
                status='pending',
                next_attempt_at=excluded.next_attempt_at,
                error_message=NULL,
                updated_at=excluded.updated_at
            """,
            (room_name, session_id, minute_index, now_iso, now_iso, now_iso),
        )
    finally:
        conn.close()


def db_claim_summary_task(now_iso: str) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            """
            SELECT room_name, session_id, minute_index, retries
            FROM summary_tasks
            WHERE status IN ('pending', 'error')
              AND next_attempt_at <= ?
              AND retries < ?
            ORDER BY updated_at ASC
            LIMIT 1
            """,
            (now_iso, OPENAI_MAX_RETRIES),
        ).fetchone()
        if task is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='processing', updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (now_iso, task["room_name"], task["session_id"], task["minute_index"]),
        )
        conn.execute("COMMIT")
        return task
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_summary_task_done(room_name: str, session_id: str, minute_index: int, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='done', updated_at=?, error_message=NULL
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (now_iso, room_name, session_id, minute_index),
        )
    finally:
        conn.close()


def db_mark_summary_task_error(
    room_name: str,
    session_id: str,
    minute_index: int,
    retries: int,
    error_message: str,
    now_iso: str,
) -> None:
    next_attempt_iso = (parse_iso_datetime(now_iso) + timedelta(seconds=15 * retries)).isoformat()
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='error',
                retries=?,
                error_message=?,
                next_attempt_at=?,
                updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (
                retries,
                error_message[:1000],
                next_attempt_iso,
                now_iso,
                room_name,
                session_id,
                minute_index,
            ),
        )
    finally:
        conn.close()


def db_get_session_minute_exports(room_name: str, session_id: str) -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT *
            FROM minute_exports
            WHERE room_name=? AND session_id=?
            ORDER BY minute_index ASC
            """,
            (room_name, session_id),
        ).fetchall()
    finally:
        conn.close()


def db_update_minute_export_summary_path(
    room_name: str, session_id: str, minute_index: int, summary_json_path: str, now_iso: str
) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE minute_exports
            SET summary_json_path=?, updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (summary_json_path, now_iso, room_name, session_id, minute_index),
        )
    finally:
        conn.close()


def db_get_finalizable_participants() -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.*
            FROM participants p
            WHERE p.state='ended'
              AND p.finalized_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM chunks c
                  WHERE c.room_name=p.room_name
                    AND c.session_id=p.session_id
                    AND c.participant_identity=p.participant_identity
                    AND c.status IN ('queued', 'processing')
              )
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def db_mark_participant_finalized(
    room_name: str, session_id: str, participant_identity: str
) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE participants
            SET state='finalized', finalized_at=?, updated_at=?
            WHERE room_name=? AND session_id=? AND participant_identity=?
            """,
            (utc_now_iso(), utc_now_iso(), room_name, session_id, participant_identity),
        )
    finally:
        conn.close()


def db_get_finalizable_rooms() -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT s.*
            FROM sessions s
            WHERE s.room_end_received=1
              AND s.finalized_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM chunks c
                  WHERE c.room_name=s.room_name
                    AND c.session_id=s.session_id
                    AND c.status IN ('queued', 'processing')
              )
              AND NOT EXISTS (
                  SELECT 1 FROM participants p
                  WHERE p.room_name=s.room_name
                    AND p.session_id=s.session_id
                    AND p.finalized_at IS NULL
              )
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def db_mark_room_finalized(room_name: str, session_id: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE sessions
            SET state='finalized', finalized_at=?, updated_at=?
            WHERE room_name=? AND session_id=?
            """,
            (utc_now_iso(), utc_now_iso(), room_name, session_id),
        )
    finally:
        conn.close()


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def db_expire_inactive_rooms(now_iso: str) -> list[dict]:
    now_dt = parse_iso_datetime(now_iso)
    cutoff_dt = now_dt - timedelta(seconds=ROOM_INACTIVITY_TIMEOUT_SECONDS)

    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        candidates = conn.execute(
            """
            SELECT room_name, session_id, last_chunk_at, started_at
            FROM sessions
            WHERE room_end_received = 0
              AND finalized_at IS NULL
              AND state IN ('active', 'room_ended')
            """
        ).fetchall()

        expired = []
        for row in candidates:
            raw_last_chunk_at = row["last_chunk_at"] or row["started_at"]
            if not raw_last_chunk_at:
                continue
            try:
                last_chunk_at_dt = parse_iso_datetime(raw_last_chunk_at)
            except ValueError:
                logger.warning(
                    "invalid datetime in sessions row room=%s session=%s last_chunk_at=%s",
                    row["room_name"],
                    row["session_id"],
                    raw_last_chunk_at,
                )
                continue
            if last_chunk_at_dt <= cutoff_dt:
                expired.append(
                    {
                        "room_name": row["room_name"],
                        "session_id": row["session_id"],
                        "last_chunk_at": raw_last_chunk_at,
                    }
                )

        for row in expired:
            conn.execute(
                """
                UPDATE sessions
                SET room_end_received=1, state='room_ended',
                    ended_at=COALESCE(ended_at, ?), updated_at=?
                WHERE room_name = ? AND session_id = ?
                  AND room_end_received = 0
                  AND finalized_at IS NULL
                """,
                (
                    now_iso,
                    now_iso,
                    row["room_name"],
                    row["session_id"],
                ),
            )
            conn.execute(
                """
                UPDATE participants
                SET state='ended', ended_at=COALESCE(ended_at, ?), updated_at=?
                WHERE room_name = ? AND session_id = ? AND state='active'
                """,
                (
                    now_iso,
                    now_iso,
                    row["room_name"],
                    row["session_id"],
                ),
            )
        conn.execute("COMMIT")
        return expired
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def decode_pcm_file(recognizer: sherpa_onnx.OfflineRecognizer, path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    samples_int16 = np.frombuffer(data, dtype=np.int16)
    if samples_int16.size == 0:
        return ""
    samples = samples_int16.astype(np.float32) / 32768.0
    return decode_with_tail_padding(recognizer, samples)


def load_openai_api_key() -> str | None:
    if not OPENAI_SUMMARY_ENABLED:
        return None
    if not OPENAI_APIKEY_FILE.is_file():
        logger.warning(
            "OPENAI_SUMMARY_ENABLED=true, mas arquivo de secret nao encontrado: %s",
            OPENAI_APIKEY_FILE,
        )
        return None
    try:
        raw = json.loads(OPENAI_APIKEY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("arquivo OPENAI_APIKEY_FILE invalido: JSON malformado")
        return None
    api_key = str(raw.get("api_key", "")).strip() if isinstance(raw, dict) else ""
    if not api_key:
        logger.warning("arquivo OPENAI_APIKEY_FILE sem campo api_key valido")
        return None
    return api_key


def extract_openai_output_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks).strip()


@dataclass(frozen=True)
class SummaryEngine:
    enabled: bool
    api_key: str
    minute_model: str
    accumulated_model: str
    final_model: str
    timeout_seconds: int

    @staticmethod
    def create() -> "SummaryEngine":
        api_key = load_openai_api_key()
        if not api_key:
            return SummaryEngine(
                False,
                "",
                OPENAI_MODEL_MINUTE_SUMMARY,
                OPENAI_MODEL_ACCUMULATED_SUMMARY,
                OPENAI_MODEL_FINAL_SUMMARY,
                OPENAI_REQUEST_TIMEOUT_SECONDS,
            )
        return SummaryEngine(
            True,
            api_key,
            OPENAI_MODEL_MINUTE_SUMMARY,
            OPENAI_MODEL_ACCUMULATED_SUMMARY,
            OPENAI_MODEL_FINAL_SUMMARY,
            OPENAI_REQUEST_TIMEOUT_SECONDS,
        )

    def _request_text(self, model: str, system_prompt: str, user_prompt: str) -> str:
        if not self.enabled:
            raise RuntimeError("summary engine disabled")
        request_body = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(request_body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
                detail = body[:1200]
            except Exception:  # pylint: disable=broad-except
                detail = ""
            raise RuntimeError(
                f"OpenAI request failed status={exc.code} model={model} detail={detail}"
            ) from exc
        text = extract_openai_output_text(payload)
        if not text:
            raise RuntimeError("OpenAI retornou resposta sem texto")
        return text

    def summarize_minute(self, minute_lines: list[dict]) -> str:
        minute_text = "\n".join(f"[{line['speaker']}] {line['text']}" for line in minute_lines)
        return self._request_text(
            self.minute_model,
            (
                "Voce e um assistente especializado em resumir trechos de chamadas corporativas em portugues do Brasil. "
                "Analise APENAS as falas deste minuto e gere um resumo factual, claro e objetivo. "
                "Regras: use somente o conteudo fornecido; nao invente contexto; nao transforme sugestoes em decisoes; "
                "nao trate duvidas como conclusoes; preserve nomes de pessoas, sistemas, produtos e funcionalidades; "
                "se houver ambiguidade, registre com cautela. "
                "Formato da resposta:\n"
                "Resumo do minuto:\n"
                "- 1 a 5 frases objetivas\n\n"
                "Decisoes identificadas:\n"
                "- liste apenas decisoes explicitas; se nao houver, escreva 'Nenhuma decisao explicita.'\n\n"
                "Pendencias ou duvidas:\n"
                "- liste apenas o que realmente estiver em aberto; se nao houver, escreva 'Nenhuma pendencia explicita.'"
            ),
            f"Minuto da chamada:\n{minute_text}",
        )

    def merge_summaries(self, previous_summary: str, minute_summary: str) -> str:
        return self._request_text(
            self.accumulated_model,
            (
                "Voce e um assistente especializado em consolidar resumos acumulados de chamadas corporativas. "
                "Recebera um resumo acumulado anterior e um novo resumo de minuto. "
                "Sua tarefa e produzir um novo resumo acumulado, preservando fatos importantes e incorporando apenas novidades reais. "
                "Regras: mantenha decisoes ja registradas, exceto se o novo conteudo as contradizer explicitamente; "
                "nao invente contexto; nao transforme sugestoes em decisoes; elimine repeticoes; "
                "mantenha pendencias ainda abertas; preserve nomes e termos tecnicos exatamente como informados. "
                "Formato da resposta:\n"
                "Resumo acumulado:\n"
                "- 1 a 3 paragrafos coesos\n\n"
                "Decisoes confirmadas:\n"
                "- liste apenas decisoes confirmadas; se nao houver, escreva 'Nenhuma decisao confirmada ate o momento.'\n\n"
                "Pendencias em aberto:\n"
                "- liste pendencias ainda abertas; se nao houver, escreva 'Nenhuma pendencia em aberto identificada.'"
            ),
            "Resumo acumulado atual:\n"
            f"{previous_summary or '(vazio)'}\n\n"
            f"Resumo novo do minuto:\n{minute_summary}",
        )

    def finalize_summary(self, merged_summary: str) -> str:
        return self._request_text(
            self.final_model,
            (
                "Voce e um assistente especializado em produzir resumos executivos de chamadas corporativas em portugues do Brasil. "
                "Com base apenas no resumo acumulado fornecido, gere um resumo final claro, objetivo e util para acompanhamento posterior. "
                "Regras: nao invente decisoes, responsaveis ou prazos; diferencie fatos discutidos, decisoes tomadas e pendencias; "
                "nao use linguagem vaga; preserve nomes e termos tecnicos. "
                "Use exatamente esta estrutura:\n\n"
                "Resumo Final Executivo da Chamada\n\n"
                "Principais Pontos:\n"
                "- ...\n\n"
                "Decisoes:\n"
                "- ...\n\n"
                "Pendencias:\n"
                "- ...\n\n"
                "Proximos Passos:\n"
                "- ..."
            ),
            f"Resumo acumulado da chamada:\n{merged_summary}",
        )

def session_summary_accumulated_path(room_name: str, session_id: str) -> str:
    call_key = f"{room_name}__{session_id}"
    return f"calls/{safe_key(call_key)}/summary/accumulated.json"


def session_final_summary_path(room_name: str, session_id: str) -> str:
    call_key = f"{room_name}__{session_id}"
    return f"calls/{safe_key(call_key)}/final/final_summary.json"


def publish_session_minute_exports(
    firebase_router: FirebaseRouter,
    room_name: str,
    session_id: str,
    now_iso: str,
    finalized: bool,
    summary_enabled: bool,
) -> int:
    session_row = db_get_session_row(room_name, session_id)
    if session_row is None:
        return -1
    done_chunks = db_get_done_chunks_for_session(room_name, session_id)
    shards = build_minute_shards(
        room_name=room_name,
        session_id=session_id,
        started_at=session_row["started_at"],
        done_chunks=done_chunks,
        minute_window_seconds=STORAGE_MINUTE_WINDOW_SECONDS,
        finalized=finalized,
    )
    last_minute_index = shards[-1].minute_index if shards else -1
    for shard in shards:
        previous = db_get_minute_export(room_name, session_id, shard.minute_index)
        should_upload = (
            previous is None
            or previous["content_hash"] != shard.transcript_hash
            or (finalized and int(previous["finalized"] or 0) == 0)
        )
        if should_upload:
            firebase_router.upload_minute_transcript(room_name, shard)
        if summary_enabled and (
            should_upload or previous is None or not previous["summary_json_path"]
        ):
            db_upsert_summary_task(room_name, session_id, shard.minute_index, now_iso)
        db_upsert_minute_export(shard, now_iso)
        firebase_router.upsert_minute_shard(room_name, shard)

    firebase_router.publish_call_index(
        room_name=room_name,
        session_id=session_id,
        status="finalized" if finalized else "processing",
        last_minute_index=last_minute_index,
        finalized=finalized,
        final_summary_path=(
            session_final_summary_path(room_name, session_id)
            if finalized and summary_enabled
            else None
        ),
        summary_accumulated_path=session_summary_accumulated_path(room_name, session_id),
    )
    return last_minute_index


async def flush_due_sessions(firebase_router: FirebaseRouter, summary_enabled: bool) -> None:
    now_iso = utc_now_iso()
    sessions = await asyncio.to_thread(
        db_get_sessions_due_for_flush, now_iso, FIREBASE_FLUSH_INTERVAL_SECONDS
    )
    for session_row in sessions:
        room_name = session_row["room_name"]
        session_id = session_row["session_id"]
        try:
            await asyncio.to_thread(
                publish_session_minute_exports,
                firebase_router,
                room_name,
                session_id,
                now_iso,
                False,
                summary_enabled,
            )
            await asyncio.to_thread(db_mark_session_flushed, room_name, session_id, now_iso)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "flush session failed room=%s session=%s error=%s",
                room_name,
                session_id,
                exc,
            )


async def finalize_entities(firebase_router: FirebaseRouter, summary_engine: SummaryEngine) -> None:
    participants = db_get_finalizable_participants()
    for participant in participants:
        room_name = participant["room_name"]
        session_id = participant["session_id"]
        participant_identity = participant["participant_identity"]
        db_mark_participant_finalized(room_name, session_id, participant_identity)
        logger.info(
            "participant finalized room=%s session=%s participant=%s",
            room_name,
            session_id,
            participant_identity,
        )

    rooms = db_get_finalizable_rooms()
    for room in rooms:
        room_name = room["room_name"]
        session_id = room["session_id"]
        now_iso = utc_now_iso()
        await asyncio.to_thread(
            publish_session_minute_exports,
            firebase_router,
            room_name,
            session_id,
            now_iso,
            True,
            summary_engine.enabled,
        )
        if summary_engine.enabled:
            await asyncio.to_thread(db_upsert_summary_task, room_name, session_id, -1, now_iso)
        db_mark_room_finalized(room_name, session_id)
        logger.info("room finalized room=%s session=%s", room_name, session_id)


async def summary_worker_loop(
    stop_event: asyncio.Event,
    firebase_router: FirebaseRouter,
    summary_engine: SummaryEngine,
) -> None:
    logger.info("summary worker started enabled=%s", summary_engine.enabled)
    while not stop_event.is_set():
        if not summary_engine.enabled:
            await asyncio.sleep(1.0)
            continue

        now_iso = utc_now_iso()
        task = await asyncio.to_thread(db_claim_summary_task, now_iso)
        if task is None:
            await asyncio.sleep(0.5)
            continue

        room_name = task["room_name"]
        session_id = task["session_id"]
        minute_index = int(task["minute_index"])
        retries = int(task["retries"])
        try:
            call_key = f"{room_name}__{session_id}"
            if minute_index >= 0:
                export_row = await asyncio.to_thread(
                    db_get_minute_export, room_name, session_id, minute_index
                )
                if export_row is None:
                    await asyncio.to_thread(
                        db_mark_summary_task_done, room_name, session_id, minute_index, now_iso
                    )
                    continue
                minute_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, room_name, export_row["transcript_json_path"]
                )
                lines = minute_payload.get("lines", []) if isinstance(minute_payload, dict) else []
                minute_summary = await asyncio.to_thread(summary_engine.summarize_minute, lines)
                summary_path = export_row["summary_json_path"] or (
                    f"calls/{safe_key(call_key)}/minutes/{minute_index:04d}/summary.json"
                )
                await asyncio.to_thread(
                    firebase_router.upload_json,
                    room_name,
                    summary_path,
                    {
                        "room_name": room_name,
                        "session_id": session_id,
                        "minute_index": minute_index,
                        "summary": minute_summary,
                        "updated_at": now_iso,
                    },
                )
                await asyncio.to_thread(
                    db_update_minute_export_summary_path,
                    room_name,
                    session_id,
                    minute_index,
                    summary_path,
                    now_iso,
                )
                await asyncio.to_thread(
                    firebase_router.update_minute_summary_link,
                    room_name,
                    session_id,
                    minute_index,
                    summary_path,
                )

                accumulated_path = session_summary_accumulated_path(room_name, session_id)
                previous_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, room_name, accumulated_path
                )
                previous_summary = ""
                if isinstance(previous_payload, dict):
                    previous_summary = str(previous_payload.get("summary", "")).strip()
                merged_summary = await asyncio.to_thread(
                    summary_engine.merge_summaries, previous_summary, minute_summary
                )
                await asyncio.to_thread(
                    firebase_router.upload_json,
                    room_name,
                    accumulated_path,
                    {
                        "room_name": room_name,
                        "session_id": session_id,
                        "last_minute_index": minute_index,
                        "summary": merged_summary,
                        "updated_at": now_iso,
                    },
                )
                await asyncio.to_thread(
                    firebase_router.publish_call_index,
                    room_name,
                    session_id,
                    "processing",
                    minute_index,
                    False,
                    None,
                    accumulated_path,
                )
            else:
                exports = await asyncio.to_thread(db_get_session_minute_exports, room_name, session_id)
                summary_parts: list[str] = []
                for export_row in exports:
                    summary_path = export_row["summary_json_path"]
                    if not summary_path:
                        continue
                    payload = await asyncio.to_thread(
                        firebase_router.fetch_json, room_name, summary_path
                    )
                    if isinstance(payload, dict):
                        text = str(payload.get("summary", "")).strip()
                        if text:
                            summary_parts.append(text)
                merged = "\n".join(summary_parts).strip()
                if merged:
                    final_summary = await asyncio.to_thread(summary_engine.finalize_summary, merged)
                    final_path = session_final_summary_path(room_name, session_id)
                    await asyncio.to_thread(
                        firebase_router.upload_json,
                        room_name,
                        final_path,
                        {
                            "room_name": room_name,
                            "session_id": session_id,
                            "summary": final_summary,
                            "updated_at": now_iso,
                        },
                    )
                    accumulated_path = session_summary_accumulated_path(room_name, session_id)
                    await asyncio.to_thread(
                        firebase_router.publish_call_index,
                        room_name,
                        session_id,
                        "finalized",
                        exports[-1]["minute_index"] if exports else -1,
                        True,
                        final_path,
                        accumulated_path,
                    )

            await asyncio.to_thread(
                db_mark_summary_task_done, room_name, session_id, minute_index, utc_now_iso()
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "summary task failed room=%s session=%s minute=%s error=%s",
                room_name,
                session_id,
                minute_index,
                exc,
            )
            await asyncio.to_thread(
                db_mark_summary_task_error,
                room_name,
                session_id,
                minute_index,
                retries + 1,
                str(exc),
                utc_now_iso(),
            )
    logger.info("summary worker stopped")


async def worker_loop(
    stop_event: asyncio.Event,
    recognizer: sherpa_onnx.OfflineRecognizer,
    firebase_router: FirebaseRouter,
    summary_engine: SummaryEngine,
) -> None:
    logger.info("worker started")
    while not stop_event.is_set():
        chunk_row = await asyncio.to_thread(db_claim_next_chunk)
        if not chunk_row:
            now_iso = utc_now_iso()
            expired_rooms = await asyncio.to_thread(db_expire_inactive_rooms, now_iso)
            for room in expired_rooms:
                logger.info(
                    "room timeout room=%s session=%s last_chunk_at=%s timeout_seconds=%s reason=inactivity_timeout",
                    room["room_name"],
                    room["session_id"],
                    room["last_chunk_at"],
                    ROOM_INACTIVITY_TIMEOUT_SECONDS,
                )
            await finalize_entities(firebase_router, summary_engine)
            await flush_due_sessions(firebase_router, summary_engine.enabled)
            await asyncio.sleep(WORKER_POLL_SECONDS)
            continue

        spool_path = Path(chunk_row["spool_path"])
        try:
            text = await asyncio.to_thread(decode_pcm_file, recognizer, spool_path)
            await asyncio.to_thread(db_mark_chunk_done, chunk_row, text)

            logger.info(
                "chunk processed room=%s session=%s participant=%s seq=%s text_len=%s",
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
                chunk_row["seq"],
                len(text),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("chunk processing failed: %s", exc)
            await asyncio.to_thread(db_mark_chunk_error, chunk_row, str(exc))
        finally:
            try:
                spool_path.unlink(missing_ok=True)
            except Exception:  # pylint: disable=broad-except
                pass
        await flush_due_sessions(firebase_router, summary_engine.enabled)

    logger.info("worker stopped")


@dataclass
class AppState:
    recognizer: sherpa_onnx.OfflineRecognizer
    hmac_keys: dict[str, str]
    firebase_router: FirebaseRouter
    summary_engine: SummaryEngine
    worker_task: asyncio.Task | None
    summary_worker_task: asyncio.Task | None
    worker_stop_event: asyncio.Event


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recognizer = build_offline_recognizer()
    hmac_keys = parse_hmac_keys()
    firebase_router = FirebaseRouter.create()
    summary_engine = SummaryEngine.create()
    worker_stop_event = asyncio.Event()
    worker_task = asyncio.create_task(
        worker_loop(worker_stop_event, recognizer, firebase_router, summary_engine)
    )
    summary_worker_task = asyncio.create_task(
        summary_worker_loop(worker_stop_event, firebase_router, summary_engine)
    )
    app.state.runtime = AppState(
        recognizer=recognizer,
        hmac_keys=hmac_keys,
        firebase_router=firebase_router,
        summary_engine=summary_engine,
        worker_task=worker_task,
        summary_worker_task=summary_worker_task,
        worker_stop_event=worker_stop_event,
    )
    yield
    runtime: AppState = app.state.runtime
    runtime.worker_stop_event.set()
    if runtime.worker_task:
        await runtime.worker_task
    if runtime.summary_worker_task:
        await runtime.summary_worker_task


app = FastAPI(title="Talk Transcript STT", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    pending = await asyncio.to_thread(db_pending_count)
    runtime: AppState = app.state.runtime
    return {
        "status": "ok",
        "model_dir": str(MODEL_DIR),
        "model_type": MODEL_TYPE,
        "language": MODEL_LANGUAGE,
        "sample_rate": SAMPLE_RATE,
        "feature_dim": FEATURE_DIM,
        "num_threads": NUM_THREADS,
        "decode_max_concurrency": DECODE_MAX_CONCURRENCY,
        "sqlite_path": str(SQLITE_PATH),
        "spool_dir": str(SPOOL_DIR),
        "queue_max_pending": QUEUE_MAX_PENDING,
        "room_inactivity_timeout_seconds": ROOM_INACTIVITY_TIMEOUT_SECONDS,
        "pending_jobs": pending,
        "firebase_enabled": FIREBASE_ENABLED,
        "firebase_namespace_configs": len(runtime.firebase_router.configs_by_namespace),
        "firebase_flush_interval_seconds": FIREBASE_FLUSH_INTERVAL_SECONDS,
        "storage_minute_window_seconds": STORAGE_MINUTE_WINDOW_SECONDS,
        "openai_summary_enabled": runtime.summary_engine.enabled,
        "openai_model_minute_summary": runtime.summary_engine.minute_model,
        "openai_model_accumulated_summary": runtime.summary_engine.accumulated_model,
        "openai_model_final_summary": runtime.summary_engine.final_model,
    }


@app.post("/v1/sessions/start")
async def session_start(request: Request) -> JSONResponse:
    body_bytes = await request.body()
    runtime: AppState = app.state.runtime
    verify_hmac_or_raise(request, body_bytes, runtime.hmac_keys)
    try:
        payload = StartRequest.model_validate_json(body_bytes)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    if extract_room_namespace(payload.room_name) is None:
        return ignored_namespace_response(
            route="/v1/sessions/start",
            room_name=payload.room_name,
            reason="invalid_namespace",
        )

    await asyncio.to_thread(db_upsert_start, payload)
    return JSONResponse({"status": "accepted"}, status_code=202)


@app.post("/v1/sessions/chunk")
async def session_chunk(request: Request) -> JSONResponse:
    body_bytes = await request.body()
    runtime: AppState = app.state.runtime
    verify_hmac_or_raise(request, body_bytes, runtime.hmac_keys)

    form = await request.form()
    meta_raw = form.get("meta")
    audio_file = form.get("audio")
    if not isinstance(meta_raw, str):
        return JSONResponse({"error": "campo meta ausente"}, status_code=400)
    if not isinstance(audio_file, UploadFile):
        return JSONResponse({"error": "campo audio ausente"}, status_code=400)

    try:
        meta_dict = json.loads(meta_raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="meta invalido") from exc

    try:
        meta = ChunkMeta.model_validate(meta_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    if extract_room_namespace(meta.room_name) is None:
        return ignored_namespace_response(
            route="/v1/sessions/chunk",
            room_name=meta.room_name,
            reason="invalid_namespace",
        )

    if await asyncio.to_thread(db_pending_count) >= QUEUE_MAX_PENDING:
        return JSONResponse({"error": "queue_overloaded"}, status_code=429)

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        return JSONResponse({"error": "audio vazio"}, status_code=400)

    spool_path = write_spool_file(meta, audio_bytes)
    enqueue_result = await asyncio.to_thread(db_enqueue_chunk, meta, spool_path)
    if enqueue_result != "accepted":
        spool_path.unlink(missing_ok=True)
        if enqueue_result == "queue_full":
            return JSONResponse({"error": "queue_overloaded"}, status_code=429)
        if enqueue_result in ("sequence_conflict", "session_not_found"):
            return JSONResponse({"error": enqueue_result}, status_code=409)
        return JSONResponse({"error": "enqueue_failed"}, status_code=500)

    return JSONResponse({"status": "accepted", "seq": meta.seq}, status_code=202)


@app.post("/v1/sessions/end")
async def session_end(request: Request) -> JSONResponse:
    body_bytes = await request.body()
    runtime: AppState = app.state.runtime
    verify_hmac_or_raise(request, body_bytes, runtime.hmac_keys)
    try:
        payload = EndRequest.model_validate_json(body_bytes)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    if extract_room_namespace(payload.room_name) is None:
        return ignored_namespace_response(
            route="/v1/sessions/end",
            room_name=payload.room_name,
            reason="invalid_namespace",
        )

    if payload.scope == "participant" and not payload.participant_identity:
        return JSONResponse(
            {"error": "participant_identity obrigatorio quando scope=participant"},
            status_code=400,
        )

    if payload.scope == "participant":
        await asyncio.to_thread(db_mark_participant_end, payload)
    else:
        await asyncio.to_thread(db_mark_room_end, payload)

    await finalize_entities(runtime.firebase_router, runtime.summary_engine)
    return JSONResponse({"status": "accepted"}, status_code=202)
