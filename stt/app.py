from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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

HMAC_WINDOW_SECONDS = int(os.environ.get("HMAC_WINDOW_SECONDS", "300"))
HMAC_KEYS_RAW = os.environ.get("STT_HMAC_KEYS", "")
HMAC_KEY_ID = os.environ.get("STT_HMAC_KEY_ID", "default")
HMAC_SECRET = os.environ.get("STT_HMAC_SECRET", "")

FIREBASE_ENABLED = os.environ.get("FIREBASE_ENABLED", "false").lower() == "true"
FIREBASE_SERVICE_ACCOUNT_FILE = os.environ.get("FIREBASE_SERVICE_ACCOUNT_FILE", "")
FIREBASE_STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")

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
    enabled: bool
    firestore_client: firestore.Client | None
    storage_bucket: storage.bucket.Bucket | None

    @staticmethod
    def create() -> "FirebaseSink":
        if not FIREBASE_ENABLED:
            logger.info("Firebase desabilitado (FIREBASE_ENABLED=false)")
            return FirebaseSink(False, None, None)

        if not firebase_admin._apps:
            cred = None
            service_file = FIREBASE_SERVICE_ACCOUNT_FILE or os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS", ""
            )
            if service_file:
                cred = credentials.Certificate(service_file)
            else:
                cred = credentials.ApplicationDefault()

            app_kwargs: dict = {}
            if FIREBASE_PROJECT_ID:
                app_kwargs["projectId"] = FIREBASE_PROJECT_ID
            if FIREBASE_STORAGE_BUCKET:
                app_kwargs["storageBucket"] = FIREBASE_STORAGE_BUCKET
            firebase_admin.initialize_app(cred, app_kwargs or None)

        fs_client = firestore.client()
        bucket = None
        if FIREBASE_STORAGE_BUCKET:
            bucket = storage.bucket(FIREBASE_STORAGE_BUCKET)
        return FirebaseSink(True, fs_client, bucket)

    def upsert_segment(self, chunk_row: sqlite3.Row, text: str) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        call_key = f"{chunk_row['room_name']}__{chunk_row['session_id']}"
        call_ref = self.firestore_client.collection("calls").document(call_key)
        participant_ref = (
            call_ref.collection("participants").document(chunk_row["participant_identity"])
        )
        segment_ref = participant_ref.collection("segments").document(str(chunk_row["seq"]))

        now = firestore.SERVER_TIMESTAMP
        call_ref.set(
            {
                "room_name": chunk_row["room_name"],
                "session_id": chunk_row["session_id"],
                "call_key": call_key,
                "status": "processing",
                "updated_at": now,
            },
            merge=True,
        )
        participant_ref.set(
            {
                "participant_identity": chunk_row["participant_identity"],
                "updated_at": now,
                "last_seq": chunk_row["seq"],
            },
            merge=True,
        )
        segment_ref.set(
            {
                "seq": chunk_row["seq"],
                "text": text,
                "chunk_started_at": chunk_row["chunk_started_at"],
                "chunk_ended_at": chunk_row["chunk_ended_at"],
                "created_at": now,
            },
            merge=True,
        )

    def upsert_participant_aggregate(
        self,
        room_name: str,
        session_id: str,
        participant_identity: str,
        transcript_text: str,
        segment_count: int,
        finalized: bool,
    ) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        call_key = f"{room_name}__{session_id}"
        now = firestore.SERVER_TIMESTAMP
        participant_ref = (
            self.firestore_client.collection("calls")
            .document(call_key)
            .collection("participants")
            .document(participant_identity)
        )
        participant_ref.set(
            {
                "transcript_text": transcript_text,
                "segment_count": segment_count,
                "finalized": finalized,
                "updated_at": now,
            },
            merge=True,
        )

    def upsert_room_aggregate(
        self,
        room_name: str,
        session_id: str,
        transcript_text: str,
        finalized: bool,
    ) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        call_key = f"{room_name}__{session_id}"
        call_ref = self.firestore_client.collection("calls").document(call_key)
        call_ref.set(
            {
                "room_name": room_name,
                "session_id": session_id,
                "call_key": call_key,
                "transcript_text": transcript_text,
                "status": "finalized" if finalized else "processing",
                "finalized": finalized,
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

    def upload_participant_artifacts(
        self,
        room_name: str,
        session_id: str,
        participant_identity: str,
        transcript_text: str,
        segments: list[dict],
    ) -> None:
        if not self.enabled or self.storage_bucket is None:
            return

        call_key = f"{room_name}__{session_id}"
        base = f"calls/{safe_key(call_key)}/participants/{safe_key(participant_identity)}"
        txt_blob = self.storage_bucket.blob(f"{base}/transcript.txt")
        txt_blob.upload_from_string(transcript_text, content_type="text/plain; charset=utf-8")

        json_blob = self.storage_bucket.blob(f"{base}/transcript.json")
        json_blob.upload_from_string(
            json.dumps(
                {
                    "room_name": room_name,
                    "session_id": session_id,
                    "participant_identity": participant_identity,
                    "segments": segments,
                    "transcript_text": transcript_text,
                },
                ensure_ascii=False,
            ),
            content_type="application/json",
        )

    def upload_room_artifacts(
        self,
        room_name: str,
        session_id: str,
        transcript_text: str,
        lines: list[dict],
    ) -> None:
        if not self.enabled or self.storage_bucket is None:
            return

        call_key = f"{room_name}__{session_id}"
        base = f"calls/{safe_key(call_key)}"
        txt_blob = self.storage_bucket.blob(f"{base}/transcript.txt")
        txt_blob.upload_from_string(transcript_text, content_type="text/plain; charset=utf-8")

        json_blob = self.storage_bucket.blob(f"{base}/transcript.json")
        json_blob.upload_from_string(
            json.dumps(
                {
                    "room_name": room_name,
                    "session_id": session_id,
                    "transcript_text": transcript_text,
                    "lines": lines,
                },
                ensure_ascii=False,
            ),
            content_type="application/json",
        )


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
                state, room_end_received, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', 0, ?, ?)
            ON CONFLICT(room_name, session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                metadata_json=COALESCE(excluded.metadata_json, sessions.metadata_json)
            """,
            (
                req.room_name,
                req.session_id,
                req.started_at,
                json.dumps(req.metadata, ensure_ascii=False) if req.metadata is not None else None,
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
            SELECT participant_identity, seq, transcript, chunk_started_at, chunk_ended_at
            FROM chunks
            WHERE room_name=? AND session_id=? AND status='done'
            ORDER BY chunk_started_at ASC, participant_identity ASC, seq ASC
            """,
            (room_name, session_id),
        ).fetchall()
        lines = []
        for row in rows:
            text = (row["transcript"] or "").strip()
            if not text:
                continue
            lines.append(
                {
                    "participant_identity": row["participant_identity"],
                    "seq": int(row["seq"]),
                    "text": text,
                    "chunk_started_at": row["chunk_started_at"],
                    "chunk_ended_at": row["chunk_ended_at"],
                }
            )

        transcript = "\n".join(
            f"[{line['participant_identity']}] {line['text']}" for line in lines
        )
        return transcript, lines
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


def decode_pcm_file(recognizer: sherpa_onnx.OfflineRecognizer, path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    samples_int16 = np.frombuffer(data, dtype=np.int16)
    if samples_int16.size == 0:
        return ""
    samples = samples_int16.astype(np.float32) / 32768.0
    return decode_with_tail_padding(recognizer, samples)


async def finalize_entities(firebase_sink: FirebaseSink) -> None:
    participants = db_get_finalizable_participants()
    for participant in participants:
        room_name = participant["room_name"]
        session_id = participant["session_id"]
        participant_identity = participant["participant_identity"]

        transcript_text, segments = db_get_participant_aggregate(
            room_name, session_id, participant_identity
        )
        await asyncio.to_thread(
            firebase_sink.upsert_participant_aggregate,
            room_name,
            session_id,
            participant_identity,
            transcript_text,
            len(segments),
            True,
        )
        await asyncio.to_thread(
            firebase_sink.upload_participant_artifacts,
            room_name,
            session_id,
            participant_identity,
            transcript_text,
            segments,
        )
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
        transcript_text, lines = db_get_room_aggregate(room_name, session_id)
        await asyncio.to_thread(
            firebase_sink.upsert_room_aggregate,
            room_name,
            session_id,
            transcript_text,
            True,
        )
        await asyncio.to_thread(
            firebase_sink.upload_room_artifacts,
            room_name,
            session_id,
            transcript_text,
            lines,
        )
        db_mark_room_finalized(room_name, session_id)
        logger.info("room finalized room=%s session=%s", room_name, session_id)


async def worker_loop(
    stop_event: asyncio.Event,
    recognizer: sherpa_onnx.OfflineRecognizer,
    firebase_sink: FirebaseSink,
) -> None:
    logger.info("worker started")
    while not stop_event.is_set():
        chunk_row = await asyncio.to_thread(db_claim_next_chunk)
        if not chunk_row:
            await finalize_entities(firebase_sink)
            await asyncio.sleep(WORKER_POLL_SECONDS)
            continue

        spool_path = Path(chunk_row["spool_path"])
        try:
            text = await asyncio.to_thread(decode_pcm_file, recognizer, spool_path)
            await asyncio.to_thread(db_mark_chunk_done, chunk_row, text)
            await asyncio.to_thread(firebase_sink.upsert_segment, chunk_row, text)

            aggregate_text, segments = await asyncio.to_thread(
                db_get_participant_aggregate,
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
            )
            await asyncio.to_thread(
                firebase_sink.upsert_participant_aggregate,
                chunk_row["room_name"],
                chunk_row["session_id"],
                chunk_row["participant_identity"],
                aggregate_text,
                len(segments),
                False,
            )

            room_text, _ = await asyncio.to_thread(
                db_get_room_aggregate,
                chunk_row["room_name"],
                chunk_row["session_id"],
            )
            await asyncio.to_thread(
                firebase_sink.upsert_room_aggregate,
                chunk_row["room_name"],
                chunk_row["session_id"],
                room_text,
                False,
            )

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

    logger.info("worker stopped")


@dataclass
class AppState:
    recognizer: sherpa_onnx.OfflineRecognizer
    hmac_keys: dict[str, str]
    firebase_sink: FirebaseSink
    worker_task: asyncio.Task | None
    worker_stop_event: asyncio.Event


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recognizer = build_offline_recognizer()
    hmac_keys = parse_hmac_keys()
    firebase_sink = FirebaseSink.create()
    worker_stop_event = asyncio.Event()
    worker_task = asyncio.create_task(worker_loop(worker_stop_event, recognizer, firebase_sink))
    app.state.runtime = AppState(
        recognizer=recognizer,
        hmac_keys=hmac_keys,
        firebase_sink=firebase_sink,
        worker_task=worker_task,
        worker_stop_event=worker_stop_event,
    )
    yield
    runtime: AppState = app.state.runtime
    runtime.worker_stop_event.set()
    if runtime.worker_task:
        await runtime.worker_task


app = FastAPI(title="Talk Transcript STT", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    pending = await asyncio.to_thread(db_pending_count)
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
        "pending_jobs": pending,
        "firebase_enabled": FIREBASE_ENABLED,
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

    await asyncio.to_thread(db_upsert_start, payload)
    return JSONResponse({"status": "accepted"}, status_code=202)


@app.post("/v1/sessions/chunk")
async def session_chunk(request: Request) -> JSONResponse:
    body_bytes = await request.body()
    runtime: AppState = app.state.runtime
    verify_hmac_or_raise(request, body_bytes, runtime.hmac_keys)

    if await asyncio.to_thread(db_pending_count) >= QUEUE_MAX_PENDING:
        return JSONResponse({"error": "queue_overloaded"}, status_code=429)

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

    if payload.scope == "participant" and not payload.participant_identity:
        return JSONResponse(
            {"error": "participant_identity obrigatorio quando scope=participant"},
            status_code=400,
        )

    if payload.scope == "participant":
        await asyncio.to_thread(db_mark_participant_end, payload)
    else:
        await asyncio.to_thread(db_mark_room_end, payload)

    await finalize_entities(runtime.firebase_sink)
    return JSONResponse({"status": "accepted"}, status_code=202)
