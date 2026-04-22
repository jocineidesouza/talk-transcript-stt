from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

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
APP_VERSION = os.environ.get("APP_VERSION", "1.00.01").strip() or "1.00.01"

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
OPENAI_REQUEST_RETRIES = max(0, int(os.environ.get("OPENAI_REQUEST_RETRIES", "2")))
OPENAI_REQUEST_RETRY_BASE_SECONDS = max(
    0.1, float(os.environ.get("OPENAI_REQUEST_RETRY_BASE_SECONDS", "1.5"))
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

CALL_INDEX_UNSET = object()

DEFAULT_SUMMARIZE_MINUTE_PROMPT = """Voce e um assistente especializado em analisar trechos de chamadas corporativas em portugues do Brasil.

Entrada:
- Um TRECHO de conversa referente a um intervalo arbitrario (ex.: 30s, 1min, 3min, 5min).
- O trecho pode estar incompleto, com interrupcoes, mudancas de contexto, erros de transcricao e falas fragmentadas.

Tarefa:
- Extrair informacoes estruturadas APENAS do trecho recebido, de forma conservadora e confiavel.
- Seguir estritamente o contrato JSON de saida fornecido separadamente.

Regras obrigatorias:
- Use somente o conteudo do trecho.
- Nao invente contexto anterior/posterior.
- Nao complete raciocinios interrompidos.
- Nao inferir datas, prazos, responsaveis ou sequencias temporais a partir de fala ambigua.
- Nao transformar sugestoes, hipoteses, perguntas, desejos, intencoes ou condicionais em decisoes/fatos.
- Acoes futuras nunca entram em facts.
- Perguntas sem resposta entram em open_items ou notes, nunca em facts.
- Se houver contradicao interna, nao resolver: registrar em notes.
- Se termo/nome estiver pouco confiavel, nao tratar como entidade confiavel.
- Priorize confiabilidade sobre completude.
- Elimine duplicacao semantica entre categorias.
- Se houver duvida entre facts e hypotheses, priorize hypotheses.
- hypotheses e notes nao podem ter confidence=high.
- Tags curtas e especificas.
- Limites: text max 240 caracteres; tags entre 1 e 4 itens; maximo 8 itens por categoria.
- Se o trecho for inutil/ruidoso, retornar arrays vazios e registrar 1 note curta.
"""

DEFAULT_MERGE_SUMMARIES_PROMPT = """Voce e um assistente responsavel por atualizar o estado acumulado de uma chamada corporativa em portugues do Brasil.

Entrada:
1) estado acumulado anterior em JSON
2) novo resumo de trecho (chunk) em JSON

Tarefa:
- Atualizar o estado acumulado usando apenas as informacoes recebidas.
- Nao reescrever do zero.
- Seguir estritamente o contrato JSON de saida fornecido separadamente.

Regras obrigatorias:
- Nao inventar informacoes.
- Nao promover hypotheses para facts/decisions sem evidencia explicita.
- Nao transformar planos/condicionais em fatos confirmados.
- Eliminar duplicacoes literais e semanticas.
- Manter a versao mais clara/completa/confiavel quando houver equivalencia.
- Nao sobrescrever fatos anteriores sem evidencia explicita.
- Em contradicoes relevantes, manter as visoes e registrar inconsistencia em notes.
- conversation_types: manter existentes e adicionar novos sem duplicar.
- next_steps so vira fact com evidencia explicita posterior.
- open_items so remove com evidencia explicita de resolucao.
- hypotheses so remove com confirmacao/invalidacao explicita.
- Consolidar agressivamente para evitar listas infladas.
- hypotheses e notes nao podem ter confidence=high.
- Limites: text max 240 caracteres; tags entre 1 e 4 itens; maximo 20 itens por categoria.
- Retornar arrays vazios quando categoria nao tiver itens.
"""

DEFAULT_FINALIZE_SUMMARY_PROMPT = """Voce e um assistente especializado em produzir resumo final estruturado de chamadas corporativas em portugues do Brasil.

Entrada:
- Estado acumulado em JSON.

Tarefa:
- Gerar resumo final fiel, claro, conservador e util para exibicao em frontend.
- Seguir estritamente o contrato JSON de saida fornecido separadamente.

Objetivo:
- Explicar o que aconteceu, o que foi decidido, o que esta em aberto, proximos passos e pontos de baixa confianca/ambiguidade.

Regras obrigatorias:
- Use apenas informacoes do estado acumulado.
- Nao invente fatos, decisoes, prazos, responsaveis ou contexto extra.
- Nao transformar hypotheses em conclusoes.
- Nao transformar next_steps em fatos realizados.
- Nao transformar pending_items em decisoes.
- Consolidar redundancias e manter apenas o que for mais claro/relevante/confiavel.
- Nao repetir o mesmo conteudo em secoes diferentes sem necessidade.
- Se nao houver decisao explicita, inserir exatamente:
  "Nenhuma decisao explicita foi registrada."
- title deve ser exatamente: "Resumo Final Executivo da Chamada".
- conversation_types deve refletir o estado acumulado.
- Limites: text max 280 caracteres; tags entre 1 e 4 itens; maximo 12 itens por categoria.
- Retornar arrays vazios quando categoria nao tiver itens (exceto regra de decisions acima).
"""

SUMMARY_KIND_MINUTE = "minute"
SUMMARY_KIND_ACCUMULATED = "accumulated"
SUMMARY_KIND_FINAL = "final"

SUMMARY_ALLOWED_TYPES = {"tecnica", "executiva", "operacional", "comercial", "mista"}
SUMMARY_ALLOWED_CONFIDENCE = {"high", "medium", "low"}

SUMMARY_SCHEMA_MINUTE = {
    "chunk_type": "tecnica|executiva|operacional|comercial|mista",
    "facts": [{"text": "string", "confidence": "high|medium|low", "status": "confirmed|uncertain", "tags": ["string"]}],
    "hypotheses": [{"text": "string", "confidence": "medium|low", "status": "uncertain", "tags": ["string"]}],
    "decisions": [{"text": "string", "confidence": "high|medium|low", "status": "confirmed", "tags": ["string"]}],
    "open_items": [{"text": "string", "confidence": "high|medium|low", "status": "open", "tags": ["string"]}],
    "next_steps": [{"text": "string", "confidence": "high|medium|low", "status": "planned", "tags": ["string"]}],
    "notes": [{"text": "string", "confidence": "medium|low", "status": "uncertain|info", "tags": ["string"]}],
}

SUMMARY_SCHEMA_ACCUMULATED = {
    "conversation_types": ["tecnica|executiva|operacional|comercial|mista"],
    "facts": [{"text": "string", "confidence": "high|medium|low", "status": "confirmed|uncertain", "tags": ["string"]}],
    "hypotheses": [{"text": "string", "confidence": "medium|low", "status": "uncertain", "tags": ["string"]}],
    "decisions": [{"text": "string", "confidence": "high|medium|low", "status": "confirmed", "tags": ["string"]}],
    "open_items": [{"text": "string", "confidence": "high|medium|low", "status": "open", "tags": ["string"]}],
    "next_steps": [{"text": "string", "confidence": "high|medium|low", "status": "planned", "tags": ["string"]}],
    "notes": [{"text": "string", "confidence": "medium|low", "status": "uncertain|info", "tags": ["string"]}],
}

SUMMARY_SCHEMA_FINAL = {
    "title": "Resumo Final Executivo da Chamada",
    "conversation_types": ["tecnica|executiva|operacional|comercial|mista"],
    "main_points": [{"text": "string", "confidence": "high|medium|low", "tags": ["string"]}],
    "decisions": [{"text": "string", "confidence": "high|medium|low", "tags": ["string"]}],
    "pending_items": [{"text": "string", "confidence": "high|medium|low", "tags": ["string"]}],
    "next_steps": [{"text": "string", "confidence": "high|medium|low", "tags": ["string"]}],
    "additional_notes": [{"text": "string", "confidence": "high|medium|low", "tags": ["string"]}],
}

CONTRACT_SUFFIX_MINUTE = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "- Regra obrigatoria de confidence por secao:\n"
    "  * hypotheses[].confidence: apenas low ou medium (nunca high)\n"
    "  * notes[].confidence: apenas low ou medium (nunca high)\n"
    "Use exatamente este schema:\n"
    f"{json.dumps(SUMMARY_SCHEMA_MINUTE, ensure_ascii=True, indent=2)}"
)

CONTRACT_SUFFIX_ACCUMULATED = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "Use exatamente este schema:\n"
    f"{json.dumps(SUMMARY_SCHEMA_ACCUMULATED, ensure_ascii=True, indent=2)}"
)

CONTRACT_SUFFIX_FINAL = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "Use exatamente este schema:\n"
    f"{json.dumps(SUMMARY_SCHEMA_FINAL, ensure_ascii=True, indent=2)}"
)


def build_effective_system_prompt(
    system_prompt_override: str | None,
    default_prompt: str,
    contract_suffix: str,
) -> str:
    base_prompt = (
        system_prompt_override.strip()
        if isinstance(system_prompt_override, str) and system_prompt_override.strip()
        else default_prompt.strip()
    )
    return f"{base_prompt}\n\n{contract_suffix}"


def strip_code_fences(value: str) -> str:
    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def parse_json_model_output(raw_output: str) -> dict:
    cleaned = strip_code_fences(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"resposta de resumo nao e JSON valido: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("resposta de resumo deve ser JSON objeto")
    return parsed


def assert_no_extra_keys(payload: dict, allowed_keys: set[str], label: str) -> None:
    extra = set(payload.keys()) - allowed_keys
    if extra:
        raise RuntimeError(f"{label}: campos extras nao permitidos: {sorted(extra)}")


def validate_summary_tags(tags: Any, label: str) -> list[str]:
    if not isinstance(tags, list):
        raise RuntimeError(f"{label}.tags deve ser array")
    if len(tags) < 1 or len(tags) > 4:
        raise RuntimeError(f"{label}.tags deve conter entre 1 e 4 itens")
    normalized: list[str] = []
    for index, tag in enumerate(tags):
        if not isinstance(tag, str):
            raise RuntimeError(f"{label}.tags[{index}] deve ser string")
        clean = tag.strip()
        if not clean:
            raise RuntimeError(f"{label}.tags[{index}] nao pode ser vazio")
        if len(clean) > 32:
            raise RuntimeError(f"{label}.tags[{index}] excede 32 caracteres")
        normalized.append(clean)
    return normalized


def validate_summary_item(
    item: Any,
    label: str,
    allowed_confidence: set[str],
    allowed_status: set[str] | None,
    max_text: int,
    include_status: bool,
) -> dict:
    if not isinstance(item, dict):
        raise RuntimeError(f"{label} deve ser objeto")
    expected_keys = {"text", "confidence", "tags"}
    if include_status:
        expected_keys.add("status")
    assert_no_extra_keys(item, expected_keys, label)
    text = item.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"{label}.text deve ser string")
    text = text.strip()
    if not text:
        raise RuntimeError(f"{label}.text nao pode ser vazio")
    if len(text) > max_text:
        raise RuntimeError(f"{label}.text excede {max_text} caracteres")
    raw_confidence = item.get("confidence")
    normalized_confidence = (
        raw_confidence.strip().lower() if isinstance(raw_confidence, str) else None
    )
    confidence: str
    if isinstance(normalized_confidence, str) and normalized_confidence in allowed_confidence:
        confidence = normalized_confidence
    else:
        confidence = "medium" if "medium" in allowed_confidence else "low"
        logger.warning(
            "summary confidence normalized field=%s raw=%s normalized=%s allowed=%s",
            label,
            raw_confidence,
            confidence,
            sorted(allowed_confidence),
        )
    normalized: dict[str, Any] = {
        "text": text,
        "confidence": confidence,
        "tags": validate_summary_tags(item.get("tags"), label),
    }
    if include_status:
        status = item.get("status")
        if not isinstance(status, str) or (allowed_status is not None and status not in allowed_status):
            raise RuntimeError(f"{label}.status invalido")
        normalized["status"] = status
    return normalized


def validate_summary_list(
    value: Any,
    key: str,
    max_items: int,
    item_validator: Callable[[Any, str], dict],
) -> list[dict]:
    if not isinstance(value, list):
        raise RuntimeError(f"{key} deve ser array")
    if len(value) > max_items:
        raise RuntimeError(f"{key} excede maximo de {max_items} itens")
    return [item_validator(item, f"{key}[{index}]") for index, item in enumerate(value)]


def default_accumulated_summary_payload() -> dict:
    return {
        "conversation_types": [],
        "facts": [],
        "hypotheses": [],
        "decisions": [],
        "open_items": [],
        "next_steps": [],
        "notes": [],
    }


def validate_minute_summary_payload(payload: dict) -> dict:
    required_keys = {"chunk_type", "facts", "hypotheses", "decisions", "open_items", "next_steps", "notes"}
    assert_no_extra_keys(payload, required_keys, SUMMARY_KIND_MINUTE)
    for key in required_keys:
        if key not in payload:
            raise RuntimeError(f"{SUMMARY_KIND_MINUTE}: campo obrigatorio ausente: {key}")
    chunk_type = payload.get("chunk_type")
    if not isinstance(chunk_type, str) or chunk_type not in SUMMARY_ALLOWED_TYPES:
        raise RuntimeError("minute.chunk_type invalido")
    return {
        "chunk_type": chunk_type,
        "facts": validate_summary_list(
            payload.get("facts"),
            "facts",
            8,
            lambda item, label: validate_summary_item(
                item,
                label,
                SUMMARY_ALLOWED_CONFIDENCE,
                {"confirmed", "uncertain"},
                240,
                True,
            ),
        ),
        "hypotheses": validate_summary_list(
            payload.get("hypotheses"),
            "hypotheses",
            8,
            lambda item, label: validate_summary_item(
                item, label, {"medium", "low"}, {"uncertain"}, 240, True
            ),
        ),
        "decisions": validate_summary_list(
            payload.get("decisions"),
            "decisions",
            8,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"confirmed"}, 240, True
            ),
        ),
        "open_items": validate_summary_list(
            payload.get("open_items"),
            "open_items",
            8,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"open"}, 240, True
            ),
        ),
        "next_steps": validate_summary_list(
            payload.get("next_steps"),
            "next_steps",
            8,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"planned"}, 240, True
            ),
        ),
        "notes": validate_summary_list(
            payload.get("notes"),
            "notes",
            8,
            lambda item, label: validate_summary_item(
                item, label, {"medium", "low"}, {"uncertain", "info"}, 240, True
            ),
        ),
    }


def validate_accumulated_summary_payload(payload: dict) -> dict:
    required_keys = {
        "conversation_types",
        "facts",
        "hypotheses",
        "decisions",
        "open_items",
        "next_steps",
        "notes",
    }
    assert_no_extra_keys(payload, required_keys, SUMMARY_KIND_ACCUMULATED)
    for key in required_keys:
        if key not in payload:
            raise RuntimeError(f"{SUMMARY_KIND_ACCUMULATED}: campo obrigatorio ausente: {key}")
    conversation_types_raw = payload.get("conversation_types")
    if not isinstance(conversation_types_raw, list):
        raise RuntimeError("accumulated.conversation_types deve ser array")
    conversation_types: list[str] = []
    seen_types: set[str] = set()
    for index, item in enumerate(conversation_types_raw):
        if not isinstance(item, str) or item not in SUMMARY_ALLOWED_TYPES:
            raise RuntimeError(f"accumulated.conversation_types[{index}] invalido")
        if item in seen_types:
            continue
        seen_types.add(item)
        conversation_types.append(item)
    return {
        "conversation_types": conversation_types,
        "facts": validate_summary_list(
            payload.get("facts"),
            "facts",
            20,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"confirmed", "uncertain"}, 240, True
            ),
        ),
        "hypotheses": validate_summary_list(
            payload.get("hypotheses"),
            "hypotheses",
            20,
            lambda item, label: validate_summary_item(
                item, label, {"medium", "low"}, {"uncertain"}, 240, True
            ),
        ),
        "decisions": validate_summary_list(
            payload.get("decisions"),
            "decisions",
            20,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"confirmed"}, 240, True
            ),
        ),
        "open_items": validate_summary_list(
            payload.get("open_items"),
            "open_items",
            20,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"open"}, 240, True
            ),
        ),
        "next_steps": validate_summary_list(
            payload.get("next_steps"),
            "next_steps",
            20,
            lambda item, label: validate_summary_item(
                item, label, SUMMARY_ALLOWED_CONFIDENCE, {"planned"}, 240, True
            ),
        ),
        "notes": validate_summary_list(
            payload.get("notes"),
            "notes",
            20,
            lambda item, label: validate_summary_item(
                item, label, {"medium", "low"}, {"uncertain", "info"}, 240, True
            ),
        ),
    }


def validate_final_summary_payload(payload: dict) -> dict:
    required_keys = {
        "title",
        "conversation_types",
        "main_points",
        "decisions",
        "pending_items",
        "next_steps",
        "additional_notes",
    }
    assert_no_extra_keys(payload, required_keys, SUMMARY_KIND_FINAL)
    for key in required_keys:
        if key not in payload:
            raise RuntimeError(f"{SUMMARY_KIND_FINAL}: campo obrigatorio ausente: {key}")
    title = payload.get("title")
    if title != "Resumo Final Executivo da Chamada":
        raise RuntimeError("final.title invalido")
    conversation_types_raw = payload.get("conversation_types")
    if not isinstance(conversation_types_raw, list):
        raise RuntimeError("final.conversation_types deve ser array")
    conversation_types: list[str] = []
    seen_types: set[str] = set()
    for index, item in enumerate(conversation_types_raw):
        if not isinstance(item, str) or item not in SUMMARY_ALLOWED_TYPES:
            raise RuntimeError(f"final.conversation_types[{index}] invalido")
        if item in seen_types:
            continue
        seen_types.add(item)
        conversation_types.append(item)

    def _final_item(item: Any, label: str) -> dict:
        return validate_summary_item(
            item,
            label,
            SUMMARY_ALLOWED_CONFIDENCE,
            None,
            280,
            False,
        )

    return {
        "title": title,
        "conversation_types": conversation_types,
        "main_points": validate_summary_list(payload.get("main_points"), "main_points", 12, _final_item),
        "decisions": validate_summary_list(payload.get("decisions"), "decisions", 12, _final_item),
        "pending_items": validate_summary_list(payload.get("pending_items"), "pending_items", 12, _final_item),
        "next_steps": validate_summary_list(payload.get("next_steps"), "next_steps", 12, _final_item),
        "additional_notes": validate_summary_list(payload.get("additional_notes"), "additional_notes", 12, _final_item),
    }


def validate_summary_payload(kind: str, payload: dict) -> dict:
    if kind == SUMMARY_KIND_MINUTE:
        return validate_minute_summary_payload(payload)
    if kind == SUMMARY_KIND_ACCUMULATED:
        return validate_accumulated_summary_payload(payload)
    if kind == SUMMARY_KIND_FINAL:
        return validate_final_summary_payload(payload)
    raise RuntimeError(f"tipo de resumo nao suportado: {kind}")


def parse_and_validate_summary_output(kind: str, raw_output: str) -> dict:
    payload = parse_json_model_output(raw_output)
    return validate_summary_payload(kind, payload)


class StartRequest(BaseModel):
    room_name: str = Field(min_length=1)
    call_session_id: str = Field(min_length=1)
    transcript_session_id: str | None = None
    participant_identity: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    participant_name: str | None = None
    track_sid: str | None = None
    metadata: dict | None = None

    @field_validator("call_session_id")
    @classmethod
    def validate_call_session_id(cls, value: str) -> str:
        if not value.startswith("RM_"):
            raise ValueError("call_session_id deve iniciar com RM_")
        return value

    @field_validator("transcript_session_id", mode="before")
    @classmethod
    def normalize_transcript_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class ChunkMeta(BaseModel):
    room_name: str = Field(min_length=1)
    call_session_id: str = Field(min_length=1)
    transcript_session_id: str | None = None
    participant_identity: str = Field(min_length=1)
    seq: int = Field(ge=1)
    chunk_started_at: str = Field(min_length=1)
    chunk_ended_at: str = Field(min_length=1)
    sample_rate: int
    channels: int
    encoding: str
    participant_name: str | None = None
    track_sid: str | None = None

    @field_validator("call_session_id")
    @classmethod
    def validate_call_session_id(cls, value: str) -> str:
        if not value.startswith("RM_"):
            raise ValueError("call_session_id deve iniciar com RM_")
        return value

    @field_validator("transcript_session_id", mode="before")
    @classmethod
    def normalize_transcript_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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
    call_session_id: str = Field(min_length=1)
    transcript_session_id: str | None = None
    scope: str = Field(min_length=1)
    participant_identity: str | None = None
    ended_at: str | None = None
    metadata: dict | None = None

    @field_validator("call_session_id")
    @classmethod
    def validate_call_session_id(cls, value: str) -> str:
        if not value.startswith("RM_"):
            raise ValueError("call_session_id deve iniciar com RM_")
        return value

    @field_validator("transcript_session_id", mode="before")
    @classmethod
    def normalize_transcript_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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
    call_session_id: str
    transcript_session_id: str | None
    minute_index: int
    minute_started_at: str
    minute_ended_at: str
    transcript_json_path: str
    summary_json_path: str
    lines: list[dict]
    line_count: int
    transcript_hash: str
    finalized: bool


@dataclass(frozen=True)
class RoomRoutingContext:
    namespace: str
    room_name: str
    room_id: str
    call_session_id: str
    transcript_session_id: str | None
    vertical: str
    slug: str
    firestore_doc_path: str
    storage_base_path: str


class RoomRoutingError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def extract_room_namespace(room_name: str) -> RoomNamespaceInfo | None:
    for namespace in sorted(ALLOWED_LIVEKIT_NAMESPACES, key=len, reverse=True):
        prefix = f"{namespace}__"
        if room_name.startswith(prefix):
            room_id = room_name[len(prefix) :].strip()
            if room_id:
                return RoomNamespaceInfo(namespace=namespace, room_id=room_id)
            return None
    return None


def build_firestore_doc_path(vertical: str, slug: str, room_id: str, call_session_id: str) -> str:
    return f"VERTICALS/{vertical}/COMPANIES/{slug}/ROOMS/{room_id}/SESSIONS/{call_session_id}"


def build_storage_base_path(vertical: str, slug: str, room_id: str, call_session_id: str) -> str:
    return f"VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}"


def build_room_session_doc_path(vertical: str, slug: str, room_id: str, call_session_id: str) -> str:
    return f"VERTICALS/{vertical}/COMPANIES/{slug}/ROOMS/{room_id}/SESSIONS/{call_session_id}"


def build_agent_prompt_doc_path(vertical: str, slug: str, agent_id: str) -> str:
    return f"VERTICALS/{vertical}/COMPANIES/{slug}/SETTINGS/ai_agents/AGENTS/{agent_id}"


def join_storage_path(storage_base_path: str, suffix: str) -> str:
    return f"{storage_base_path.rstrip('/')}/{suffix.lstrip('/')}"


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
                call_session_id TEXT NOT NULL,
                transcript_session_id TEXT,
                room_id TEXT,
                vertical TEXT,
                slug TEXT,
                firestore_doc_path TEXT,
                storage_base_path TEXT,
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
        if "call_session_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN call_session_id TEXT")
        if "transcript_session_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN transcript_session_id TEXT")
        if "room_id" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN room_id TEXT")
        if "vertical" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN vertical TEXT")
        if "slug" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN slug TEXT")
        if "firestore_doc_path" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN firestore_doc_path TEXT")
        if "storage_base_path" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN storage_base_path TEXT")
        conn.execute("UPDATE sessions SET call_session_id = room_id WHERE call_session_id IS NULL OR call_session_id = ''")
        conn.execute(
            """
            UPDATE sessions
            SET transcript_session_id = session_id
            WHERE transcript_session_id IS NULL OR transcript_session_id = ''
            """
        )
        conn.execute("UPDATE sessions SET last_chunk_at = started_at WHERE last_chunk_at IS NULL")
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

    if MODEL_TYPE.startswith("omnilingual_asr_ctc"):
        return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=require_one([MODEL_DIR / "model.int8.onnx", MODEL_DIR / "model.onnx"]),
            tokens=require_file(MODEL_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
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

    def _doc_ref(self, firestore_doc_path: str):
        if self.firestore_client is None:
            raise RuntimeError("firestore_client indisponivel")
        return self.firestore_client.document(firestore_doc_path)

    def publish_call_index(
        self,
        routing: RoomRoutingContext,
        status: str,
        last_minute_index: int,
        finalized: bool,
        minute_window_seconds: int,
        flush_interval_seconds: int,
        final_summary_path: str | None | object = CALL_INDEX_UNSET,
        summary_accumulated_path: str | None | object = CALL_INDEX_UNSET,
        final_summary_ready: bool | object = CALL_INDEX_UNSET,
        final_transcript_path: str | None | object = CALL_INDEX_UNSET,
        final_transcript_ready: bool | object = CALL_INDEX_UNSET,
    ) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        call_ref = self._doc_ref(routing.firestore_doc_path)
        payload = {
            "room_name": routing.room_name,
            "room_id": routing.room_id,
            "transcript_session_id": routing.transcript_session_id,
            "call_session_id": routing.call_session_id,
            "namespace": self.namespace,
            "vertical": routing.vertical,
            "slug": routing.slug,
            "status": status,
            "finalized": finalized,
            "last_minute_index": last_minute_index,
            "minute_window_seconds": minute_window_seconds,
            "flush_interval_seconds": flush_interval_seconds,
            "storage_base": routing.storage_base_path,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        if summary_accumulated_path is not CALL_INDEX_UNSET:
            payload["summary_accumulated_path"] = summary_accumulated_path
        if final_summary_path is not CALL_INDEX_UNSET:
            payload["final_summary_path"] = final_summary_path
        if final_summary_ready is not CALL_INDEX_UNSET:
            payload["final_summary_ready"] = final_summary_ready
        if final_transcript_path is not CALL_INDEX_UNSET:
            payload["final_transcript_path"] = final_transcript_path
        if final_transcript_ready is not CALL_INDEX_UNSET:
            payload["final_transcript_ready"] = final_transcript_ready

        call_ref.set(payload, merge=True)

    def upsert_room_session_links_on_start(self, routing: RoomRoutingContext) -> None:
        if not self.enabled or self.firestore_client is None:
            return

        session_ref = self._doc_ref(routing.firestore_doc_path)
        transcript_session_ids = (
            firestore.ArrayUnion([routing.transcript_session_id]) if routing.transcript_session_id else None
        )
        payload = {
            "call_session_id": routing.call_session_id,
            "transcript_session_id": routing.transcript_session_id,
            "agent_stt_id": routing.transcript_session_id,
            "status": "active",
            "started_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        if transcript_session_ids is not None:
            payload["transcript_session_ids"] = transcript_session_ids
        session_ref.set(payload, merge=True)

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

    def upload_minute_transcript(
        self, routing: RoomRoutingContext, payload: MinuteShardPayload
    ) -> None:
        self.upload_json(
            payload.transcript_json_path,
            {
                "room_name": payload.room_name,
                "transcript_session_id": payload.transcript_session_id,
                "call_session_id": payload.call_session_id,
                "namespace": self.namespace,
                "vertical": routing.vertical,
                "slug": routing.slug,
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

    def sink_for_namespace(self, namespace: str) -> FirebaseSink:
        return self._get_or_create_sink(namespace)
    def resolve_room_routing_context(
        self,
        room_name: str,
        call_session_id: str,
        transcript_session_id: str | None,
    ) -> RoomRoutingContext:
        info = extract_room_namespace(room_name)
        if info is None:
            raise RoomRoutingError("invalid_namespace", f"namespace invalido para room {room_name}")

        if not self.enabled:
            vertical = "disabled"
            slug = "disabled"
            return RoomRoutingContext(
                namespace=info.namespace,
                room_name=room_name,
                room_id=info.room_id,
                call_session_id=call_session_id,
                transcript_session_id=transcript_session_id,
                vertical=vertical,
                slug=slug,
                firestore_doc_path=build_firestore_doc_path(vertical, slug, info.room_id, call_session_id),
                storage_base_path=build_storage_base_path(vertical, slug, info.room_id, call_session_id),
            )

        sink = self._get_or_create_sink(info.namespace)
        if not sink.enabled or sink.firestore_client is None:
            raise RoomRoutingError(
                "room_index_invalid",
                f"namespace sem firestore habilitado para room {room_name}",
            )

        index_ref = sink.firestore_client.collection("LIVEKIT_ROOM_INDEX").document(room_name)
        snapshot = index_ref.get()
        if not snapshot.exists:
            raise RoomRoutingError(
                "room_index_not_found",
                f"indice LIVEKIT_ROOM_INDEX nao encontrado para room {room_name}",
            )
        raw_payload = snapshot.to_dict()
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        vertical = str(payload.get("vertical", "")).strip()
        slug = str(payload.get("slug", "")).strip()
        if not vertical or not slug:
            raise RoomRoutingError(
                "room_index_invalid",
                f"indices vertical/slug invalidos para room {room_name}",
            )

        return RoomRoutingContext(
            namespace=info.namespace,
            room_name=room_name,
            room_id=info.room_id,
            call_session_id=call_session_id,
            transcript_session_id=transcript_session_id,
            vertical=vertical,
            slug=slug,
            firestore_doc_path=build_firestore_doc_path(vertical, slug, info.room_id, call_session_id),
            storage_base_path=build_storage_base_path(vertical, slug, info.room_id, call_session_id),
        )
    def publish_call_index(
        self,
        routing: RoomRoutingContext,
        status: str,
        last_minute_index: int,
        finalized: bool,
        final_summary_path: str | None | object = CALL_INDEX_UNSET,
        summary_accumulated_path: str | None | object = CALL_INDEX_UNSET,
        final_summary_ready: bool | object = CALL_INDEX_UNSET,
        final_transcript_path: str | None | object = CALL_INDEX_UNSET,
        final_transcript_ready: bool | object = CALL_INDEX_UNSET,
    ) -> None:
        self.sink_for_namespace(routing.namespace).publish_call_index(
            routing,
            status,
            last_minute_index,
            finalized,
            STORAGE_MINUTE_WINDOW_SECONDS,
            FIREBASE_FLUSH_INTERVAL_SECONDS,
            final_summary_path=final_summary_path,
            summary_accumulated_path=summary_accumulated_path,
            final_summary_ready=final_summary_ready,
            final_transcript_path=final_transcript_path,
            final_transcript_ready=final_transcript_ready,
        )

    def upsert_room_session_links_on_start(self, routing: RoomRoutingContext) -> None:
        self.sink_for_namespace(routing.namespace).upsert_room_session_links_on_start(routing)

    def upload_minute_transcript(
        self, routing: RoomRoutingContext, payload: MinuteShardPayload
    ) -> None:
        self.sink_for_namespace(routing.namespace).upload_minute_transcript(routing, payload)

    def upload_json(self, routing: RoomRoutingContext, object_path: str, body: dict) -> None:
        self.sink_for_namespace(routing.namespace).upload_json(object_path, body)

    def fetch_json(self, routing: RoomRoutingContext, object_path: str) -> dict | None:
        return self.sink_for_namespace(routing.namespace).fetch_json(object_path)

    def fetch_agent_prompt(self, routing: RoomRoutingContext, agent_id: str) -> str | None:
        agent_key = str(agent_id or "").strip()
        if not agent_key:
            logger.warning(
                "agent prompt fallback: agent_id invalido room=%s session=%s",
                routing.room_name,
                routing.call_session_id,
            )
            return None
        sink = self.sink_for_namespace(routing.namespace)
        if not sink.enabled or sink.firestore_client is None:
            logger.warning(
                "agent prompt fallback: firestore indisponivel room=%s session=%s agent_id=%s",
                routing.room_name,
                routing.call_session_id,
                agent_key,
            )
            return None
        prompt_doc_path = build_agent_prompt_doc_path(routing.vertical, routing.slug, agent_key)
        try:
            snapshot = sink.firestore_client.document(prompt_doc_path).get()
            if not snapshot.exists:
                return None
            raw_payload = snapshot.to_dict()
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            prompt_raw = payload.get("prompt")
            if not isinstance(prompt_raw, str):
                logger.warning(
                    "agent prompt fallback: campo prompt invalido room=%s session=%s agent_id=%s path=%s",
                    routing.room_name,
                    routing.call_session_id,
                    agent_key,
                    prompt_doc_path,
                )
                return None
            prompt = prompt_raw.strip()
            if not prompt:
                logger.warning(
                    "agent prompt fallback: campo prompt vazio room=%s session=%s agent_id=%s path=%s",
                    routing.room_name,
                    routing.call_session_id,
                    agent_key,
                    prompt_doc_path,
                )
                return None
            logger.info(
                "agent prompt override: usando prompt do firebase room=%s session=%s agent_id=%s path=%s",
                routing.room_name,
                routing.call_session_id,
                agent_key,
                prompt_doc_path,
            )
            return prompt
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "agent prompt fallback: erro ao ler firestore room=%s session=%s agent_id=%s path=%s error=%s",
                routing.room_name,
                routing.call_session_id,
                agent_key,
                prompt_doc_path,
                exc,
            )
            return None


def db_pending_count() -> int:
    conn = read_db_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE status IN ('queued','processing')"
        ).fetchone()
        return int(row["total"] if row else 0)
    finally:
        conn.close()


def db_upsert_start(req: StartRequest, routing: RoomRoutingContext) -> None:
    now = utc_now_iso()
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO sessions(
                room_name, session_id, call_session_id, transcript_session_id, room_id, vertical, slug,
                firestore_doc_path, storage_base_path,
                started_at, metadata_json,
                state, room_end_received, last_chunk_at, created_at, updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, ?, ?)
            ON CONFLICT(room_name, session_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                call_session_id=excluded.call_session_id,
                transcript_session_id=excluded.transcript_session_id,
                room_id=excluded.room_id,
                vertical=excluded.vertical,
                slug=excluded.slug,
                firestore_doc_path=excluded.firestore_doc_path,
                storage_base_path=excluded.storage_base_path,
                metadata_json=COALESCE(excluded.metadata_json, sessions.metadata_json),
                last_chunk_at=COALESCE(sessions.last_chunk_at, excluded.last_chunk_at)
            """,
            (
                req.room_name,
                req.call_session_id,
                req.call_session_id,
                req.transcript_session_id,
                routing.room_id,
                routing.vertical,
                routing.slug,
                routing.firestore_doc_path,
                routing.storage_base_path,
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
                req.call_session_id,
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
    call_dir = SPOOL_DIR / safe_key(f"{meta.room_name}__{meta.call_session_id}") / safe_key(
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
            "SELECT 1 FROM sessions WHERE room_name = ? AND session_id = ? AND call_session_id = ?",
            (meta.room_name, meta.call_session_id, meta.call_session_id),
        ).fetchone()
        if not session:
            conn.execute("ROLLBACK")
            return "session_not_found"

        participant = conn.execute(
            """
            SELECT last_seq FROM participants
            WHERE room_name = ? AND session_id = ? AND participant_identity = ?
            """,
            (meta.room_name, meta.call_session_id, meta.participant_identity),
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
                    meta.call_session_id,
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
                    meta.call_session_id,
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
                meta.call_session_id,
                meta.participant_identity,
            ),
        )
        conn.execute(
            """
            UPDATE sessions
            SET last_chunk_at = ?,
                transcript_session_id = COALESCE(transcript_session_id, ?),
                updated_at = ?
            WHERE room_name = ? AND session_id = ?
            """,
            (now, meta.transcript_session_id, now, meta.room_name, meta.call_session_id),
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
            (now, utc_now_iso(), req.room_name, req.call_session_id, req.participant_identity),
        )
        conn.execute(
            """
            UPDATE sessions
            SET transcript_session_id = COALESCE(transcript_session_id, ?), updated_at=?
            WHERE room_name = ? AND session_id = ?
            """,
            (req.transcript_session_id, utc_now_iso(), req.room_name, req.call_session_id),
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
                ended_at=COALESCE(ended_at, ?),
                transcript_session_id=COALESCE(transcript_session_id, ?),
                updated_at=?
            WHERE room_name = ? AND session_id = ?
            """,
            (now, req.transcript_session_id, utc_now_iso(), req.room_name, req.call_session_id),
        )
        conn.execute(
            """
            UPDATE participants
            SET state='ended', ended_at=COALESCE(ended_at, ?), updated_at=?
            WHERE room_name = ? AND session_id = ? AND state='active'
            """,
            (now, utc_now_iso(), req.room_name, req.call_session_id),
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
            SELECT room_name, COALESCE(call_session_id, session_id) AS call_session_id, started_at, room_end_received
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


def db_mark_session_flushed(room_name: str, call_session_id: str, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE sessions
            SET last_firebase_flush_at=?, updated_at=?
            WHERE room_name=? AND (session_id=? OR call_session_id=?)
            """,
            (now_iso, now_iso, room_name, call_session_id, call_session_id),
        )
    finally:
        conn.close()


def db_get_session_row(room_name: str, call_session_id: str) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT room_name, session_id, call_session_id, transcript_session_id, room_id, vertical, slug,
                   firestore_doc_path, storage_base_path,
                   started_at, room_end_received, finalized_at
            FROM sessions
            WHERE room_name=? AND (session_id=? OR call_session_id=?)
            """,
            (room_name, call_session_id, call_session_id),
        ).fetchone()
    finally:
        conn.close()


def routing_context_from_session_row(row: sqlite3.Row) -> RoomRoutingContext:
    room_name = str(row["room_name"] or "")
    namespace_info = extract_room_namespace(room_name)
    namespace_value = namespace_info.namespace if namespace_info is not None else ""
    room_id = str(row["room_id"] or "").strip()
    call_session_id = str(row["call_session_id"] or row["session_id"] or "").strip()
    transcript_session_id = str(row["transcript_session_id"] or "").strip() or None
    vertical = str(row["vertical"] or "").strip()
    slug = str(row["slug"] or "").strip()
    firestore_doc_path = str(row["firestore_doc_path"] or "").strip()
    storage_base_path = str(row["storage_base_path"] or "").strip()

    if not room_name or not room_id or not call_session_id:
        raise RuntimeError("session routing context incompleto: room_name/room_id/call_session_id")
    if not vertical or not slug:
        raise RuntimeError("session routing context incompleto: vertical/slug")
    if not firestore_doc_path or not storage_base_path:
        raise RuntimeError("session routing context incompleto: firestore_doc_path/storage_base_path")
    if not namespace_value:
        raise RuntimeError("session routing context invalido: namespace ausente")

    return RoomRoutingContext(
        namespace=namespace_value,
        room_name=room_name,
        room_id=room_id,
        call_session_id=call_session_id,
        transcript_session_id=transcript_session_id,
        vertical=vertical,
        slug=slug,
        firestore_doc_path=firestore_doc_path,
        storage_base_path=storage_base_path,
    )
def db_get_done_chunks_for_session(room_name: str, call_session_id: str) -> list[sqlite3.Row]:
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
            (room_name, call_session_id),
        ).fetchall()
    finally:
        conn.close()


def build_minute_shards(
    room_name: str,
    call_session_id: str,
    transcript_session_id: str | None,
    storage_base_path: str,
    started_at: str,
    done_chunks: list[sqlite3.Row],
    minute_window_seconds: int,
    finalized: bool,
) -> list[MinuteShardPayload]:
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
        call_base = join_storage_path(storage_base_path, f"minutes/{minute_index:04d}")
        transcript_path = f"{call_base}/transcript.json"
        summary_path = f"{call_base}/summary.json"
        minute_lines = grouped[minute_index]
        raw_hash = hashlib.sha256(
            json.dumps(minute_lines, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        shards.append(
            MinuteShardPayload(
                room_name=room_name,
                call_session_id=call_session_id,
                transcript_session_id=transcript_session_id,
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
    room_name: str, call_session_id: str, minute_index: int
) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT *
            FROM minute_exports
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (room_name, call_session_id, minute_index),
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
                payload.call_session_id,
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


def db_upsert_summary_task(room_name: str, call_session_id: str, minute_index: int, now_iso: str) -> None:
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
            (room_name, call_session_id, minute_index, now_iso, now_iso, now_iso),
        )
    finally:
        conn.close()


def db_claim_summary_task(now_iso: str) -> sqlite3.Row | None:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            """
            SELECT room_name, session_id AS call_session_id, minute_index, retries
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
            (now_iso, task["room_name"], task["call_session_id"], task["minute_index"]),
        )
        conn.execute("COMMIT")
        return task
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_mark_summary_task_done(room_name: str, call_session_id: str, minute_index: int, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='done', updated_at=?, error_message=NULL
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (now_iso, room_name, call_session_id, minute_index),
        )
    finally:
        conn.close()


def db_mark_summary_task_error(
    room_name: str,
    call_session_id: str,
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
                call_session_id,
                minute_index,
            ),
        )
    finally:
        conn.close()


def db_get_session_minute_exports(room_name: str, call_session_id: str) -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT *
            FROM minute_exports
            WHERE room_name=? AND session_id=?
            ORDER BY minute_index ASC
            """,
            (room_name, call_session_id),
        ).fetchall()
    finally:
        conn.close()


def db_update_minute_export_summary_path(
    room_name: str, call_session_id: str, minute_index: int, summary_json_path: str, now_iso: str
) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE minute_exports
            SET summary_json_path=?, updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (summary_json_path, now_iso, room_name, call_session_id, minute_index),
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


def db_claim_room_finalization(room_name: str, session_id: str, now_iso: str) -> bool:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT 1
            FROM sessions
            WHERE room_name=? AND session_id=? AND finalized_at IS NULL AND state='room_ended'
            """,
            (room_name, session_id),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return False
        conn.execute(
            """
            UPDATE sessions
            SET state='finalizing', updated_at=?
            WHERE room_name=? AND session_id=? AND finalized_at IS NULL AND state='room_ended'
            """,
            (now_iso, room_name, session_id),
        )
        claimed = int(conn.total_changes) > 0
        conn.execute("COMMIT")
        return claimed
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_release_room_finalization(room_name: str, session_id: str, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE sessions
            SET state='room_ended', updated_at=?
            WHERE room_name=? AND session_id=? AND finalized_at IS NULL AND state='finalizing'
            """,
            (now_iso, room_name, session_id),
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
    request_retries: int
    retry_base_seconds: float

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
                OPENAI_REQUEST_RETRIES,
                OPENAI_REQUEST_RETRY_BASE_SECONDS,
            )
        return SummaryEngine(
            True,
            api_key,
            OPENAI_MODEL_MINUTE_SUMMARY,
            OPENAI_MODEL_ACCUMULATED_SUMMARY,
            OPENAI_MODEL_FINAL_SUMMARY,
            OPENAI_REQUEST_TIMEOUT_SECONDS,
            OPENAI_REQUEST_RETRIES,
            OPENAI_REQUEST_RETRY_BASE_SECONDS,
        )

    def _retry_delay_seconds(self, attempt: int) -> float:
        return min(8.0, self.retry_base_seconds * (2**attempt))

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
        max_attempts = self.request_retries + 1
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    body = exc.read().decode("utf-8", errors="ignore")
                    detail = body[:1200]
                except Exception:  # pylint: disable=broad-except
                    detail = ""
                should_retry = exc.code in (408, 409, 429, 500, 502, 503, 504)
                if should_retry and attempt < (max_attempts - 1):
                    delay = self._retry_delay_seconds(attempt)
                    logger.warning(
                        "OpenAI transient HTTP error model=%s status=%s attempt=%s/%s retry_in=%.1fs",
                        model,
                        exc.code,
                        attempt + 1,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"OpenAI request failed status={exc.code} model={model} detail={detail}"
                ) from exc
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                if attempt < (max_attempts - 1):
                    delay = self._retry_delay_seconds(attempt)
                    logger.warning(
                        "OpenAI request retry model=%s attempt=%s/%s error=%s retry_in=%.1fs",
                        model,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI request failed model={model} error={exc}") from exc
        else:
            raise RuntimeError(f"OpenAI request failed model={model} error=retry loop exhausted")
        text = extract_openai_output_text(payload)
        if not text:
            raise RuntimeError("OpenAI retornou resposta sem texto")
        return text

    def _request_json(
        self,
        kind: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        raw_output = self._request_text(model, system_prompt, user_prompt)
        return parse_and_validate_summary_output(kind, raw_output)

    def summarize_minute(
        self,
        minute_lines: list[dict],
        system_prompt: str | None = None,
    ) -> dict:
        minute_text = "\n".join(f"[{line['speaker']}] {line['text']}" for line in minute_lines)
        effective_prompt = build_effective_system_prompt(
            system_prompt,
            DEFAULT_SUMMARIZE_MINUTE_PROMPT,
            CONTRACT_SUFFIX_MINUTE,
        )
        return self._request_json(
            SUMMARY_KIND_MINUTE,
            self.minute_model,
            effective_prompt,
            f"Minuto da chamada:\n{minute_text}",
        )

    def merge_summaries(
        self,
        previous_summary: dict,
        minute_summary: dict,
        system_prompt: str | None = None,
    ) -> dict:
        effective_prompt = build_effective_system_prompt(
            system_prompt,
            DEFAULT_MERGE_SUMMARIES_PROMPT,
            CONTRACT_SUFFIX_ACCUMULATED,
        )
        previous_payload = (
            previous_summary if isinstance(previous_summary, dict) else default_accumulated_summary_payload()
        )
        minute_payload = minute_summary if isinstance(minute_summary, dict) else {}
        return self._request_json(
            SUMMARY_KIND_ACCUMULATED,
            self.accumulated_model,
            effective_prompt,
            "Resumo acumulado atual:\n"
            f"{json.dumps(previous_payload, ensure_ascii=True, indent=2)}\n\n"
            "Resumo novo do minuto:\n"
            f"{json.dumps(minute_payload, ensure_ascii=True, indent=2)}",
        )

    def finalize_summary(self, merged_summary: dict, system_prompt: str | None = None) -> dict:
        effective_prompt = build_effective_system_prompt(
            system_prompt,
            DEFAULT_FINALIZE_SUMMARY_PROMPT,
            CONTRACT_SUFFIX_FINAL,
        )
        merged_payload = merged_summary if isinstance(merged_summary, dict) else default_accumulated_summary_payload()
        return self._request_json(
            SUMMARY_KIND_FINAL,
            self.final_model,
            effective_prompt,
            "Resumo acumulado da chamada:\n"
            f"{json.dumps(merged_payload, ensure_ascii=True, indent=2)}",
        )

def session_summary_accumulated_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "summary/accumulated.json")


def session_final_summary_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_summary.json")


def session_final_transcript_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_transcript.json")


def build_final_transcript_payload(
    room_name: str,
    call_session_id: str,
    transcript_session_id: str | None,
    now_iso: str,
) -> dict:
    transcript, lines = db_get_room_aggregate(room_name, call_session_id)
    return {
        "room_name": room_name,
        "transcript_session_id": transcript_session_id,
        "call_session_id": call_session_id,
        "transcript": transcript,
        "lines": lines,
        "line_count": len(lines),
        "updated_at": now_iso,
    }


def publish_session_minute_exports(
    firebase_router: FirebaseRouter,
    room_name: str,
    call_session_id: str,
    now_iso: str,
    finalized: bool,
    summary_enabled: bool,
) -> int:
    session_row = db_get_session_row(room_name, call_session_id)
    if session_row is None:
        return -1
    routing = routing_context_from_session_row(session_row)
    done_chunks = db_get_done_chunks_for_session(room_name, call_session_id)
    shards = build_minute_shards(
        room_name=room_name,
        call_session_id=routing.call_session_id,
        transcript_session_id=routing.transcript_session_id,
        storage_base_path=routing.storage_base_path,
        started_at=session_row["started_at"],
        done_chunks=done_chunks,
        minute_window_seconds=STORAGE_MINUTE_WINDOW_SECONDS,
        finalized=finalized,
    )
    last_minute_index = shards[-1].minute_index if shards else -1
    for shard in shards:
        previous = db_get_minute_export(room_name, call_session_id, shard.minute_index)
        should_upload = (
            previous is None
            or previous["content_hash"] != shard.transcript_hash
            or (finalized and int(previous["finalized"] or 0) == 0)
        )
        if should_upload:
            firebase_router.upload_minute_transcript(routing, shard)
        if summary_enabled and (
            should_upload or previous is None or not previous["summary_json_path"]
        ):
            db_upsert_summary_task(room_name, call_session_id, shard.minute_index, now_iso)
        db_upsert_minute_export(shard, now_iso)

    firebase_router.publish_call_index(
        routing=routing,
        status="finalized" if finalized else "processing",
        last_minute_index=last_minute_index,
        finalized=finalized,
        final_summary_path=(
            session_final_summary_path(routing.storage_base_path)
            if finalized and summary_enabled
            else None
        ),
        summary_accumulated_path=session_summary_accumulated_path(routing.storage_base_path),
        final_summary_ready=False,
        final_transcript_path=None,
        final_transcript_ready=False,
    )
    return last_minute_index


async def flush_due_sessions(firebase_router: FirebaseRouter, summary_enabled: bool) -> None:
    now_iso = utc_now_iso()
    sessions = await asyncio.to_thread(
        db_get_sessions_due_for_flush, now_iso, FIREBASE_FLUSH_INTERVAL_SECONDS
    )
    for session_row in sessions:
        room_name = session_row["room_name"]
        call_session_id = session_row["call_session_id"]
        try:
            await asyncio.to_thread(
                publish_session_minute_exports,
                firebase_router,
                room_name,
                call_session_id,
                now_iso,
                False,
                summary_enabled,
            )
            await asyncio.to_thread(db_mark_session_flushed, room_name, call_session_id, now_iso)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "flush session failed room=%s session=%s error=%s",
                room_name,
                call_session_id,
                exc,
            )


async def finalize_entities(firebase_router: FirebaseRouter, summary_engine: SummaryEngine) -> None:
    participants = db_get_finalizable_participants()
    for participant in participants:
        room_name = participant["room_name"]
        call_session_id = participant["session_id"]
        participant_identity = participant["participant_identity"]
        db_mark_participant_finalized(room_name, call_session_id, participant_identity)
        logger.info(
            "participant finalized room=%s session=%s participant=%s",
            room_name,
            call_session_id,
            participant_identity,
        )

    rooms = db_get_finalizable_rooms()
    for room in rooms:
        room_name = room["room_name"]
        call_session_id = room["session_id"]
        now_iso = utc_now_iso()
        claimed = await asyncio.to_thread(
            db_claim_room_finalization, room_name, call_session_id, now_iso
        )
        if not claimed:
            logger.info(
                "room finalization skipped room=%s session=%s reason=already_claimed_or_finalized",
                room_name,
                call_session_id,
            )
            continue
        try:
            last_minute_index = await asyncio.to_thread(
                publish_session_minute_exports,
                firebase_router,
                room_name,
                call_session_id,
                now_iso,
                True,
                summary_engine.enabled,
            )
            routing = routing_context_from_session_row(room)
            final_transcript_path = session_final_transcript_path(routing.storage_base_path)
            final_transcript_payload = await asyncio.to_thread(
                build_final_transcript_payload,
                room_name,
                call_session_id,
                routing.transcript_session_id,
                now_iso,
            )
            await asyncio.to_thread(
                firebase_router.upload_json,
                routing,
                final_transcript_path,
                final_transcript_payload,
            )
            logger.info(
                "final transcript uploaded room=%s session=%s path=%s line_count=%s",
                room_name,
                call_session_id,
                final_transcript_path,
                final_transcript_payload["line_count"],
            )
            await asyncio.to_thread(
                firebase_router.publish_call_index,
                routing=routing,
                status="finalized",
                last_minute_index=last_minute_index,
                finalized=True,
                final_summary_path=(
                    session_final_summary_path(routing.storage_base_path)
                    if summary_engine.enabled
                    else None
                ),
                summary_accumulated_path=session_summary_accumulated_path(routing.storage_base_path),
                final_summary_ready=False,
                final_transcript_path=final_transcript_path,
                final_transcript_ready=True,
            )

            if summary_engine.enabled:
                await asyncio.to_thread(db_upsert_summary_task, room_name, call_session_id, -1, now_iso)
                logger.info(
                    "final summary task queued room=%s session=%s minute=-1",
                    room_name,
                    call_session_id,
                )
            db_mark_room_finalized(room_name, call_session_id)
            logger.info("room finalized room=%s session=%s", room_name, call_session_id)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "room finalization failed room=%s session=%s error=%s",
                room_name,
                call_session_id,
                exc,
            )
            await asyncio.to_thread(
                db_release_room_finalization, room_name, call_session_id, utc_now_iso()
            )


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
        call_session_id = task["call_session_id"]
        minute_index = int(task["minute_index"])
        retries = int(task["retries"])
        try:
            session_row = await asyncio.to_thread(db_get_session_row, room_name, call_session_id)
            if session_row is None:
                await asyncio.to_thread(
                    db_mark_summary_task_done, room_name, call_session_id, minute_index, now_iso
                )
                continue
            routing = routing_context_from_session_row(session_row)
            session_is_finalized = bool(session_row["finalized_at"])
            if minute_index >= 0:
                export_row = await asyncio.to_thread(
                    db_get_minute_export, room_name, call_session_id, minute_index
                )
                if export_row is None:
                    await asyncio.to_thread(
                        db_mark_summary_task_done, room_name, call_session_id, minute_index, now_iso
                    )
                    continue
                minute_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, routing, export_row["transcript_json_path"]
                )
                lines = minute_payload.get("lines", []) if isinstance(minute_payload, dict) else []
                minute_system_prompt = await asyncio.to_thread(
                    firebase_router.fetch_agent_prompt, routing, "stt_summarize_minute"
                )
                minute_summary = await asyncio.to_thread(
                    summary_engine.summarize_minute, lines, minute_system_prompt
                )
                summary_path = export_row["summary_json_path"] or (
                    join_storage_path(routing.storage_base_path, f"minutes/{minute_index:04d}/summary.json")
                )
                await asyncio.to_thread(
                    firebase_router.upload_json,
                    routing,
                    summary_path,
                    {
                        "room_name": room_name,
                        "transcript_session_id": routing.transcript_session_id,
                        "call_session_id": routing.call_session_id,
                        "minute_index": minute_index,
                        "summary": minute_summary,
                        "updated_at": now_iso,
                    },
                )
                await asyncio.to_thread(
                    db_update_minute_export_summary_path,
                    room_name,
                    call_session_id,
                    minute_index,
                    summary_path,
                    now_iso,
                )

                accumulated_path = session_summary_accumulated_path(routing.storage_base_path)
                previous_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, routing, accumulated_path
                )
                previous_summary: dict = default_accumulated_summary_payload()
                if isinstance(previous_payload, dict):
                    raw_previous_summary = previous_payload.get("summary")
                    if isinstance(raw_previous_summary, dict):
                        previous_summary = validate_accumulated_summary_payload(raw_previous_summary)
                merge_system_prompt = await asyncio.to_thread(
                    firebase_router.fetch_agent_prompt, routing, "stt_merge_summaries"
                )
                merged_summary = await asyncio.to_thread(
                    summary_engine.merge_summaries,
                    previous_summary,
                    minute_summary,
                    merge_system_prompt,
                )
                await asyncio.to_thread(
                    firebase_router.upload_json,
                    routing,
                    accumulated_path,
                    {
                        "room_name": room_name,
                        "transcript_session_id": routing.transcript_session_id,
                        "call_session_id": routing.call_session_id,
                        "last_minute_index": minute_index,
                        "summary": merged_summary,
                        "updated_at": now_iso,
                    },
                )
                await asyncio.to_thread(
                    firebase_router.publish_call_index,
                    routing=routing,
                    status="finalized" if session_is_finalized else "processing",
                    last_minute_index=minute_index,
                    finalized=session_is_finalized,
                    summary_accumulated_path=accumulated_path,
                    final_summary_ready=False if not session_is_finalized else CALL_INDEX_UNSET,
                    final_transcript_path=None if not session_is_finalized else CALL_INDEX_UNSET,
                    final_transcript_ready=False if not session_is_finalized else CALL_INDEX_UNSET,
                )
            else:
                accumulated_path = session_summary_accumulated_path(routing.storage_base_path)
                accumulated_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, routing, accumulated_path
                )
                merged_summary: dict | None = None
                if isinstance(accumulated_payload, dict):
                    raw_summary = accumulated_payload.get("summary")
                    if isinstance(raw_summary, dict):
                        merged_summary = validate_accumulated_summary_payload(raw_summary)
                if merged_summary is not None:
                    final_system_prompt = await asyncio.to_thread(
                        firebase_router.fetch_agent_prompt, routing, "stt_finalize_summary"
                    )
                    final_summary = await asyncio.to_thread(
                        summary_engine.finalize_summary,
                        merged_summary,
                        final_system_prompt,
                    )
                    final_path = session_final_summary_path(routing.storage_base_path)
                    await asyncio.to_thread(
                        firebase_router.upload_json,
                        routing,
                        final_path,
                        {
                            "room_name": room_name,
                            "transcript_session_id": routing.transcript_session_id,
                            "call_session_id": routing.call_session_id,
                            "summary": final_summary,
                            "updated_at": now_iso,
                        },
                    )
                    exports = await asyncio.to_thread(
                        db_get_session_minute_exports, room_name, call_session_id
                    )
                    final_transcript_path = session_final_transcript_path(routing.storage_base_path)
                    await asyncio.to_thread(
                        firebase_router.publish_call_index,
                        routing=routing,
                        status="finalized",
                        last_minute_index=exports[-1]["minute_index"] if exports else -1,
                        finalized=True,
                        final_summary_path=final_path,
                        summary_accumulated_path=accumulated_path,
                        final_summary_ready=True,
                        final_transcript_path=final_transcript_path,
                        final_transcript_ready=True,
                    )
                    logger.info(
                        "final summary uploaded room=%s session=%s path=%s",
                        room_name,
                        call_session_id,
                        final_path,
                    )

            await asyncio.to_thread(
                db_mark_summary_task_done, room_name, call_session_id, minute_index, utc_now_iso()
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception(
                "summary task failed room=%s session=%s minute=%s error=%s",
                room_name,
                call_session_id,
                minute_index,
                exc,
            )
            await asyncio.to_thread(
                db_mark_summary_task_error,
                room_name,
                call_session_id,
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
    logger.info("iniciando servico version=%s", APP_VERSION)
    init_db()
    logger.info("inicializando reconhecedor modelo=%s tipo=%s", MODEL_DIR, MODEL_TYPE)
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
    logger.info("servico pronto version=%s", APP_VERSION)
    yield
    logger.info("encerrando servico")
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
        "version": APP_VERSION,
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

    try:
        routing = await asyncio.to_thread(
            runtime.firebase_router.resolve_room_routing_context,
            payload.room_name,
            payload.call_session_id,
            payload.transcript_session_id,
        )
    except RoomRoutingError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc

    try:
        await asyncio.to_thread(runtime.firebase_router.upsert_room_session_links_on_start, routing)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(
            "session start failed while writing room_session_links room=%s transcript_session=%s call_session=%s error=%s",
            payload.room_name,
            payload.transcript_session_id,
            payload.call_session_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="firebase_start_write_failed") from exc

    await asyncio.to_thread(db_upsert_start, payload, routing)
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


















































