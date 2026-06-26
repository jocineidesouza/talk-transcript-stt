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
import unicodedata
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
APP_VERSION = os.environ.get("APP_VERSION", "0.1.4").strip() or "0.1.4"

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

SUMMARY_ENABLED = os.environ.get("SUMMARY_ENABLED", "false").lower() == "true"
SUMMARY_MODE = os.environ.get("SUMMARY_MODE", "rolling_json").strip().lower()
if SUMMARY_MODE not in {"rolling_json", "ata_progressiva"}:
    SUMMARY_MODE = "rolling_json"

SUMMARY_PROGRESSIVE_FINAL_SOURCE = (
    os.environ.get("SUMMARY_PROGRESSIVE_FINAL_SOURCE", "auto").strip().lower()
)
if SUMMARY_PROGRESSIVE_FINAL_SOURCE not in {"auto", "delta_only", "full_transcript"}:
    SUMMARY_PROGRESSIVE_FINAL_SOURCE = "auto"
try:
    SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS = max(
        1000, int(os.environ.get("SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS", "120000"))
    )
except ValueError:
    SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS = 120000
SUMMARY_PROVIDER = os.environ.get("SUMMARY_PROVIDER", "openrouter").strip().lower()
if SUMMARY_PROVIDER not in {"openrouter", "openai"}:
    SUMMARY_PROVIDER = "openrouter"

OPENAI_APIKEY_FILE = Path(os.environ.get("OPENAI_APIKEY_FILE", "/secrets/openai_apikey.json"))
OPENROUTER_APIKEY_FILE = Path(
    os.environ.get("OPENROUTER_APIKEY_FILE", "/secrets/openrouter_apikey.json")
)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
OPENROUTER_HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
OPENROUTER_X_TITLE = os.environ.get("OPENROUTER_X_TITLE", "").strip()

SUMMARY_MODEL_MINUTE = os.environ.get(
    "SUMMARY_MODEL_MINUTE", "openai/gpt-4.1-mini"
).strip()
SUMMARY_MODEL_ACCUMULATED = os.environ.get(
    "SUMMARY_MODEL_ACCUMULATED", "openai/gpt-4.1-mini"
).strip()
SUMMARY_MODEL_FINAL = os.environ.get(
    "SUMMARY_MODEL_FINAL", "openai/gpt-4.1-mini"
).strip()
SUMMARY_MODEL_FINAL_TEXT = os.environ.get(
    "SUMMARY_MODEL_FINAL_TEXT", "openai/gpt-4.1-mini"
).strip()
SUMMARY_FINAL_TEXT_FORMAT = os.environ.get("SUMMARY_FINAL_TEXT_FORMAT", "html").strip().lower()
if SUMMARY_FINAL_TEXT_FORMAT not in {"markdown", "html", "text"}:
    SUMMARY_FINAL_TEXT_FORMAT = "html"
SUMMARY_REQUEST_TIMEOUT_SECONDS = max(
    5, int(os.environ.get("SUMMARY_REQUEST_TIMEOUT_SECONDS", "300"))
)
SUMMARY_REQUEST_RETRIES = max(0, int(os.environ.get("SUMMARY_REQUEST_RETRIES", "2")))
SUMMARY_REQUEST_RETRY_BASE_SECONDS = max(
    0.1, float(os.environ.get("SUMMARY_REQUEST_RETRY_BASE_SECONDS", "1.5"))
)
SUMMARY_MAX_RETRIES = max(1, int(os.environ.get("SUMMARY_MAX_RETRIES", "3")))
SUMMARY_ACCUMULATED_MAX_ITEMS = max(
    1, int(os.environ.get("SUMMARY_ACCUMULATED_MAX_ITEMS", "40"))
)
SUMMARY_RECONCILE_INTERVAL_SECONDS = max(
    10, int(os.environ.get("SUMMARY_RECONCILE_INTERVAL_SECONDS", "60"))
)
SUMMARY_PROCESSING_STALE_SECONDS = max(
    30, int(os.environ.get("SUMMARY_PROCESSING_STALE_SECONDS", "300"))
)
SUMMARY_FINALIZATION_GRACE_SECONDS = max(
    0, int(os.environ.get("SUMMARY_FINALIZATION_GRACE_SECONDS", "180"))
)
SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES = (
    os.environ.get("SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES", "true").lower() == "true"
)
SUMMARY_FINAL_ENABLE_DETERMINISTIC_FALLBACK = (
    os.environ.get("SUMMARY_FINAL_ENABLE_DETERMINISTIC_FALLBACK", "true").lower() == "true"
)

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


class SummaryContractValidationError(RuntimeError):
    def __init__(self, kind: str, model: str, raw_output: str, message: str) -> None:
        self.kind = kind
        self.model = model
        self.raw_output = raw_output
        super().__init__(message)

DEFAULT_SUMMARIZE_MINUTE_PROMPT = """Voce e um assistente especializado em analisar trechos de chamadas corporativas em portugues do Brasil.

Entrada:
- Um TRECHO de conversa referente a um intervalo arbitrario (ex.: 30s, 1min, 3min, 5min).
- O trecho pode estar incompleto, com interrupcoes, mudancas de contexto, erros de transcricao e falas fragmentadas.

Tarefa:
- Extrair informacoes estruturadas APENAS do trecho recebido, de forma conservadora e confiavel.
- Identificar os assuntos efetivamente tratados no trecho.
- Associar cada fato, hipotese, decisao, pendencia, proximo passo e nota a um assunto principal.
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
- Detecte mudancas de assunto dentro do trecho.
- Crie nomes curtos, estaveis e objetivos para os assuntos.
- Nao criar assunto para ruido social isolado, como agradecimentos, despedidas, confirmacoes vazias ou fillers.
- Se o trecho for inutil/ruidoso, retornar arrays vazios e registrar 1 note curta.
- Tags curtas e especificas.
- Limites: text max 240 caracteres; tags entre 0 e 4 itens (preferencialmente curtas); maximo 20 itens por categoria.
"""

DEFAULT_MERGE_SUMMARIES_PROMPT = """Voce e um assistente responsavel por atualizar o estado acumulado de uma chamada corporativa em portugues do Brasil.

Entrada:
1) estado acumulado anterior em JSON
2) novo resumo de trecho (chunk) em JSON

Tarefa:
- Atualizar o estado acumulado usando apenas as informacoes recebidas.
- Manter uma memoria consolidada da chamada organizada por assuntos.
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

Regras especificas para assuntos:
- Mesclar assuntos equivalentes mesmo com nomes levemente diferentes.
- Preferir nomes curtos, claros, amigaveis, com acentuacao normal e reutilizaveis para assuntos recorrentes.
- Nao criar microassuntos desnecessarios.
- Se um novo item pertencer claramente a um assunto ja existente, inserir o item dentro desse assunto.
- Atualizar o summary do assunto para refletir o estado mais atual e mais util do tema.
- Cada item de facts, hypotheses, decisions, open_items, next_steps e notes deve ficar dentro de exatamente um topico principal.
- Um assunto pode ter status: active, open, resolved ou uncertain.
- Nao remover assunto antigo so porque ele nao apareceu no trecho atual.
- Se houver continuidade clara, manter o mesmo nome do assunto anterior.

Limites:
- text max 240 caracteres
- tags entre 0 e 4 itens
- maximo 20 itens por categoria
- Retornar arrays vazios quando categoria nao tiver itens.
"""

DEFAULT_FINALIZE_SUMMARY_PROMPT = """Voce e um assistente especializado em produzir ata final estruturada de chamadas corporativas em portugues do Brasil.

Entrada:
- Estado acumulado em JSON, com facts, hypotheses, decisions, open_items, next_steps e notes organizados dentro de cada topico.

Tarefa:
- Gerar uma ata final fiel, clara, conservadora e util para exibicao em frontend.
- Organizar a ata por assuntos tratados ao longo da chamada.
- Seguir estritamente o contrato JSON de saida fornecido separadamente.

Objetivo:
- Explicar os principais assuntos tratados.
- Mostrar o que foi decidido em cada assunto.
- Mostrar o que ficou em aberto.
- Mostrar proximos passos.
- Preservar pontos de baixa confianca ou ambiguidade sem trata-los como fato.

Regras obrigatorias:
- Use apenas informacoes do estado acumulado.
- Nao invente fatos, decisoes, prazos, responsaveis ou contexto extra.
- Nao transformar hypotheses em conclusoes.
- Nao transformar next_steps em fatos realizados.
- Nao transformar pending_items em decisoes.
- Consolidar redundancias e manter apenas o que for mais claro/relevante/confiavel.
- Nao repetir o mesmo conteudo em secoes diferentes sem necessidade.
- A secao topics deve refletir os principais assuntos efetivamente discutidos.
- Para cada assunto, gerar um resumo objetivo do que foi tratado.
- Para cada assunto, incluir decisoes, pending_items e next_steps quando existirem.
- Se nao houver decisao explicita global, inserir exatamente:
  "Nenhuma decisao explicita foi registrada."
- title deve ser um titulo curto, especifico da chamada e alinhado ao principal assunto discutido.
- conversation_types deve refletir o estado acumulado.
- Limites: text max 280 caracteres; tags entre 0 e 4 itens; maximo 30 itens por categoria.
- Retornar arrays vazios quando categoria nao tiver itens (exceto regra de global_decisions acima).
"""

DEFAULT_FINALIZE_SUMMARY_TEXT_PROMPT = """Você é responsável por transformar um resumo estruturado de reunião em uma ata profissional.

Você receberá um JSON consolidado de uma reunião finalizada.

Gere uma ata de reunião clara, profissional e fiel aos dados fornecidos.

Regras obrigatórias:
1. Use exclusivamente as informações presentes no JSON.
2. Não invente nenhuma informação.
3. Não adicione participantes, datas, horários, responsáveis, decisões, próximos passos ou pendências que não estejam no JSON.
4. Não transforme expectativas, hipóteses, comentários ou observações em decisões.
5. Não transforme tópicos mencionados em deliberações formais se isso não estiver explícito.
6. Se uma informação estiver ausente, use “Não informado” ou omita a seção quando a omissão deixar o documento mais limpo.
7. Preserve o sentido original de cada campo.
8. Use linguagem profissional, objetiva e adequada para registro corporativo.
9. Não mencione o JSON, o modelo, a IA ou o processo de geração.
10. Retorne somente o conteúdo final da ata no formato solicitado.
11. Não use bloco de código.
12. Não inclua explicações antes ou depois da ata.
13. É proibido iniciar com análise, justificativa, plano, comentário ou raciocínio.
14. Não escreva nenhum texto antes ou depois da ata.

Campos esperados no JSON:
- title
- updated_at
- room_name
- transcript_session_id
- call_session_id
- conversation_types
- executive_summary
- topics
- global_decisions
- global_pending_items
- global_next_steps
- additional_notes

Instruções de interpretação:
- title deve ser usado como título da reunião.
- conversation_types deve ser exibido como tipo de reunião, se existir.
- executive_summary deve compor o resumo executivo.
- topics devem ser listados como temas discutidos.
- decisions e global_decisions devem ser tratados como decisões somente se estiverem explicitamente registrados como decisões.
- pending_items e global_pending_items devem ser tratados como pendências.
- next_steps e global_next_steps devem ser tratados como próximos passos.
- additional_notes deve ser tratado como observações adicionais.
- tags podem ser usadas apenas como apoio organizacional, sem virar conteúdo novo.
- confidence pode ser omitido, salvo quando for importante indicar incerteza.
- room_name, transcript_session_id e call_session_id podem aparecer apenas na identificação se forem úteis; não use esses campos para inferir participantes, contexto ou conteúdo.

Estrutura obrigatória de conteúdo:

Ata de Reunião

1. Identificação

- Título: [usar title]
- Tipo de reunião: [usar conversation_types, se existir]
- Data: usar o campo "updated_at" do JSON para montar a data. Formato: "{dia} de {nome do mês} de {ano} ({dia da semana})". Se updated_at estiver ausente, usar "Não informado".

2. Objetivo da reunião
[usar executive_summary]

3. Principais assuntos discutidos
Para cada item em topics:
3.x [name]

[summary]

Só inserir Se existirem decisões do tópico:
Decisões:
[decisão]

Só inserir Se existirem pendências do tópico:
Pendências:
[pendência]

Só inserir Se existirem próximos passos do tópico:
Próximos passos:
[próximo passo]

Se existirem tags relevantes:
Tags: [tags]

4. Decisões registradas
Listar global_decisions.
Se não houver decisões explícitas, escrever:
Nenhuma decisão explícita foi registrada.

5. Pendências
Listar global_pending_items.
Se não houver pendências gerais, escrever:
Nenhuma pendência geral foi registrada.


6. Próximos passos
Listar global_next_steps.
Se não houver próximos passos gerais, escrever:
Nenhum próximo passo geral foi registrado.

7. Observações adicionais
Listar additional_notes, quando existirem e forem relevantes.

Se não houver observações adicionais, escrever:
Nenhuma observação adicional foi registrada.
"""

DEFAULT_PROGRESSIVE_ATA_PROMPT = """Você é um assistente responsável por manter uma ata progressiva de uma chamada corporativa em português do Brasil.

Entrada:
- Uma ATA ACUMULADA ANTERIOR em HTML fragment, que pode estar vazia.
- Um NOVO TRECHO TRANSCRITO da chamada.
- Metadados básicos da chamada.

Tarefa:
- Gerar uma nova versão completa da ata acumulada, incorporando somente informações sustentadas pelo novo trecho e preservando informações anteriores úteis.
- A ata deve ser adequada para uma daily operacional sempre que o conteúdo parecer daily/status report.
- Consolidar assuntos equivalentes e evitar duplicar itens quando o novo trecho continuar algo já registrado.

Regras obrigatórias:
- Use apenas a ata anterior e o novo trecho transcrito.
- Não invente participantes, papéis, datas, horários, responsáveis, status, decisões, prazos ou impedimentos.
- Não transforme intenção, hipótese, pergunta ou comentário ambíguo em decisão ou fato concluído.
- Se um item novo for continuação de uma tarefa já listada, atualize a linha existente em vez de criar outra.
- Se houver responsável claro, organize por responsável.
- Se não houver responsável claro, use "Não informado".
- Se não houver bloqueio explícito, informe que não há bloqueios críticos declarados.
- Preserve comentários gerais relevantes, mas remova ruído social sem valor operacional.
- Retorne somente o HTML fragment final, sem explicações e sem bloco de código.
- A primeira linha deve começar com <h1>.

Estrutura preferida:
<h1>ATA - Resumo - [data/intervalo quando informado]</h1>
<h2>Participantes</h2>
<ul>...</ul>
<h2>Resumo das Atividades</h2>
<table>
  <thead><tr><th>Responsável</th><th>Tarefa / Item</th><th>Status</th><th>Observação</th></tr></thead>
  <tbody>...</tbody>
</table>
<h2>Bloqueios / Impedimentos</h2>
<ul>...</ul>
<h2>Próximos Passos</h2>
<ul>...</ul>
<h2>Comentários Gerais</h2>
<ul>...</ul>
"""

DEFAULT_FINALIZE_PROGRESSIVE_ATA_PROMPT = """Você é um assistente responsável por produzir a versão final de uma ata progressiva em português do Brasil.

Entrada:
- A última ATA ACUMULADA em HTML fragment.
- Uma fonte final de transcrição que pode ser a transcrição completa ou apenas trechos ainda não incorporados.
- Metadados básicos da chamada.

Tarefa:
- Gerar a ata final em HTML fragment, limpa, consolidada e fiel ao conteúdo recebido.
- Revisar duplicações, corrigir continuidade entre trechos e manter o formato operacional de daily quando aplicável.

Regras obrigatórias:
- Use apenas a ata acumulada e a fonte de transcrição recebida.
- Não invente participantes, papéis, datas, horários, responsáveis, decisões, prazos ou impedimentos.
- Não reclassifique comentário ambíguo como decisão.
- Se a fonte final tiver apenas delta, não remova informação válida da ata acumulada só porque ela não aparece no delta.
- Se a fonte final for a transcrição completa, use-a para corrigir lacunas e duplicações da ata acumulada.
- Retorne somente o HTML fragment final, sem explicações e sem bloco de código.
- A primeira linha deve começar com <h1>.

Estrutura preferida:
<h1>ATA - Daily - [data/intervalo quando informado]</h1>
<h2>Participantes</h2>
<ul>...</ul>
<h2>Resumo das Atividades</h2>
<table>
  <thead><tr><th>Responsável</th><th>Tarefa / Item</th><th>Status</th><th>Observação</th></tr></thead>
  <tbody>...</tbody>
</table>
<h2>Bloqueios / Impedimentos</h2>
<ul>...</ul>
<h2>Próximos Passos</h2>
<ul>...</ul>
<h2>Comentários Gerais</h2>
<ul>...</ul>
"""

SUMMARY_PROGRESSIVE_ATA_PROMPT = os.environ.get("SUMMARY_PROGRESSIVE_ATA_PROMPT", "").strip()
SUMMARY_FINALIZE_PROGRESSIVE_ATA_PROMPT = os.environ.get(
    "SUMMARY_FINALIZE_PROGRESSIVE_ATA_PROMPT", ""
).strip()

SUMMARY_KIND_MINUTE = "minute"
SUMMARY_KIND_ACCUMULATED = "accumulated"
SUMMARY_KIND_FINAL = "final"

SUMMARY_ALLOWED_TYPES = {"tecnica", "executiva", "operacional", "comercial", "mista"}
SUMMARY_ALLOWED_CONFIDENCE = {"high", "medium", "low"}
SUMMARY_ALLOWED_TOPIC_STATUS_MINUTE = {"new", "continuing", "uncertain"}
SUMMARY_ALLOWED_TOPIC_STATUS_ACCUMULATED = {"active", "open", "resolved", "uncertain"}
SUMMARY_RESPONSE_FORMAT_NAME_BY_KIND = {
    SUMMARY_KIND_MINUTE: "summary_minute_v1",
    SUMMARY_KIND_ACCUMULATED: "summary_accumulated_v1",
    SUMMARY_KIND_FINAL: "summary_final_v1",
}

SUMMARY_STRING_ARRAY_SCHEMA = {"type": "array", "items": {"type": "string"}}
SUMMARY_CONVERSATION_TYPE_ENUM = ["tecnica", "executiva", "operacional", "comercial", "mista"]
SUMMARY_FINAL_TITLE_MIN_CHARS = 8
SUMMARY_FINAL_TITLE_MAX_CHARS = 120
SUMMARY_FINAL_TITLE_FALLBACK = "Ata da reunião"
SUMMARY_MINUTE_MAX_ITEMS = 20
SUMMARY_FINAL_MAX_ITEMS = 30
SUMMARY_ACCUMULATED_SECTION_KEYS = (
    "facts",
    "hypotheses",
    "decisions",
    "open_items",
    "next_steps",
    "notes",
)


def _topic_schema(topic_status: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": topic_status},
            "tags": SUMMARY_STRING_ARRAY_SCHEMA,
        },
        "required": ["name", "summary", "status", "tags"],
    }


def _summary_item_without_topic_schema(status_values: list[str], confidence_values: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "string", "enum": confidence_values},
            "status": {"type": "string", "enum": status_values},
            "tags": SUMMARY_STRING_ARRAY_SCHEMA,
        },
        "required": ["text", "confidence", "status", "tags"],
    }


def _summary_item_schema(status_values: list[str], confidence_values: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "string", "enum": confidence_values},
            "status": {"type": "string", "enum": status_values},
            "tags": SUMMARY_STRING_ARRAY_SCHEMA,
            "name": {"type": "string"},
        },
        "required": ["text", "confidence", "status", "tags", "name"],
    }


def _accumulated_topic_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "open", "resolved", "uncertain"]},
            "tags": SUMMARY_STRING_ARRAY_SCHEMA,
            "facts": {
                "type": "array",
                "items": _summary_item_without_topic_schema(
                    ["confirmed", "uncertain"], ["high", "medium", "low"]
                ),
            },
            "hypotheses": {
                "type": "array",
                "items": _summary_item_without_topic_schema(["uncertain"], ["medium", "low"]),
            },
            "decisions": {
                "type": "array",
                "items": _summary_item_without_topic_schema(["confirmed"], ["high", "medium", "low"]),
            },
            "open_items": {
                "type": "array",
                "items": _summary_item_without_topic_schema(["open"], ["high", "medium", "low"]),
            },
            "next_steps": {
                "type": "array",
                "items": _summary_item_without_topic_schema(["planned"], ["high", "medium", "low"]),
            },
            "notes": {
                "type": "array",
                "items": _summary_item_without_topic_schema(["uncertain", "info"], ["medium", "low"]),
            },
        },
        "required": [
            "name",
            "summary",
            "status",
            "tags",
            "facts",
            "hypotheses",
            "decisions",
            "open_items",
            "next_steps",
            "notes",
        ],
    }


def _final_item_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "tags": SUMMARY_STRING_ARRAY_SCHEMA,
        },
        "required": ["text", "confidence", "tags"],
    }


SUMMARY_SCHEMA_MINUTE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "chunk_type": {"type": "string", "enum": SUMMARY_CONVERSATION_TYPE_ENUM},
        "topics": {"type": "array", "items": _topic_schema(["new", "continuing", "uncertain"])},
        "facts": {
            "type": "array",
            "items": _summary_item_schema(["confirmed", "uncertain"], ["high", "medium", "low"]),
        },
        "hypotheses": {
            "type": "array",
            "items": _summary_item_schema(["uncertain"], ["medium", "low"]),
        },
        "decisions": {
            "type": "array",
            "items": _summary_item_schema(["confirmed"], ["high", "medium", "low"]),
        },
        "open_items": {
            "type": "array",
            "items": _summary_item_schema(["open"], ["high", "medium", "low"]),
        },
        "next_steps": {
            "type": "array",
            "items": _summary_item_schema(["planned"], ["high", "medium", "low"]),
        },
        "notes": {
            "type": "array",
            "items": _summary_item_schema(["uncertain", "info"], ["medium", "low"]),
        },
    },
    "required": [
        "chunk_type",
        "topics",
        "facts",
        "hypotheses",
        "decisions",
        "open_items",
        "next_steps",
        "notes",
    ],
}

SUMMARY_SCHEMA_ACCUMULATED = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "conversation_types": {
            "type": "array",
            "items": {"type": "string", "enum": SUMMARY_CONVERSATION_TYPE_ENUM},
        },
        "topics": {"type": "array", "items": _accumulated_topic_schema()},
    },
    "required": [
        "conversation_types",
        "topics",
    ],
}

SUMMARY_SCHEMA_FINAL = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {
            "type": "string",
            "minLength": SUMMARY_FINAL_TITLE_MIN_CHARS,
            "maxLength": SUMMARY_FINAL_TITLE_MAX_CHARS,
        },
        "conversation_types": {
            "type": "array",
            "items": {"type": "string", "enum": SUMMARY_CONVERSATION_TYPE_ENUM},
        },
        "executive_summary": {"type": "string"},
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "decisions": SUMMARY_STRING_ARRAY_SCHEMA,
                    "pending_items": SUMMARY_STRING_ARRAY_SCHEMA,
                    "next_steps": SUMMARY_STRING_ARRAY_SCHEMA,
                    "tags": SUMMARY_STRING_ARRAY_SCHEMA,
                },
                "required": ["name", "summary", "decisions", "pending_items", "next_steps", "tags"],
            },
        },
        "global_decisions": {"type": "array", "items": _final_item_schema()},
        "global_pending_items": {"type": "array", "items": _final_item_schema()},
        "global_next_steps": {"type": "array", "items": _final_item_schema()},
        "additional_notes": {"type": "array", "items": _final_item_schema()},
    },
    "required": [
        "title",
        "conversation_types",
        "executive_summary",
        "topics",
        "global_decisions",
        "global_pending_items",
        "global_next_steps",
        "additional_notes",
    ],
}

SUMMARY_SCHEMA_BY_KIND = {
    SUMMARY_KIND_MINUTE: SUMMARY_SCHEMA_MINUTE,
    SUMMARY_KIND_ACCUMULATED: SUMMARY_SCHEMA_ACCUMULATED,
    SUMMARY_KIND_FINAL: SUMMARY_SCHEMA_FINAL,
}


def build_openai_structured_output_schema(kind: str) -> dict:
    schema = SUMMARY_SCHEMA_BY_KIND.get(kind)
    if not schema:
        raise RuntimeError(f"tipo de resumo nao suportado: {kind}")
    return schema


def build_openai_response_format(kind: str) -> dict:
    schema_name = SUMMARY_RESPONSE_FORMAT_NAME_BY_KIND.get(kind)
    if not schema_name:
        raise RuntimeError(f"tipo de resumo nao suportado: {kind}")
    return {
        "type": "json_schema",
        "name": schema_name,
        "strict": True,
        "schema": build_openai_structured_output_schema(kind),
    }

CONTRACT_SUFFIX_MINUTE = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "- Regra obrigatoria de confidence por secao:\n"
    "  * hypotheses[].confidence: apenas low ou medium (nunca high)\n"
    "  * notes[].confidence: apenas low ou medium (nunca high)\n"
    "- Regra obrigatoria de topicos:\n"
    "  * Cada item em facts, hypotheses, decisions, open_items, next_steps e notes deve ter um campo name.\n"
    "  * Se vier campo topic, normalize para name.\n"
    "  * name deve corresponder ao name de um item em topics, exceto quando o trecho for inutil/ruidoso.\n"
    "- O formato final e controlado por schema estrito da API; siga as regras sem adicionar texto fora do JSON.\n"
    "- Para cada item em facts, hypotheses, decisions, open_items, next_steps e notes:\n"
    "  * item.name deve ser EXATAMENTE igual a um topics[].name.\n"
    "  * Nao use sinonimo, variacao, plural/singular ou caixa diferente.\n"
    "- Processo obrigatorio:\n"
    "  1) Gere topics primeiro.\n"
    "  2) Reutilize SOMENTE nomes existentes em topics.\n"
    "  3) Rode checklist de consistencia e corrija antes de responder.\n"
    "- Se nao conseguir mapear com seguranca, reduza saida (menos itens), nunca quebre a regra.\n"
)

CONTRACT_SUFFIX_ACCUMULATED = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "- Regra obrigatoria de topicos:\n"
    "  * facts, hypotheses, decisions, open_items, next_steps e notes ficam dentro de cada item de topics.\n"
    "  * Itens internos nao devem ter campo name nem topic.\n"
    "  * topics[].name deve ser curto, claro, amigavel, em portugues natural e pode usar acentos.\n"
    "  * Se houver continuidade clara, mantenha o mesmo topics[].name usado antes.\n"
    "- O formato final e controlado por schema estrito da API; siga as regras sem adicionar texto fora do JSON.\n"
)

CONTRACT_SUFFIX_FINAL = (
    "Saida:\n"
    "- Retorne APENAS JSON valido, sem markdown, sem explicacoes e sem texto adicional.\n"
    "- Nao adicionar campos fora do contrato.\n"
    "- executive_summary deve ser um texto curto e claro.\n"
    "- topics deve conter os principais assuntos da chamada, ja consolidados.\n"
    "- O formato final e controlado por schema estrito da API; siga as regras sem adicionar texto fora do JSON.\n"
)


FINAL_SUMMARY_TEXT_REQUIRED_PREFIX = "# Ata de Reunião"
FINAL_SUMMARY_TEXT_HTML_REQUIRED_PREFIX = "<h1"
FINAL_SUMMARY_TEXT_MAX_ATTEMPTS = 3


def normalize_final_summary_text_format(output_format: str) -> str:
    normalized = str(output_format or "").strip().lower()
    if normalized not in {"markdown", "html", "text"}:
        return "markdown"
    return normalized


def final_summary_text_format_instructions(output_format: str) -> str:
    normalized = normalize_final_summary_text_format(output_format)
    common = (
        "Regras obrigatorias de formato:\n"
        "- Retorne apenas o documento final da ata.\n"
        "- Nao inclua explicacoes, analise, justificativa, plano, raciocinio ou comentarios.\n"
        "- Nao use blocos de codigo.\n"
    )
    if normalized == "html":
        return common + (
            "- Formato de saida: HTML.\n"
            "- Gere um fragmento HTML simples, nao um documento completo.\n"
            "- A primeira linha deve comecar diretamente com <h1>Ata de Reuniao</h1> ou <h1>Ata de Reunião</h1>.\n"
            "- Use tags semanticas simples como h1, h2, h3, p, ul, li e strong.\n"
            "- Nao use Markdown; nao comece com #; nao use ```html.\n"
        )
    if normalized == "text":
        return common + (
            "- Formato de saida: texto simples.\n"
            "- Nao use Markdown estrutural, HTML ou blocos de codigo.\n"
            "- Comece diretamente pelo titulo Ata de Reuniao.\n"
        )
    return common + (
        "- Formato de saida: Markdown.\n"
        f"- A primeira linha da resposta deve ser exatamente: {FINAL_SUMMARY_TEXT_REQUIRED_PREFIX}\n"
        "- Use cabecalhos Markdown e listas Markdown quando necessario.\n"
    )


def build_final_summary_text_system_prompt(
    system_prompt: str | None,
    output_format: str,
) -> str:
    base_prompt = (
        system_prompt.strip()
        if isinstance(system_prompt, str) and system_prompt.strip()
        else DEFAULT_FINALIZE_SUMMARY_TEXT_PROMPT.strip()
    )
    return f"{base_prompt}\n\n{final_summary_text_format_instructions(output_format)}"


def validate_final_summary_text_output(raw_output: str, output_format: str = "markdown") -> str:
    normalized_format = normalize_final_summary_text_format(output_format)
    text = raw_output.strip() if isinstance(raw_output, str) else ""
    if not text:
        raise RuntimeError("ata textual final invalida: resposta vazia")
    if text.startswith("```"):
        raise RuntimeError("ata textual final invalida: resposta nao deve usar bloco de codigo")
    if normalized_format == "html":
        if text.startswith("#"):
            raise RuntimeError("ata textual final invalida: HTML nao deve iniciar com Markdown")
        if not text.lower().startswith(FINAL_SUMMARY_TEXT_HTML_REQUIRED_PREFIX):
            raise RuntimeError(
                "ata textual final invalida: resposta HTML deve iniciar diretamente com '<h1'"
            )
        return text
    if normalized_format == "text":
        return text
    if not text.startswith(FINAL_SUMMARY_TEXT_REQUIRED_PREFIX):
        raise RuntimeError(
            "ata textual final invalida: resposta deve iniciar exatamente com "
            f"{FINAL_SUMMARY_TEXT_REQUIRED_PREFIX!r}"
        )
    return text


def transcript_lines_to_text(lines: list[dict]) -> str:
    rendered: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        speaker = str(line.get("speaker") or line.get("participant_identity") or "Não informado").strip()
        text = str(line.get("text") or "").strip()
        if text:
            rendered.append(f"[{speaker}] {text}")
    return "\n".join(rendered)


def transcript_payload_to_text(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return ""
    lines = payload.get("lines")
    if isinstance(lines, list):
        return transcript_lines_to_text(lines)
    transcript = payload.get("transcript")
    return str(transcript or "").strip()


def validate_progressive_ata_output(raw_output: str) -> str:
    return validate_final_summary_text_output(raw_output, "html")


def progressive_ata_prompt(default_prompt: str, override_prompt: str) -> str:
    return override_prompt if override_prompt else default_prompt.strip()


def build_summary_metadata(
    routing: RoomRoutingContext,
    room_name: str,
    updated_at: str,
) -> dict:
    return {
        "namespace": routing.namespace,
        "vertical": routing.vertical,
        "slug": routing.slug,
        "room_id": routing.room_id,
        "room_name": room_name,
        "transcript_session_id": routing.transcript_session_id,
        "call_session_id": routing.call_session_id,
        "updated_at": updated_at,
    }


def build_progressive_delta_transcript_text(
    exports: list[sqlite3.Row],
    last_minute_index: int,
    fetch_minute_payload: Callable[[sqlite3.Row], dict | None],
) -> tuple[str, list[int]]:
    parts: list[str] = []
    included_minutes: list[int] = []
    for export in exports:
        minute_index = int(export["minute_index"])
        if minute_index < 0 or minute_index <= last_minute_index:
            continue
        payload = fetch_minute_payload(export)
        text = transcript_payload_to_text(payload)
        if not text:
            continue
        included_minutes.append(minute_index)
        parts.append(f"Trecho {minute_index:04d}:\n{text}")
    return "\n\n".join(parts), included_minutes


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
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise RuntimeError(f"{label}.tags deve ser array")
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
    if len(normalized) > 4:
        logger.warning(
            "summary tags normalized field=%s raw_len=%s kept=%s",
            label,
            len(normalized),
            4,
        )
        normalized = normalized[:4]
    return normalized


def validate_summary_item(
    item: Any,
    label: str,
    allowed_confidence: set[str],
    allowed_status: set[str] | None,
    max_text: int,
    include_status: bool,
    include_topic: bool,
    strict_confidence: bool = False,
) -> dict:
    if not isinstance(item, dict):
        raise RuntimeError(f"{label} deve ser objeto")
    expected_keys = {"text", "confidence", "tags"}
    if include_status:
        expected_keys.add("status")
    if include_topic:
        # Accept both for backward compatibility, normalize to `name`.
        expected_keys.update({"name", "topic"})
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
        if strict_confidence:
            raise RuntimeError(f"{label}.confidence invalido")
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
    if include_topic:
        raw_name = item.get("name")
        raw_topic = item.get("topic")
        resolved_name: str | None = None
        if isinstance(raw_name, str):
            resolved_name = raw_name.strip()
        elif isinstance(raw_topic, str):
            resolved_name = raw_topic.strip()
            logger.warning(
                "summary topic alias normalized field=%s alias=topic->name",
                label,
            )
        if not isinstance(resolved_name, str):
            raise RuntimeError(f"{label}.name deve ser string")
        if not resolved_name:
            raise RuntimeError(f"{label}.name nao pode ser vazio")
        if len(resolved_name) > 80:
            raise RuntimeError(f"{label}.name excede 80 caracteres")
        normalized["name"] = resolved_name
    return normalized


def validate_summary_list(
    value: Any,
    key: str,
    max_items: int,
    item_validator: Callable[[Any, str], dict],
    truncate_excess: bool = False,
) -> list[dict]:
    if not isinstance(value, list):
        raise RuntimeError(f"{key} deve ser array")
    if len(value) > max_items:
        if not truncate_excess:
            raise RuntimeError(f"{key} excede maximo de {max_items} itens")
        logger.warning(
            "summary list truncated field=%s raw_len=%s kept=%s",
            key,
            len(value),
            max_items,
        )
        value = value[:max_items]
    return [item_validator(item, f"{key}[{index}]") for index, item in enumerate(value)]


def validate_text_list(
    value: Any,
    key: str,
    max_items: int,
    max_len: int,
    truncate_excess: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{key} deve ser array")
    if len(value) > max_items:
        if not truncate_excess:
            raise RuntimeError(f"{key} excede maximo de {max_items} itens")
        logger.warning(
            "summary text list truncated field=%s raw_len=%s kept=%s",
            key,
            len(value),
            max_items,
        )
        value = value[:max_items]
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise RuntimeError(f"{key}[{index}] deve ser string")
        text = item.strip()
        if not text:
            raise RuntimeError(f"{key}[{index}] nao pode ser vazio")
        if len(text) > max_len:
            raise RuntimeError(f"{key}[{index}] excede {max_len} caracteres")
        normalized.append(text)
    return normalized


def validate_summary_topic(
    item: Any,
    label: str,
    allowed_status: set[str],
    max_summary: int,
) -> dict:
    if not isinstance(item, dict):
        raise RuntimeError(f"{label} deve ser objeto")
    expected_keys = {"name", "summary", "status", "tags"}
    assert_no_extra_keys(item, expected_keys, label)
    name = item.get("name")
    if not isinstance(name, str):
        raise RuntimeError(f"{label}.name deve ser string")
    name = name.strip()
    if not name:
        raise RuntimeError(f"{label}.name nao pode ser vazio")
    if len(name) > 80:
        raise RuntimeError(f"{label}.name excede 80 caracteres")
    summary = item.get("summary")
    if not isinstance(summary, str):
        raise RuntimeError(f"{label}.summary deve ser string")
    summary = summary.strip()
    if not summary:
        raise RuntimeError(f"{label}.summary nao pode ser vazio")
    if len(summary) > max_summary:
        raise RuntimeError(f"{label}.summary excede {max_summary} caracteres")
    status = item.get("status")
    if not isinstance(status, str) or status not in allowed_status:
        raise RuntimeError(f"{label}.status invalido")
    return {
        "name": name,
        "summary": summary,
        "status": status,
        "tags": validate_summary_tags(item.get("tags"), label),
    }


def validate_summary_topics(
    value: Any,
    key: str,
    max_items: int | None,
    allowed_status: set[str],
    max_summary: int,
) -> list[dict]:
    if not isinstance(value, list):
        raise RuntimeError(f"{key} deve ser array")
    if max_items is not None and len(value) > max_items:
        raise RuntimeError(f"{key} excede maximo de {max_items} itens")
    normalized: list[dict] = []
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        topic = validate_summary_topic(
            item,
            f"{key}[{index}]",
            allowed_status,
            max_summary,
        )
        if topic["name"] in seen_names:
            raise RuntimeError(f"{key}[{index}].name duplicado")
        seen_names.add(topic["name"])
        normalized.append(topic)
    return normalized


def validate_topic_reference(
    items: list[dict],
    key: str,
    topic_names: set[str],
    allow_unmapped_without_topics: bool = False,
) -> None:
    for index, item in enumerate(items):
        topic_name = item.get("name")
        if topic_name in topic_names:
            continue
        if allow_unmapped_without_topics and not topic_names:
            continue
        raise RuntimeError(f"{key}[{index}].name deve corresponder a um topics[].name")


def _canonical_topic_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    no_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    simplified = re.sub(r"[^\w\s]", " ", no_accents.casefold())
    return re.sub(r"\s+", " ", simplified).strip()


def reconcile_accumulated_topics_and_references(
    topics: list[dict],
    sections: dict[str, list[dict]],
) -> tuple[list[dict], dict[str, list[dict]]]:
    canonical_to_topic: dict[str, str] = {}
    ambiguous_keys: set[str] = set()
    for topic in topics:
        name = str(topic.get("name", "")).strip()
        if not name:
            continue
        canonical = _canonical_topic_key(name)
        if not canonical:
            continue
        existing = canonical_to_topic.get(canonical)
        if existing is None:
            canonical_to_topic[canonical] = name
            continue
        if existing != name:
            ambiguous_keys.add(canonical)
    for key in ambiguous_keys:
        canonical_to_topic.pop(key, None)

    topic_names = {str(topic.get("name", "")).strip() for topic in topics if isinstance(topic, dict)}
    inferred_topics: list[str] = []
    for section_key, items in sections.items():
        for item in items:
            raw_name = item.get("name")
            if not isinstance(raw_name, str):
                continue
            resolved_name = raw_name.strip()
            if resolved_name in topic_names:
                item["name"] = resolved_name
                continue
            canonical = _canonical_topic_key(resolved_name)
            mapped_name = canonical_to_topic.get(canonical) if canonical else None
            if mapped_name:
                item["name"] = mapped_name
                logger.warning(
                    "summary topic reference normalized section=%s from=%s to=%s",
                    section_key,
                    resolved_name,
                    mapped_name,
                )
                continue
            if resolved_name in inferred_topics:
                continue
            inferred_topics.append(resolved_name)

    if inferred_topics:
        for inferred_name in inferred_topics:
            topics.append(
                {
                    "name": inferred_name,
                    "summary": "Topico inferido automaticamente para manter consistencia referencial.",
                    "status": "uncertain",
                    "tags": ["auto_topic"],
                }
            )
        logger.warning(
            "summary inferred missing topics count=%s names=%s",
            len(inferred_topics),
            inferred_topics,
        )
    return topics, sections


def validate_final_topic_item(item: Any, label: str) -> dict:
    if not isinstance(item, dict):
        raise RuntimeError(f"{label} deve ser objeto")
    expected_keys = {"name", "summary", "decisions", "pending_items", "next_steps", "tags"}
    assert_no_extra_keys(item, expected_keys, label)
    name = item.get("name")
    if not isinstance(name, str):
        raise RuntimeError(f"{label}.name deve ser string")
    name = name.strip()
    if not name:
        raise RuntimeError(f"{label}.name nao pode ser vazio")
    if len(name) > 80:
        raise RuntimeError(f"{label}.name excede 80 caracteres")
    summary = item.get("summary")
    if not isinstance(summary, str):
        raise RuntimeError(f"{label}.summary deve ser string")
    summary = summary.strip()
    if not summary:
        raise RuntimeError(f"{label}.summary nao pode ser vazio")
    if len(summary) > 280:
        raise RuntimeError(f"{label}.summary excede 280 caracteres")
    return {
        "name": name,
        "summary": summary,
        "decisions": validate_text_list(
            item.get("decisions"), f"{label}.decisions", SUMMARY_FINAL_MAX_ITEMS, 280, truncate_excess=True
        ),
        "pending_items": validate_text_list(
            item.get("pending_items"), f"{label}.pending_items", SUMMARY_FINAL_MAX_ITEMS, 280, truncate_excess=True
        ),
        "next_steps": validate_text_list(
            item.get("next_steps"), f"{label}.next_steps", SUMMARY_FINAL_MAX_ITEMS, 280, truncate_excess=True
        ),
        "tags": validate_summary_tags(item.get("tags"), label),
    }


def default_accumulated_summary_payload() -> dict:
    return {
        "conversation_types": [],
        "topics": [],
    }


def validate_minute_summary_payload(payload: dict) -> dict:
    required_keys = {"chunk_type", "topics", "facts", "hypotheses", "decisions", "open_items", "next_steps", "notes"}
    assert_no_extra_keys(payload, required_keys, SUMMARY_KIND_MINUTE)
    for key in required_keys:
        if key not in payload:
            raise RuntimeError(f"{SUMMARY_KIND_MINUTE}: campo obrigatorio ausente: {key}")
    chunk_type = payload.get("chunk_type")
    if not isinstance(chunk_type, str) or chunk_type not in SUMMARY_ALLOWED_TYPES:
        raise RuntimeError("minute.chunk_type invalido")
    topics = validate_summary_topics(
        payload.get("topics"),
        "topics",
        None,
        SUMMARY_ALLOWED_TOPIC_STATUS_MINUTE,
        240,
    )
    topic_names = {topic["name"] for topic in topics}
    facts = validate_summary_list(
        payload.get("facts"),
        "facts",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item,
            label,
            SUMMARY_ALLOWED_CONFIDENCE,
            {"confirmed", "uncertain"},
            240,
            True,
            True,
        ),
        truncate_excess=True,
    )
    hypotheses = validate_summary_list(
        payload.get("hypotheses"),
        "hypotheses",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item, label, {"medium", "low"}, {"uncertain"}, 240, True, True
        ),
        truncate_excess=True,
    )
    decisions = validate_summary_list(
        payload.get("decisions"),
        "decisions",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item, label, SUMMARY_ALLOWED_CONFIDENCE, {"confirmed"}, 240, True, True
        ),
        truncate_excess=True,
    )
    open_items = validate_summary_list(
        payload.get("open_items"),
        "open_items",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item, label, SUMMARY_ALLOWED_CONFIDENCE, {"open"}, 240, True, True
        ),
        truncate_excess=True,
    )
    next_steps = validate_summary_list(
        payload.get("next_steps"),
        "next_steps",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item, label, SUMMARY_ALLOWED_CONFIDENCE, {"planned"}, 240, True, True
        ),
        truncate_excess=True,
    )
    notes = validate_summary_list(
        payload.get("notes"),
        "notes",
        SUMMARY_MINUTE_MAX_ITEMS,
        lambda item, label: validate_summary_item(
            item, label, {"medium", "low"}, {"uncertain", "info"}, 240, True, True
        ),
        truncate_excess=True,
    )
    validate_topic_reference(facts, "facts", topic_names, allow_unmapped_without_topics=True)
    validate_topic_reference(
        hypotheses,
        "hypotheses",
        topic_names,
        allow_unmapped_without_topics=True,
    )
    validate_topic_reference(
        decisions,
        "decisions",
        topic_names,
        allow_unmapped_without_topics=True,
    )
    validate_topic_reference(
        open_items,
        "open_items",
        topic_names,
        allow_unmapped_without_topics=True,
    )
    validate_topic_reference(
        next_steps,
        "next_steps",
        topic_names,
        allow_unmapped_without_topics=True,
    )
    validate_topic_reference(notes, "notes", topic_names, allow_unmapped_without_topics=True)
    return {
        "chunk_type": chunk_type,
        "topics": topics,
        "facts": facts,
        "hypotheses": hypotheses,
        "decisions": decisions,
        "open_items": open_items,
        "next_steps": next_steps,
        "notes": notes,
    }


def validate_accumulated_summary_payload(payload: dict) -> dict:
    required_keys = {
        "conversation_types",
        "topics",
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

    topics_raw = payload.get("topics")
    if not isinstance(topics_raw, list):
        raise RuntimeError("topics deve ser array")
    topics: list[dict] = []
    seen_topic_names: set[str] = set()
    for index, item in enumerate(topics_raw):
        label = f"topics[{index}]"
        if not isinstance(item, dict):
            raise RuntimeError(f"{label} deve ser objeto")
        expected_keys = {"name", "summary", "status", "tags", *SUMMARY_ACCUMULATED_SECTION_KEYS}
        assert_no_extra_keys(item, expected_keys, label)
        topic = validate_summary_topic(
            {
                "name": item.get("name"),
                "summary": item.get("summary"),
                "status": item.get("status"),
                "tags": item.get("tags"),
            },
            label,
            SUMMARY_ALLOWED_TOPIC_STATUS_ACCUMULATED,
            240,
        )
        if topic["name"] in seen_topic_names:
            raise RuntimeError(f"{label}.name duplicado")
        seen_topic_names.add(topic["name"])
        topic["facts"] = validate_summary_list(
            item.get("facts"),
            f"{label}.facts",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item,
                item_label,
                SUMMARY_ALLOWED_CONFIDENCE,
                {"confirmed", "uncertain"},
                240,
                True,
                False,
            ),
            truncate_excess=True,
        )
        topic["hypotheses"] = validate_summary_list(
            item.get("hypotheses"),
            f"{label}.hypotheses",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item,
                item_label,
                {"medium", "low"},
                {"uncertain"},
                240,
                True,
                False,
                True,
            ),
            truncate_excess=True,
        )
        topic["decisions"] = validate_summary_list(
            item.get("decisions"),
            f"{label}.decisions",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item, item_label, SUMMARY_ALLOWED_CONFIDENCE, {"confirmed"}, 240, True, False
            ),
            truncate_excess=True,
        )
        topic["open_items"] = validate_summary_list(
            item.get("open_items"),
            f"{label}.open_items",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item, item_label, SUMMARY_ALLOWED_CONFIDENCE, {"open"}, 240, True, False
            ),
            truncate_excess=True,
        )
        topic["next_steps"] = validate_summary_list(
            item.get("next_steps"),
            f"{label}.next_steps",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item, item_label, SUMMARY_ALLOWED_CONFIDENCE, {"planned"}, 240, True, False
            ),
            truncate_excess=True,
        )
        topic["notes"] = validate_summary_list(
            item.get("notes"),
            f"{label}.notes",
            SUMMARY_ACCUMULATED_MAX_ITEMS,
            lambda raw_item, item_label: validate_summary_item(
                raw_item,
                item_label,
                {"medium", "low"},
                {"uncertain", "info"},
                240,
                True,
                False,
                True,
            ),
            truncate_excess=True,
        )
        topics.append(topic)

    return {
        "conversation_types": conversation_types,
        "topics": topics,
    }


def validate_final_summary_payload(payload: dict) -> dict:
    required_keys = {
        "title",
        "conversation_types",
        "executive_summary",
        "topics",
        "global_decisions",
        "global_pending_items",
        "global_next_steps",
        "additional_notes",
    }
    assert_no_extra_keys(payload, required_keys, SUMMARY_KIND_FINAL)
    for key in required_keys:
        if key not in payload:
            raise RuntimeError(f"{SUMMARY_KIND_FINAL}: campo obrigatorio ausente: {key}")
    title = payload.get("title")
    if not isinstance(title, str):
        raise RuntimeError("final.title deve ser string")
    title = title.strip()
    if not title:
        raise RuntimeError("final.title nao pode ser vazio")
    if len(title) < SUMMARY_FINAL_TITLE_MIN_CHARS:
        raise RuntimeError(f"final.title deve ter ao menos {SUMMARY_FINAL_TITLE_MIN_CHARS} caracteres")
    if len(title) > SUMMARY_FINAL_TITLE_MAX_CHARS:
        raise RuntimeError(f"final.title excede {SUMMARY_FINAL_TITLE_MAX_CHARS} caracteres")
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
    executive_summary = payload.get("executive_summary")
    if not isinstance(executive_summary, str):
        raise RuntimeError("final.executive_summary deve ser string")
    executive_summary = executive_summary.strip()
    if not executive_summary:
        raise RuntimeError("final.executive_summary nao pode ser vazio")

    topics_raw = payload.get("topics")
    if not isinstance(topics_raw, list):
        raise RuntimeError("final.topics deve ser array")
    topics: list[dict] = []
    seen_topic_names: set[str] = set()
    for index, item in enumerate(topics_raw):
        topic = validate_final_topic_item(item, f"topics[{index}]")
        if topic["name"] in seen_topic_names:
            raise RuntimeError(f"topics[{index}].name duplicado")
        seen_topic_names.add(topic["name"])
        topics.append(topic)

    def _final_item(item: Any, label: str) -> dict:
        return validate_summary_item(
            item,
            label,
            SUMMARY_ALLOWED_CONFIDENCE,
            None,
            280,
            False,
            False,
        )

    return {
        "title": title,
        "conversation_types": conversation_types,
        "executive_summary": executive_summary,
        "topics": topics,
        "global_decisions": validate_summary_list(
            payload.get("global_decisions"),
            "global_decisions",
            SUMMARY_FINAL_MAX_ITEMS,
            _final_item,
            truncate_excess=True,
        ),
        "global_pending_items": validate_summary_list(
            payload.get("global_pending_items"),
            "global_pending_items",
            SUMMARY_FINAL_MAX_ITEMS,
            _final_item,
            truncate_excess=True,
        ),
        "global_next_steps": validate_summary_list(
            payload.get("global_next_steps"),
            "global_next_steps",
            SUMMARY_FINAL_MAX_ITEMS,
            _final_item,
            truncate_excess=True,
        ),
        "additional_notes": validate_summary_list(
            payload.get("additional_notes"),
            "additional_notes",
            SUMMARY_FINAL_MAX_ITEMS,
            _final_item,
            truncate_excess=True,
        ),
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


class AdminSummaryReprocessTarget(BaseModel):
    namespace: str = Field(min_length=1)
    vertical: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    call_session_id: str = Field(min_length=1)

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        namespace = str(value).strip()
        if namespace not in ALLOWED_LIVEKIT_NAMESPACES:
            raise ValueError("namespace nao permitido")
        return namespace

    @field_validator("vertical", "slug", "room_id", mode="before")
    @classmethod
    def normalize_non_empty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("campo obrigatorio")
        return text

    @field_validator("call_session_id")
    @classmethod
    def validate_call_session_id(cls, value: str) -> str:
        text = str(value).strip()
        if not text.startswith("RM_"):
            raise ValueError("call_session_id deve iniciar com RM_")
        return text

    @property
    def room_name(self) -> str:
        return f"{self.namespace}__{self.room_id}"


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
        final_summary_text_path: str | None | object = CALL_INDEX_UNSET,
        final_summary_text_ready: bool | object = CALL_INDEX_UNSET,
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
        if final_summary_text_path is not CALL_INDEX_UNSET:
            payload["final_summary_text_path"] = final_summary_text_path
        if final_summary_text_ready is not CALL_INDEX_UNSET:
            payload["final_summary_text_ready"] = final_summary_text_ready

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

    def upload_text(self, object_path: str, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        if not self.enabled or self.storage_bucket is None:
            return
        text_blob = self.storage_bucket.blob(object_path)
        text_blob.upload_from_string(body, content_type=content_type)

    def fetch_text(self, object_path: str) -> str | None:
        if not self.enabled or self.storage_bucket is None:
            return None
        blob = self.storage_bucket.blob(object_path)
        if not blob.exists():
            return None
        return blob.download_as_text()

    def delete_object(self, object_path: str) -> None:
        if not self.enabled or self.storage_bucket is None:
            return
        blob = self.storage_bucket.blob(object_path)
        if blob.exists():
            blob.delete()

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
        final_summary_text_path: str | None | object = CALL_INDEX_UNSET,
        final_summary_text_ready: bool | object = CALL_INDEX_UNSET,
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
            final_summary_text_path=final_summary_text_path,
            final_summary_text_ready=final_summary_text_ready,
        )

    def upsert_room_session_links_on_start(self, routing: RoomRoutingContext) -> None:
        self.sink_for_namespace(routing.namespace).upsert_room_session_links_on_start(routing)

    def upload_minute_transcript(
        self, routing: RoomRoutingContext, payload: MinuteShardPayload
    ) -> None:
        self.sink_for_namespace(routing.namespace).upload_minute_transcript(routing, payload)

    def upload_json(self, routing: RoomRoutingContext, object_path: str, body: dict) -> None:
        self.sink_for_namespace(routing.namespace).upload_json(object_path, body)

    def upload_text(
        self,
        routing: RoomRoutingContext,
        object_path: str,
        body: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        self.sink_for_namespace(routing.namespace).upload_text(
            object_path,
            body,
            content_type=content_type,
        )

    def fetch_json(self, routing: RoomRoutingContext, object_path: str) -> dict | None:
        return self.sink_for_namespace(routing.namespace).fetch_json(object_path)

    def fetch_text(self, routing: RoomRoutingContext, object_path: str) -> str | None:
        return self.sink_for_namespace(routing.namespace).fetch_text(object_path)

    def delete_object(self, routing: RoomRoutingContext, object_path: str) -> None:
        self.sink_for_namespace(routing.namespace).delete_object(object_path)

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


def assert_admin_target_matches_session(
    target: AdminSummaryReprocessTarget, row: sqlite3.Row
) -> RoomRoutingContext:
    routing = routing_context_from_session_row(row)
    mismatches: list[str] = []
    if routing.namespace != target.namespace:
        mismatches.append("namespace")
    if routing.vertical != target.vertical:
        mismatches.append("vertical")
    if routing.slug != target.slug:
        mismatches.append("slug")
    if routing.room_id != target.room_id:
        mismatches.append("room_id")
    if routing.call_session_id != target.call_session_id:
        mismatches.append("call_session_id")
    if mismatches:
        raise HTTPException(
            status_code=409,
            detail=f"session_target_mismatch:{','.join(mismatches)}",
        )
    return routing


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
        summary_path = (
            f"{call_base}/summary_text.txt"
            if SUMMARY_MODE == "ata_progressiva"
            else f"{call_base}/summary.json"
        )
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
            ORDER BY
              CASE WHEN minute_index < 0 THEN 1 ELSE 0 END ASC,
              minute_index ASC,
              updated_at ASC
            LIMIT 1
            """,
            (now_iso, SUMMARY_MAX_RETRIES),
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


def db_reschedule_summary_task(
    room_name: str,
    call_session_id: str,
    minute_index: int,
    next_attempt_at: str,
    error_message: str,
    now_iso: str,
) -> None:
    conn = read_db_connection()
    try:
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='pending',
                next_attempt_at=?,
                error_message=?,
                updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=?
            """,
            (
                next_attempt_at,
                error_message[:1000],
                now_iso,
                room_name,
                call_session_id,
                minute_index,
            ),
        )
    finally:
        conn.close()


def db_get_summary_task_rows(room_name: str, call_session_id: str) -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT minute_index, status, retries, next_attempt_at, error_message, updated_at
            FROM summary_tasks
            WHERE room_name=? AND session_id=?
            ORDER BY minute_index ASC
            """,
            (room_name, call_session_id),
        ).fetchall()
    finally:
        conn.close()


def db_schedule_final_summary_task(
    room_name: str,
    call_session_id: str,
    now_iso: str,
    force: bool = False,
) -> bool:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT status
            FROM summary_tasks
            WHERE room_name=? AND session_id=? AND minute_index=-1
            """,
            (room_name, call_session_id),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO summary_tasks(
                    room_name, session_id, minute_index,
                    status, retries, next_attempt_at,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, -1, 'pending', 0, ?, NULL, ?, ?)
                """,
                (room_name, call_session_id, now_iso, now_iso, now_iso),
            )
            conn.execute("COMMIT")
            return True
        status = str(row["status"])
        if not force and status in {"processing", "done"}:
            conn.execute("COMMIT")
            return False
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='pending',
                retries=0,
                next_attempt_at=?,
                error_message=NULL,
                updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=-1
            """,
            (now_iso, now_iso, room_name, call_session_id),
        )
        conn.execute("COMMIT")
        return int(conn.total_changes) > 0
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_get_finalized_sessions_for_summary_reconcile() -> list[sqlite3.Row]:
    conn = read_db_connection()
    try:
        return conn.execute(
            """
            SELECT room_name, COALESCE(call_session_id, session_id) AS call_session_id, finalized_at
            FROM sessions
            WHERE finalized_at IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()


def db_recover_stale_summary_tasks(now_iso: str, stale_seconds: int) -> int:
    now_dt = parse_iso_datetime(now_iso)
    cutoff_dt = now_dt - timedelta(seconds=stale_seconds)
    conn = read_db_connection()
    recovered = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        candidates = conn.execute(
            """
            SELECT room_name, session_id, minute_index, retries, updated_at
            FROM summary_tasks
            WHERE status='processing'
            """
        ).fetchall()
        for row in candidates:
            raw_updated_at = str(row["updated_at"] or "")
            try:
                updated_dt = parse_iso_datetime(raw_updated_at)
            except ValueError:
                continue
            if updated_dt > cutoff_dt:
                continue
            retries = int(row["retries"]) + 1
            next_attempt_iso = (now_dt + timedelta(seconds=15 * retries)).isoformat()
            changes_before = int(conn.total_changes)
            conn.execute(
                """
                UPDATE summary_tasks
                SET status='error',
                    retries=?,
                    error_message=?,
                    next_attempt_at=?,
                    updated_at=?
                WHERE room_name=? AND session_id=? AND minute_index=?
                  AND status='processing'
                """,
                (
                    retries,
                    f"task processing stale timeout>{stale_seconds}s",
                    next_attempt_iso,
                    now_iso,
                    row["room_name"],
                    row["session_id"],
                    int(row["minute_index"]),
                ),
            )
            if int(conn.total_changes) > changes_before:
                recovered += 1
        conn.execute("COMMIT")
        return recovered
    except Exception:
        conn.execute("ROLLBACK")
        raise
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


def db_force_finalize_session_for_reprocess(room_name: str, call_session_id: str, now_iso: str) -> None:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE sessions
            SET room_end_received=1,
                state='finalized',
                ended_at=COALESCE(ended_at, ?),
                finalized_at=COALESCE(finalized_at, ?),
                updated_at=?
            WHERE room_name=? AND (session_id=? OR call_session_id=?)
            """,
            (now_iso, now_iso, now_iso, room_name, call_session_id, call_session_id),
        )
        conn.execute(
            """
            UPDATE participants
            SET state='ended',
                ended_at=COALESCE(ended_at, ?),
                finalized_at=COALESCE(finalized_at, ?),
                updated_at=?
            WHERE room_name=? AND session_id=?
            """,
            (now_iso, now_iso, now_iso, room_name, call_session_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def db_reset_summary_reprocess_state(room_name: str, call_session_id: str, now_iso: str) -> dict[str, int]:
    conn = read_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        export_rows = conn.execute(
            """
            SELECT minute_index
            FROM minute_exports
            WHERE room_name=? AND session_id=? AND minute_index>=0
            ORDER BY minute_index ASC
            """,
            (room_name, call_session_id),
        ).fetchall()
        minute_indexes = [int(row["minute_index"]) for row in export_rows]

        conn.execute(
            """
            DELETE FROM summary_tasks
            WHERE room_name=? AND session_id=? AND minute_index>=0
              AND minute_index NOT IN (
                SELECT minute_index
                FROM minute_exports
                WHERE room_name=? AND session_id=? AND minute_index>=0
              )
            """,
            (room_name, call_session_id, room_name, call_session_id),
        )
        conn.execute(
            """
            UPDATE minute_exports
            SET summary_json_path=NULL, updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index>=0
            """,
            (now_iso, room_name, call_session_id),
        )
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='pending',
                retries=0,
                next_attempt_at=?,
                error_message=NULL,
                updated_at=?
            WHERE room_name=? AND session_id=?
            """,
            (now_iso, now_iso, room_name, call_session_id),
        )
        conn.execute(
            """
            INSERT INTO summary_tasks(
                room_name, session_id, minute_index,
                status, retries, next_attempt_at,
                error_message, created_at, updated_at
            )
            SELECT me.room_name, me.session_id, me.minute_index,
                   'pending', 0, ?, NULL, ?, ?
            FROM minute_exports me
            LEFT JOIN summary_tasks st
              ON st.room_name=me.room_name
             AND st.session_id=me.session_id
             AND st.minute_index=me.minute_index
            WHERE me.room_name=? AND me.session_id=? AND me.minute_index>=0
              AND st.minute_index IS NULL
            """,
            (now_iso, now_iso, now_iso, room_name, call_session_id),
        )
        conn.execute(
            """
            INSERT INTO summary_tasks(
                room_name, session_id, minute_index,
                status, retries, next_attempt_at,
                error_message, created_at, updated_at
            )
            SELECT ?, ?, -1, 'pending', 0, ?, NULL, ?, ?
            WHERE NOT EXISTS (
                SELECT 1
                FROM summary_tasks
                WHERE room_name=? AND session_id=? AND minute_index=-1
            )
            """,
            (
                room_name,
                call_session_id,
                now_iso,
                now_iso,
                now_iso,
                room_name,
                call_session_id,
            ),
        )
        conn.execute(
            """
            UPDATE summary_tasks
            SET status='pending',
                retries=0,
                next_attempt_at=?,
                error_message=NULL,
                updated_at=?
            WHERE room_name=? AND session_id=? AND minute_index=-1
            """,
            (now_iso, now_iso, room_name, call_session_id),
        )
        conn.execute("COMMIT")
        return {"minute_exports": len(minute_indexes)}
    except Exception:
        conn.execute("ROLLBACK")
        raise
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


def summary_provider_apikey_file(provider: str) -> Path:
    if provider == "openai":
        return OPENAI_APIKEY_FILE
    return OPENROUTER_APIKEY_FILE


def summary_provider_base_url(provider: str) -> str:
    if provider == "openai":
        return OPENAI_BASE_URL.rstrip("/")
    return OPENROUTER_BASE_URL.rstrip("/")


def summary_provider_extra_headers(provider: str) -> dict[str, str]:
    if provider != "openrouter":
        return {}
    headers: dict[str, str] = {}
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE
    return headers


def load_summary_api_key(provider: str) -> str | None:
    if not SUMMARY_ENABLED:
        return None
    apikey_file = summary_provider_apikey_file(provider)
    if not apikey_file.is_file():
        logger.warning(
            "SUMMARY_ENABLED=true, mas arquivo de secret nao encontrado provider=%s file=%s",
            provider,
            apikey_file,
        )
        return None
    try:
        raw = json.loads(apikey_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("arquivo de secret invalido provider=%s file=%s", provider, apikey_file)
        return None
    api_key = str(raw.get("api_key", "")).strip() if isinstance(raw, dict) else ""
    if not api_key:
        logger.warning(
            "arquivo de secret sem campo api_key valido provider=%s file=%s",
            provider,
            apikey_file,
        )
        return None
    return api_key


def extract_summary_output_text(payload: dict) -> str:
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


def extract_summary_refusal_reason(payload: dict) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            refusal = part.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
            if part.get("type") == "refusal":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                return "model refusal"
    return ""


@dataclass(frozen=True)
class SummaryEngine:
    enabled: bool
    provider: str
    api_key: str
    base_url: str
    extra_headers: dict[str, str]
    minute_model: str
    accumulated_model: str
    final_model: str
    final_text_model: str
    timeout_seconds: int
    request_retries: int
    retry_base_seconds: float

    @staticmethod
    def create() -> "SummaryEngine":
        provider = SUMMARY_PROVIDER
        api_key = load_summary_api_key(provider)
        base_url = summary_provider_base_url(provider)
        extra_headers = summary_provider_extra_headers(provider)
        if not api_key:
            return SummaryEngine(
                False,
                provider,
                "",
                base_url,
                extra_headers,
                SUMMARY_MODEL_MINUTE,
                SUMMARY_MODEL_ACCUMULATED,
                SUMMARY_MODEL_FINAL,
                SUMMARY_MODEL_FINAL_TEXT,
                SUMMARY_REQUEST_TIMEOUT_SECONDS,
                SUMMARY_REQUEST_RETRIES,
                SUMMARY_REQUEST_RETRY_BASE_SECONDS,
            )
        return SummaryEngine(
            True,
            provider,
            api_key,
            base_url,
            extra_headers,
            SUMMARY_MODEL_MINUTE,
            SUMMARY_MODEL_ACCUMULATED,
            SUMMARY_MODEL_FINAL,
            SUMMARY_MODEL_FINAL_TEXT,
            SUMMARY_REQUEST_TIMEOUT_SECONDS,
            SUMMARY_REQUEST_RETRIES,
            SUMMARY_REQUEST_RETRY_BASE_SECONDS,
        )

    def _retry_delay_seconds(self, attempt: int) -> float:
        return min(8.0, self.retry_base_seconds * (2**attempt))

    def _request_text(
        self,
        kind: str | None,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
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
        if kind:
            request_body["text"] = {
                "format": build_openai_response_format(kind),
            }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)
        req = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(request_body).encode("utf-8"),
            method="POST",
            headers=headers,
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
                        "Summary request transient HTTP error provider=%s model=%s status=%s attempt=%s/%s retry_in=%.1fs",
                        self.provider,
                        model,
                        exc.code,
                        attempt + 1,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Summary request failed provider={self.provider} status={exc.code} model={model} detail={detail}"
                ) from exc
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                if attempt < (max_attempts - 1):
                    delay = self._retry_delay_seconds(attempt)
                    logger.warning(
                        "Summary request retry provider=%s model=%s attempt=%s/%s error=%s retry_in=%.1fs",
                        self.provider,
                        model,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Summary request failed provider={self.provider} model={model} error={exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"Summary request failed provider={self.provider} model={model} error=retry loop exhausted"
            )
        refusal = extract_summary_refusal_reason(payload)
        if refusal:
            raise RuntimeError(
                f"Summary refusal provider={self.provider} model={model} kind={kind} detail={refusal[:800]}"
            )
        text = extract_summary_output_text(payload)
        if not text:
            raise RuntimeError(
                f"Summary provider={self.provider} retornou resposta sem texto model={model} kind={kind}"
            )
        return text

    def _request_json(
        self,
        kind: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        raw_output = self._request_text(kind, model, system_prompt, user_prompt)
        try:
            return parse_and_validate_summary_output(kind, raw_output)
        except RuntimeError as exc:
            raise SummaryContractValidationError(kind, model, raw_output, str(exc)) from exc

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

    def finalize_summary_text(
        self,
        final_summary: dict,
        output_format: str,
        system_prompt: str | None = None,
    ) -> str:
        normalized_format = normalize_final_summary_text_format(output_format)
        effective_prompt = build_final_summary_text_system_prompt(system_prompt, normalized_format)
        summary_payload = final_summary if isinstance(final_summary, dict) else {}
        user_prompt = (
            "Gere a ata textual final a partir do JSON abaixo.\n"
            f"Formato de saida esperado: {normalized_format}.\n"
            "Retorne apenas o documento final em texto, sem blocos de codigo.\n\n"
            "JSON do resumo final:\n"
            f"{json.dumps(summary_payload, ensure_ascii=True, indent=2)}"
        )
        request_prompt = user_prompt
        first_error: RuntimeError | None = None
        last_error: RuntimeError | None = None
        for attempt in range(FINAL_SUMMARY_TEXT_MAX_ATTEMPTS):
            raw_output = self._request_text(
                kind=None,
                model=self.final_text_model,
                system_prompt=effective_prompt,
                user_prompt=request_prompt,
            )
            try:
                return validate_final_summary_text_output(raw_output, normalized_format)
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
                last_error = exc
                if attempt >= FINAL_SUMMARY_TEXT_MAX_ATTEMPTS - 1:
                    break
                request_prompt = (
                    f"A resposta anterior foi inválida: {exc}.\n"
                    "Gere novamente obedecendo estritamente as regras obrigatorias de formato.\n"
                    "Não inclua análise, justificativa, plano, raciocínio, comentários ou blocos de código.\n\n"
                    f"{final_summary_text_format_instructions(normalized_format)}\n"
                    f"{user_prompt}"
                )
        raise RuntimeError(str(last_error or first_error)) from first_error

    def update_progressive_ata(
        self,
        previous_ata_text: str,
        minute_lines: list[dict],
        metadata: dict,
        system_prompt: str | None = None,
    ) -> str:
        effective_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else progressive_ata_prompt(DEFAULT_PROGRESSIVE_ATA_PROMPT, SUMMARY_PROGRESSIVE_ATA_PROMPT)
        )
        minute_text = transcript_lines_to_text(minute_lines)
        user_prompt = (
            "Metadados da chamada:\n"
            f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
            "ATA ACUMULADA ANTERIOR:\n"
            f"{previous_ata_text.strip() if previous_ata_text else '(vazia)'}\n\n"
            "NOVO TRECHO TRANSCRITO:\n"
            f"{minute_text if minute_text else '(sem falas úteis)'}\n\n"
            "Retorne a nova ata acumulada completa em HTML fragment."
        )
        raw_output = self._request_text(
            kind=None,
            model=self.accumulated_model,
            system_prompt=effective_prompt,
            user_prompt=user_prompt,
        )
        return validate_progressive_ata_output(raw_output)

    def finalize_progressive_ata(
        self,
        accumulated_ata_text: str,
        final_source_text: str,
        final_source_mode: str,
        metadata: dict,
        system_prompt: str | None = None,
    ) -> str:
        effective_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else progressive_ata_prompt(
                DEFAULT_FINALIZE_PROGRESSIVE_ATA_PROMPT,
                SUMMARY_FINALIZE_PROGRESSIVE_ATA_PROMPT,
            )
        )
        source_label = (
            "transcrição completa"
            if final_source_mode == "full_transcript"
            else "trechos ainda não incorporados"
        )
        user_prompt = (
            "Metadados da chamada:\n"
            f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
            "ÚLTIMA ATA ACUMULADA:\n"
            f"{accumulated_ata_text.strip() if accumulated_ata_text else '(vazia)'}\n\n"
            f"FONTE FINAL ({source_label}):\n"
            f"{final_source_text.strip() if final_source_text else '(sem trechos adicionais)'}\n\n"
            "Retorne a ata final completa em HTML fragment."
        )
        raw_output = self._request_text(
            kind=None,
            model=self.final_text_model,
            system_prompt=effective_prompt,
            user_prompt=user_prompt,
        )
        return validate_progressive_ata_output(raw_output)

def session_summary_accumulated_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "summary/accumulated.json")


def session_progressive_ata_accumulated_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "summary/accumulated.txt")


def session_progressive_ata_meta_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "summary/accumulated_meta.json")


def minute_progressive_ata_summary_path(storage_base_path: str, minute_index: int) -> str:
    return join_storage_path(storage_base_path, f"minutes/{minute_index:04d}/summary_text.txt")


def session_final_summary_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_summary.json")


def session_final_summary_temp_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_summary_temp.json")


def session_final_transcript_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_transcript.json")


def session_final_summary_text_path(storage_base_path: str) -> str:
    return join_storage_path(storage_base_path, "final/final_summary_text.txt")


def build_final_summary_temp_payload(
    room_name: str,
    transcript_session_id: str | None,
    call_session_id: str,
    model: str,
    kind: str,
    error: str,
    raw_output: str,
    updated_at: str,
) -> dict:
    return {
        "room_name": room_name,
        "transcript_session_id": transcript_session_id,
        "call_session_id": call_session_id,
        "model": model,
        "kind": kind,
        "error": error,
        "raw_output": raw_output,
        "updated_at": updated_at,
    }


def _minutes_label(minutes: list[int]) -> str:
    unique = sorted({int(minute) for minute in minutes})
    return ", ".join(str(minute) for minute in unique)


def _truncate_final_title(title: str) -> str:
    clean = title.strip()
    if len(clean) <= SUMMARY_FINAL_TITLE_MAX_CHARS:
        return clean
    return clean[: SUMMARY_FINAL_TITLE_MAX_CHARS - 3].rstrip() + "..."


def _build_deterministic_final_title(accumulated_summary: dict, missing_minutes: list[int]) -> str:
    topic_names: list[str] = []
    seen: set[str] = set()
    for topic in accumulated_summary.get("topics", []):
        if not isinstance(topic, dict):
            continue
        name = str(topic.get("name", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        topic_names.append(name)
        if len(topic_names) >= 2:
            break

    if len(topic_names) >= 2:
        base_title = f"Resumo Executivo: {topic_names[0]} e {topic_names[1]}"
    elif topic_names:
        base_title = f"Resumo Executivo: {topic_names[0]}"
    else:
        conversation_types_raw = accumulated_summary.get("conversation_types")
        conversation_type = ""
        if isinstance(conversation_types_raw, list):
            for item in conversation_types_raw:
                if isinstance(item, str) and item in SUMMARY_ALLOWED_TYPES:
                    conversation_type = item
                    break
        type_label_map = {
            "tecnica": "Tecnico",
            "executiva": "Executivo",
            "operacional": "Operacional",
            "comercial": "Comercial",
            "mista": "Misto",
        }
        type_label = type_label_map.get(conversation_type)
        base_title = (
            f"Resumo {type_label} da Chamada" if type_label else SUMMARY_FINAL_TITLE_FALLBACK
        )

    if missing_minutes:
        base_title = f"Documento Parcial: {base_title}"
    return _truncate_final_title(base_title)


def _build_degradation_additional_note(missing_minutes: list[int], reasons: list[str]) -> dict:
    minute_label = _minutes_label(missing_minutes)
    reason = reasons[0] if reasons else "houve falha na consolidacao total da chamada"
    text = f"Ata parcial: minutos nao processados [{minute_label}]; motivo: {reason}."
    if len(text) > 280:
        text = text[:277].rstrip() + "..."
    return {
        "text": text,
        "confidence": "medium",
        "tags": ["degradacao", "ata_parcial", "minutos"],
    }


def inject_degradation_disclosure(
    final_summary: dict,
    missing_minutes: list[int],
    reasons: list[str],
) -> dict:
    if not missing_minutes:
        return final_summary
    summary = dict(final_summary)
    minute_label = _minutes_label(missing_minutes)
    disclosure = f"Documento parcial: minutos sem consolidacao completa [{minute_label}]."
    executive_summary = str(summary.get("executive_summary", "")).strip()
    if disclosure not in executive_summary:
        executive_summary = (
            f"{disclosure} {executive_summary}".strip() if executive_summary else disclosure
        )
    summary["executive_summary"] = executive_summary
    notes = summary.get("additional_notes")
    if not isinstance(notes, list):
        notes = []
    note_item = _build_degradation_additional_note(missing_minutes, reasons)
    if note_item["text"] not in {str(item.get("text")) for item in notes if isinstance(item, dict)}:
        notes.append(note_item)
    if len(notes) > SUMMARY_FINAL_MAX_ITEMS:
        notes = notes[:SUMMARY_FINAL_MAX_ITEMS]
    summary["additional_notes"] = notes
    return validate_final_summary_payload(summary)


def _map_minute_topic_status_to_accumulated(status: str) -> str:
    if status in {"new", "continuing"}:
        return "active"
    return "uncertain"


def build_accumulated_from_minute_summaries(minute_summaries: list[dict]) -> dict:
    accumulated = default_accumulated_summary_payload()
    seen_types: set[str] = set()
    topic_index: dict[str, int] = {}
    seen_items: dict[str, set[tuple[str, str, str]]] = {
        "facts": set(),
        "hypotheses": set(),
        "decisions": set(),
        "open_items": set(),
        "next_steps": set(),
        "notes": set(),
    }

    def ensure_topic(name: str, summary: str, status: str, tags: list[str]) -> dict | None:
        if name in topic_index:
            idx = topic_index[name]
            current = accumulated["topics"][idx]
            if summary:
                current["summary"] = summary
            if status != "uncertain" or current.get("status") == "uncertain":
                current["status"] = status
            current["tags"] = validate_summary_tags(current.get("tags", []) + tags, f"topics[{idx}]")
            return current
        if len(accumulated["topics"]) >= 20:
            return None
        idx = len(accumulated["topics"])
        topic_index[name] = idx
        topic = {
            "name": name,
            "summary": summary or "Assunto consolidado a partir dos minutos recuperados.",
            "status": status,
            "tags": validate_summary_tags(tags, f"topics[{idx}]"),
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        accumulated["topics"].append(topic)
        return topic

    for minute_summary in minute_summaries:
        chunk_type = str(minute_summary.get("chunk_type", "")).strip()
        if chunk_type in SUMMARY_ALLOWED_TYPES and chunk_type not in seen_types:
            seen_types.add(chunk_type)
            accumulated["conversation_types"].append(chunk_type)

        for topic in minute_summary.get("topics", []):
            if not isinstance(topic, dict):
                continue
            topic_name = str(topic.get("name", "")).strip()
            if not topic_name:
                continue
            topic_summary = str(topic.get("summary", "")).strip()
            topic_status = _map_minute_topic_status_to_accumulated(str(topic.get("status", "")).strip())
            topic_tags = topic.get("tags")
            tags = topic_tags if isinstance(topic_tags, list) else []
            ensure_topic(topic_name, topic_summary, topic_status, tags)

        for key in ("facts", "hypotheses", "decisions", "open_items", "next_steps", "notes"):
            for item in minute_summary.get(key, []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                topic = ensure_topic(
                    name,
                    "Assunto consolidado a partir dos minutos recuperados.",
                    "uncertain",
                    [],
                )
                if topic is None:
                    continue
                text = str(item.get("text", "")).strip()
                status = str(item.get("status", "")).strip()
                if not text or not status:
                    continue
                dedupe_key = (text, status, name)
                if dedupe_key in seen_items[key]:
                    continue
                if len(topic[key]) >= SUMMARY_ACCUMULATED_MAX_ITEMS:
                    continue
                seen_items[key].add(dedupe_key)
                item_tags = item.get("tags")
                tags = item_tags if isinstance(item_tags, list) else []
                topic[key].append(
                    {
                        "text": text,
                        "confidence": str(item.get("confidence", "medium")).strip().lower(),
                        "status": status,
                        "tags": validate_summary_tags(tags, f"{key}[{len(topic[key])}]"),
                    }
                )

    return validate_accumulated_summary_payload(accumulated)


def build_deterministic_final_summary(
    accumulated_summary: dict,
    missing_minutes: list[int],
    reasons: list[str],
) -> dict:
    normalized_acc = (
        validate_accumulated_summary_payload(accumulated_summary)
        if isinstance(accumulated_summary, dict)
        else default_accumulated_summary_payload()
    )

    def _collect_topic_texts(topic: dict, source_key: str, limit: int = SUMMARY_FINAL_MAX_ITEMS) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for item in topic.get(source_key, []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= limit:
                break
        return texts

    final_topics: list[dict] = []
    for topic in normalized_acc.get("topics", []):
        if not isinstance(topic, dict):
            continue
        name = str(topic.get("name", "")).strip()
        if not name:
            continue
        summary = str(topic.get("summary", "")).strip() or "Resumo consolidado do assunto."
        final_topics.append(
            {
                "name": name,
                "summary": summary,
                "decisions": _collect_topic_texts(topic, "decisions"),
                "pending_items": _collect_topic_texts(topic, "open_items"),
                "next_steps": _collect_topic_texts(topic, "next_steps"),
                "tags": validate_summary_tags(topic.get("tags"), f"final.topics[{len(final_topics)}]"),
            }
        )

    def _global_items(source_key: str) -> list[dict]:
        items: list[dict] = []
        seen: set[str] = set()
        for topic in normalized_acc.get("topics", []):
            if not isinstance(topic, dict):
                continue
            for item in topic.get(source_key, []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                tags = item.get("tags")
                items.append(
                    {
                        "text": text,
                        "confidence": str(item.get("confidence", "medium")).strip().lower(),
                        "tags": validate_summary_tags(tags if isinstance(tags, list) else [], source_key),
                    }
                )
                if len(items) >= SUMMARY_FINAL_MAX_ITEMS:
                    return items
        return items

    additional_notes = _global_items("notes")
    if missing_minutes:
        additional_notes.append(_build_degradation_additional_note(missing_minutes, reasons))
        if len(additional_notes) > SUMMARY_FINAL_MAX_ITEMS:
            additional_notes = additional_notes[:SUMMARY_FINAL_MAX_ITEMS]

    global_decisions = _global_items("decisions")
    if not global_decisions:
        global_decisions = [
            {
                "text": "Nenhuma decisao explicita foi registrada.",
                "confidence": "medium",
                "tags": [],
            }
        ]

    executive_summary = "Ata final consolidada com os dados recuperados da chamada."
    if final_topics:
        executive_summary = f"Ata final consolidada com {len(final_topics)} assunto(s) principal(is)."
    if missing_minutes:
        executive_summary = (
            f"Documento parcial: minutos sem consolidacao completa [{_minutes_label(missing_minutes)}]. "
            f"{executive_summary}"
        )

    payload = {
        "title": _build_deterministic_final_title(normalized_acc, missing_minutes),
        "conversation_types": normalized_acc.get("conversation_types", []),
        "executive_summary": executive_summary,
        "topics": final_topics,
        "global_decisions": global_decisions,
        "global_pending_items": _global_items("open_items"),
        "global_next_steps": _global_items("next_steps"),
        "additional_notes": additional_notes,
    }
    return validate_final_summary_payload(payload)


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


async def process_progressive_ata_minute_task(
    firebase_router: FirebaseRouter,
    summary_engine: SummaryEngine,
    routing: RoomRoutingContext,
    session_is_finalized: bool,
    room_name: str,
    call_session_id: str,
    minute_index: int,
    export_row: sqlite3.Row,
    now_iso: str,
) -> None:
    minute_payload = await asyncio.to_thread(
        firebase_router.fetch_json, routing, export_row["transcript_json_path"]
    )
    lines = minute_payload.get("lines", []) if isinstance(minute_payload, dict) else []
    accumulated_path = session_progressive_ata_accumulated_path(routing.storage_base_path)
    accumulated_meta_path = session_progressive_ata_meta_path(routing.storage_base_path)
    previous_text = await asyncio.to_thread(firebase_router.fetch_text, routing, accumulated_path)
    metadata = build_summary_metadata(routing, room_name, now_iso)
    metadata.update(
        {
            "summary_mode": SUMMARY_MODE,
            "minute_index": minute_index,
            "minute_started_at": (
                minute_payload.get("minute_started_at") if isinstance(minute_payload, dict) else None
            ),
            "minute_ended_at": (
                minute_payload.get("minute_ended_at") if isinstance(minute_payload, dict) else None
            ),
        }
    )
    logger.info(
        "Gerando ata progressiva até minuto %04d room=%s session=%s model=%s",
        minute_index,
        room_name,
        call_session_id,
        summary_engine.accumulated_model,
    )
    started_at = time.monotonic()
    accumulated_text = await asyncio.to_thread(
        summary_engine.update_progressive_ata,
        previous_text or "",
        lines,
        metadata,
        None,
    )
    logger.info(
        "Ata progressiva até minuto %04d gerada duration_seconds=%.3f room=%s session=%s",
        minute_index,
        time.monotonic() - started_at,
        room_name,
        call_session_id,
    )
    summary_text_path = export_row["summary_json_path"] or minute_progressive_ata_summary_path(
        routing.storage_base_path, minute_index
    )
    await asyncio.to_thread(
        firebase_router.upload_text,
        routing,
        summary_text_path,
        accumulated_text,
        "text/plain; charset=utf-8",
    )
    await asyncio.to_thread(
        firebase_router.upload_text,
        routing,
        accumulated_path,
        accumulated_text,
        "text/plain; charset=utf-8",
    )
    await asyncio.to_thread(
        firebase_router.upload_json,
        routing,
        accumulated_meta_path,
        {
            **metadata,
            "source": "ata_progressiva",
            "last_minute_index": minute_index,
            "accumulated_path": accumulated_path,
            "summary_text_path": summary_text_path,
        },
    )
    await asyncio.to_thread(
        db_update_minute_export_summary_path,
        room_name,
        call_session_id,
        minute_index,
        summary_text_path,
        now_iso,
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
        final_summary_text_path=None if not session_is_finalized else CALL_INDEX_UNSET,
        final_summary_text_ready=False if not session_is_finalized else CALL_INDEX_UNSET,
    )
    if session_is_finalized and SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES:
        await asyncio.to_thread(
            db_schedule_final_summary_task,
            room_name,
            call_session_id,
            utc_now_iso(),
            True,
        )


async def process_progressive_ata_final_task(
    firebase_router: FirebaseRouter,
    summary_engine: SummaryEngine,
    routing: RoomRoutingContext,
    session_row: sqlite3.Row,
    room_name: str,
    call_session_id: str,
    minute_index: int,
    now_iso: str,
) -> bool:
    task_rows = await asyncio.to_thread(db_get_summary_task_rows, room_name, call_session_id)
    exports = await asyncio.to_thread(db_get_session_minute_exports, room_name, call_session_id)
    expected_minutes = {
        int(export["minute_index"]) for export in exports if int(export["minute_index"]) >= 0
    }
    status_by_minute = {
        int(row["minute_index"]): str(row["status"] or "")
        for row in task_rows
        if int(row["minute_index"]) >= 0
    }
    pending_task_minutes = {
        minute for minute, status in status_by_minute.items() if status != "done"
    }
    for minute in expected_minutes:
        if status_by_minute.get(minute) != "done":
            pending_task_minutes.add(minute)

    if pending_task_minutes:
        finalized_at_raw = str(session_row["finalized_at"] or "").strip()
        if finalized_at_raw and SUMMARY_FINALIZATION_GRACE_SECONDS > 0:
            try:
                now_dt = parse_iso_datetime(now_iso)
                finalized_dt = parse_iso_datetime(finalized_at_raw)
                elapsed = (now_dt - finalized_dt).total_seconds()
            except ValueError:
                elapsed = float(SUMMARY_FINALIZATION_GRACE_SECONDS)
                now_dt = parse_iso_datetime(now_iso)
                finalized_dt = now_dt
            if elapsed < SUMMARY_FINALIZATION_GRACE_SECONDS:
                deadline_dt = finalized_dt + timedelta(seconds=SUMMARY_FINALIZATION_GRACE_SECONDS)
                if deadline_dt <= now_dt:
                    deadline_dt = now_dt + timedelta(seconds=5)
                await asyncio.to_thread(
                    db_reschedule_summary_task,
                    room_name,
                    call_session_id,
                    minute_index,
                    deadline_dt.isoformat(),
                    (
                        "aguardando minutos pendentes antes da ata progressiva final: "
                        f"[{_minutes_label(sorted(pending_task_minutes))}]"
                    ),
                    now_iso,
                )
                logger.info(
                    "progressive final delayed by grace room=%s session=%s pending_minutes=%s grace_seconds=%s",
                    room_name,
                    call_session_id,
                    sorted(pending_task_minutes),
                    SUMMARY_FINALIZATION_GRACE_SECONDS,
                )
                return False

    accumulated_path = session_progressive_ata_accumulated_path(routing.storage_base_path)
    accumulated_meta_path = session_progressive_ata_meta_path(routing.storage_base_path)
    final_transcript_path = session_final_transcript_path(routing.storage_base_path)
    final_summary_text_path = session_final_summary_text_path(routing.storage_base_path)
    accumulated_text = await asyncio.to_thread(firebase_router.fetch_text, routing, accumulated_path)
    accumulated_meta = await asyncio.to_thread(firebase_router.fetch_json, routing, accumulated_meta_path)
    last_minute_index = -1
    if isinstance(accumulated_meta, dict):
        try:
            last_minute_index = int(accumulated_meta.get("last_minute_index", -1))
        except (TypeError, ValueError):
            last_minute_index = -1

    final_transcript_payload = await asyncio.to_thread(
        firebase_router.fetch_json, routing, final_transcript_path
    )
    if not isinstance(final_transcript_payload, dict):
        final_transcript_payload = await asyncio.to_thread(
            build_final_transcript_payload,
            room_name,
            call_session_id,
            routing.transcript_session_id,
            now_iso,
        )
    full_transcript_text = transcript_payload_to_text(final_transcript_payload)

    def fetch_minute_payload(export: sqlite3.Row) -> dict | None:
        return firebase_router.fetch_json(routing, export["transcript_json_path"])

    selected_source = SUMMARY_PROGRESSIVE_FINAL_SOURCE
    if selected_source == "auto":
        selected_source = (
            "full_transcript"
            if len(full_transcript_text) <= SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS
            else "delta_only"
        )
    elif (
        selected_source == "full_transcript"
        and len(full_transcript_text) > SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS
    ):
        logger.warning(
            "progressive final forced full_transcript exceeds max chars room=%s session=%s chars=%s max=%s",
            room_name,
            call_session_id,
            len(full_transcript_text),
            SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS,
        )

    included_minutes: list[int] = []
    if selected_source == "full_transcript":
        final_source_text = full_transcript_text
    else:
        final_source_text, included_minutes = build_progressive_delta_transcript_text(
            exports,
            last_minute_index,
            fetch_minute_payload,
        )

    metadata = build_summary_metadata(routing, room_name, now_iso)
    metadata.update(
        {
            "summary_mode": SUMMARY_MODE,
            "final_source_requested": SUMMARY_PROGRESSIVE_FINAL_SOURCE,
            "final_source_used": selected_source,
            "full_transcript_chars": len(full_transcript_text),
            "full_transcript_max_chars": SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS,
            "last_accumulated_minute_index": last_minute_index,
            "delta_minute_indexes": included_minutes,
        }
    )
    logger.info(
        "Gerando ata progressiva final room=%s session=%s source=%s model=%s",
        room_name,
        call_session_id,
        selected_source,
        summary_engine.final_text_model,
    )
    started_at = time.monotonic()
    final_summary_text = await asyncio.to_thread(
        summary_engine.finalize_progressive_ata,
        accumulated_text or "",
        final_source_text,
        selected_source,
        metadata,
        None,
    )
    logger.info(
        "Ata progressiva final gerada duration_seconds=%.3f room=%s session=%s source=%s",
        time.monotonic() - started_at,
        room_name,
        call_session_id,
        selected_source,
    )
    await asyncio.to_thread(
        firebase_router.upload_text,
        routing,
        final_summary_text_path,
        final_summary_text,
        "text/plain; charset=utf-8",
    )
    await asyncio.to_thread(
        firebase_router.publish_call_index,
        routing=routing,
        status="finalized",
        last_minute_index=max(expected_minutes, default=-1),
        finalized=True,
        final_summary_path=None,
        summary_accumulated_path=accumulated_path,
        final_summary_ready=False,
        final_transcript_path=final_transcript_path,
        final_transcript_ready=True,
        final_summary_text_path=final_summary_text_path,
        final_summary_text_ready=True,
    )
    return True


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
            if finalized and summary_enabled and SUMMARY_MODE == "rolling_json"
            else None
        ),
        summary_accumulated_path=(
            session_progressive_ata_accumulated_path(routing.storage_base_path)
            if SUMMARY_MODE == "ata_progressiva"
            else session_summary_accumulated_path(routing.storage_base_path)
        ),
        final_summary_ready=False,
        final_transcript_path=None,
        final_transcript_ready=False,
        final_summary_text_path=None,
        final_summary_text_ready=False,
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
            final_summary_text_path = session_final_summary_text_path(routing.storage_base_path)
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
                    if summary_engine.enabled and SUMMARY_MODE == "rolling_json"
                    else None
                ),
                summary_accumulated_path=(
                    session_progressive_ata_accumulated_path(routing.storage_base_path)
                    if SUMMARY_MODE == "ata_progressiva"
                    else session_summary_accumulated_path(routing.storage_base_path)
                ),
                final_summary_ready=False,
                final_transcript_path=final_transcript_path,
                final_transcript_ready=True,
                final_summary_text_path=(
                    final_summary_text_path if summary_engine.enabled else None
                ),
                final_summary_text_ready=False,
            )

            if summary_engine.enabled:
                queued = await asyncio.to_thread(
                    db_schedule_final_summary_task,
                    room_name,
                    call_session_id,
                    now_iso,
                    False,
                )
                if queued:
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


async def run_summary_reconciliation_once() -> None:
    now_iso = utc_now_iso()
    queued_count = 0
    reopened_count = 0
    late_requeued_count = 0
    recovered = await asyncio.to_thread(
        db_recover_stale_summary_tasks,
        now_iso,
        SUMMARY_PROCESSING_STALE_SECONDS,
    )
    if recovered > 0:
        logger.warning(
            "summary reconcile recovered stale processing tasks count=%s stale_seconds=%s",
            recovered,
            SUMMARY_PROCESSING_STALE_SECONDS,
        )

    finalized_sessions = await asyncio.to_thread(db_get_finalized_sessions_for_summary_reconcile)
    for row in finalized_sessions:
        room_name = str(row["room_name"] or "")
        call_session_id = str(row["call_session_id"] or "").strip()
        if not room_name or not call_session_id:
            continue
        task_rows = await asyncio.to_thread(db_get_summary_task_rows, room_name, call_session_id)
        final_row = next((task for task in task_rows if int(task["minute_index"]) < 0), None)
        if final_row is None:
            queued = await asyncio.to_thread(
                db_schedule_final_summary_task,
                room_name,
                call_session_id,
                now_iso,
                False,
            )
            if queued:
                queued_count += 1
                logger.info(
                    "summary reconcile queued missing final task room=%s session=%s minute=-1",
                    room_name,
                    call_session_id,
                )
            continue

        final_status = str(final_row["status"] or "")
        final_retries = int(final_row["retries"] or 0)
        final_error_message = str(final_row["error_message"] or "")
        if (
            final_status == "error"
            and final_retries >= SUMMARY_MAX_RETRIES
            and SUMMARY_FINAL_ENABLE_DETERMINISTIC_FALLBACK
            and "ata final markdown" not in final_error_message.lower()
            and "ata final textual" not in final_error_message.lower()
        ):
            reopened = await asyncio.to_thread(
                db_schedule_final_summary_task,
                room_name,
                call_session_id,
                now_iso,
                True,
            )
            if reopened:
                reopened_count += 1
                logger.warning(
                    "summary reconcile reopened exhausted final task room=%s session=%s",
                    room_name,
                    call_session_id,
                )

        if not SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES or final_status != "done":
            continue
        minute_rows = [task for task in task_rows if int(task["minute_index"]) >= 0]
        if not minute_rows:
            continue
        try:
            latest_minute_update = max(
                parse_iso_datetime(str(task["updated_at"])) for task in minute_rows
            )
            final_update = parse_iso_datetime(str(final_row["updated_at"]))
        except ValueError:
            continue
        if latest_minute_update <= final_update:
            continue
        reopened = await asyncio.to_thread(
            db_schedule_final_summary_task,
            room_name,
            call_session_id,
            now_iso,
            True,
        )
        if reopened:
            late_requeued_count += 1
            logger.info(
                "summary reconcile requeued final task after late minute room=%s session=%s",
                room_name,
                call_session_id,
            )
    logger.info(
        "Reconciliando summaries: recovered=%s sessions=%s queued=%s reopened=%s late_requeued=%s",
        recovered,
        len(finalized_sessions),
        queued_count,
        reopened_count,
        late_requeued_count,
    )


async def summary_worker_loop(
    stop_event: asyncio.Event,
    firebase_router: FirebaseRouter,
    summary_engine: SummaryEngine,
) -> None:
    logger.info(
        "summary worker started enabled=%s provider=%s",
        summary_engine.enabled,
        summary_engine.provider,
    )
    logger.info("iniciando geração do sumario")
    logger.info(
        "modelos: { minuto: %s, acumulado: %s, final: %s, final_text: %s }",
        summary_engine.minute_model,
        summary_engine.accumulated_model,
        summary_engine.final_model,
        summary_engine.final_text_model,
    )
    next_reconcile_at = 0.0
    while not stop_event.is_set():
        if not summary_engine.enabled:
            await asyncio.sleep(1.0)
            continue

        now_monotonic = time.monotonic()
        if now_monotonic >= next_reconcile_at:
            try:
                await run_summary_reconciliation_once()
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("summary reconcile failed: %s", exc)
            next_reconcile_at = now_monotonic + SUMMARY_RECONCILE_INTERVAL_SECONDS

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
            if SUMMARY_MODE == "ata_progressiva":
                if minute_index >= 0:
                    export_row = await asyncio.to_thread(
                        db_get_minute_export, room_name, call_session_id, minute_index
                    )
                    if export_row is None:
                        await asyncio.to_thread(
                            db_mark_summary_task_done, room_name, call_session_id, minute_index, now_iso
                        )
                        continue
                    await process_progressive_ata_minute_task(
                        firebase_router,
                        summary_engine,
                        routing,
                        session_is_finalized,
                        room_name,
                        call_session_id,
                        minute_index,
                        export_row,
                        now_iso,
                    )
                else:
                    processed = await process_progressive_ata_final_task(
                        firebase_router,
                        summary_engine,
                        routing,
                        session_row,
                        room_name,
                        call_session_id,
                        minute_index,
                        now_iso,
                    )
                    if not processed:
                        continue
                await asyncio.to_thread(
                    db_mark_summary_task_done, room_name, call_session_id, minute_index, utc_now_iso()
                )
                continue
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
                logger.info(
                    "Gerando o sumario do %04d room=%s session=%s model=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                    summary_engine.minute_model,
                )
                summary_started_at = time.monotonic()
                minute_summary = await asyncio.to_thread(
                    summary_engine.summarize_minute, lines, minute_system_prompt
                )
                logger.info(
                    "Gerando o sumario do %04d duration_seconds=%.3f room=%s session=%s",
                    minute_index,
                    time.monotonic() - summary_started_at,
                    room_name,
                    call_session_id,
                )
                summary_path = export_row["summary_json_path"] or (
                    join_storage_path(routing.storage_base_path, f"minutes/{minute_index:04d}/summary.json")
                )
                logger.info(
                    "enviando sumario do %04d para o storage room=%s session=%s path=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                    summary_path,
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
                logger.info(
                    "registrando metadata sumario do %04d room=%s session=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                )
                await asyncio.to_thread(
                    db_update_minute_export_summary_path,
                    room_name,
                    call_session_id,
                    minute_index,
                    summary_path,
                    now_iso,
                )
                logger.info(
                    "metadata sumario do %04d registrada room=%s session=%s",
                    minute_index,
                    room_name,
                    call_session_id,
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
                logger.info(
                    "Gerando acumulado até minuto %04d room=%s session=%s model=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                    summary_engine.accumulated_model,
                )
                accumulated_started_at = time.monotonic()
                merged_summary = await asyncio.to_thread(
                    summary_engine.merge_summaries,
                    previous_summary,
                    minute_summary,
                    merge_system_prompt,
                )
                logger.info(
                    "Gerando acumulado até minuto %04d duration_seconds=%.3f room=%s session=%s",
                    minute_index,
                    time.monotonic() - accumulated_started_at,
                    room_name,
                    call_session_id,
                )
                logger.info(
                    "enviando sumario acumulado até %04d para o storage room=%s session=%s path=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                    accumulated_path,
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
                logger.info(
                    "registrando metadata sumario acumulado até %04d no firestore room=%s session=%s",
                    minute_index,
                    room_name,
                    call_session_id,
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
                    final_summary_text_path=None if not session_is_finalized else CALL_INDEX_UNSET,
                    final_summary_text_ready=False if not session_is_finalized else CALL_INDEX_UNSET,
                )
                logger.info(
                    "metadata sumario acumulado até %04d registrada no firestore room=%s session=%s",
                    minute_index,
                    room_name,
                    call_session_id,
                )
                if session_is_finalized and SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES:
                    await asyncio.to_thread(
                        db_schedule_final_summary_task,
                        room_name,
                        call_session_id,
                        utc_now_iso(),
                        True,
                    )
            else:
                task_rows = await asyncio.to_thread(db_get_summary_task_rows, room_name, call_session_id)
                exports = await asyncio.to_thread(
                    db_get_session_minute_exports, room_name, call_session_id
                )
                expected_minutes = {
                    int(export["minute_index"])
                    for export in exports
                    if int(export["minute_index"]) >= 0
                }
                status_by_minute: dict[int, str] = {
                    int(row["minute_index"]): str(row["status"] or "")
                    for row in task_rows
                    if int(row["minute_index"]) >= 0
                }
                pending_task_minutes = {
                    minute
                    for minute, status in status_by_minute.items()
                    if status != "done"
                }
                for minute in expected_minutes:
                    if status_by_minute.get(minute) != "done":
                        pending_task_minutes.add(minute)

                if pending_task_minutes:
                    finalized_at_raw = str(session_row["finalized_at"] or "").strip()
                    if finalized_at_raw and SUMMARY_FINALIZATION_GRACE_SECONDS > 0:
                        try:
                            now_dt = parse_iso_datetime(now_iso)
                            finalized_dt = parse_iso_datetime(finalized_at_raw)
                            elapsed = (now_dt - finalized_dt).total_seconds()
                        except ValueError:
                            elapsed = float(SUMMARY_FINALIZATION_GRACE_SECONDS)
                            now_dt = parse_iso_datetime(now_iso)
                            finalized_dt = now_dt
                        if elapsed < SUMMARY_FINALIZATION_GRACE_SECONDS:
                            deadline_dt = finalized_dt + timedelta(
                                seconds=SUMMARY_FINALIZATION_GRACE_SECONDS
                            )
                            if deadline_dt <= now_dt:
                                deadline_dt = now_dt + timedelta(seconds=5)
                            await asyncio.to_thread(
                                db_reschedule_summary_task,
                                room_name,
                                call_session_id,
                                minute_index,
                                deadline_dt.isoformat(),
                                (
                                    "aguardando minutos pendentes antes da ata final: "
                                    f"[{_minutes_label(sorted(pending_task_minutes))}]"
                                ),
                                now_iso,
                            )
                            logger.info(
                                "final summary delayed by grace room=%s session=%s pending_minutes=%s grace_seconds=%s",
                                room_name,
                                call_session_id,
                                sorted(pending_task_minutes),
                                SUMMARY_FINALIZATION_GRACE_SECONDS,
                            )
                            continue

                accumulated_path = session_summary_accumulated_path(routing.storage_base_path)
                accumulated_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, routing, accumulated_path
                )
                merged_summary: dict | None = None
                recovery_reasons: list[str] = []
                if isinstance(accumulated_payload, dict):
                    raw_summary = accumulated_payload.get("summary")
                    if isinstance(raw_summary, dict):
                        try:
                            merged_summary = validate_accumulated_summary_payload(raw_summary)
                        except RuntimeError as exc:
                            recovery_reasons.append("resumo acumulado invalido")
                            logger.warning(
                                "invalid accumulated payload before finalization room=%s session=%s error=%s",
                                room_name,
                                call_session_id,
                                exc,
                            )
                else:
                    recovery_reasons.append("resumo acumulado indisponivel")

                recovered_minute_summaries: list[dict] = []
                unavailable_minutes: set[int] = set()
                for export in exports:
                    minute = int(export["minute_index"])
                    if minute < 0:
                        continue
                    summary_path = str(export["summary_json_path"] or "").strip()
                    if not summary_path:
                        unavailable_minutes.add(minute)
                        continue
                    try:
                        minute_payload = await asyncio.to_thread(
                            firebase_router.fetch_json,
                            routing,
                            summary_path,
                        )
                        if not isinstance(minute_payload, dict):
                            unavailable_minutes.add(minute)
                            continue
                        raw_minute_summary = minute_payload.get("summary")
                        if not isinstance(raw_minute_summary, dict):
                            unavailable_minutes.add(minute)
                            continue
                        recovered_minute_summaries.append(
                            validate_minute_summary_payload(raw_minute_summary)
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        unavailable_minutes.add(minute)
                        logger.warning(
                            "failed to recover minute summary for finalization room=%s session=%s minute=%s error=%s",
                            room_name,
                            call_session_id,
                            minute,
                            exc,
                        )

                if merged_summary is None:
                    if recovered_minute_summaries:
                        merged_summary = build_accumulated_from_minute_summaries(
                            recovered_minute_summaries
                        )
                        recovery_reasons.append(
                            "acumulado reconstruido a partir de minutos recuperados"
                        )
                    else:
                        merged_summary = default_accumulated_summary_payload()
                        recovery_reasons.append("nenhum minuto valido recuperado")

                missing_minutes = sorted(pending_task_minutes | unavailable_minutes)
                if missing_minutes:
                    recovery_reasons.append(
                        f"minutos sem consolidacao total: [{_minutes_label(missing_minutes)}]"
                    )

                final_path = session_final_summary_path(routing.storage_base_path)
                final_temp_path = session_final_summary_temp_path(routing.storage_base_path)
                final_summary_text_path = session_final_summary_text_path(routing.storage_base_path)
                final_summary: dict | None = None
                final_payload = await asyncio.to_thread(
                    firebase_router.fetch_json, routing, final_path
                )
                if isinstance(final_payload, dict) and isinstance(final_payload.get("summary"), dict):
                    try:
                        final_summary = validate_final_summary_payload(final_payload["summary"])
                        logger.info(
                            "Ata final existente reutilizada room=%s session=%s path=%s",
                            room_name,
                            call_session_id,
                            final_path,
                        )
                    except RuntimeError as exc:
                        logger.warning(
                            "final_summary.json existente invalido; gerando novamente room=%s session=%s path=%s error=%s",
                            room_name,
                            call_session_id,
                            final_path,
                            exc,
                        )
                try:
                    if final_summary is None:
                        logger.info(
                            "Gerando Ata final room=%s session=%s model=%s",
                            room_name,
                            call_session_id,
                            summary_engine.final_model,
                        )
                        final_system_prompt = await asyncio.to_thread(
                            firebase_router.fetch_agent_prompt,
                            routing,
                            "stt_finalize_summary",
                        )
                        final_started_at = time.monotonic()
                        final_summary = await asyncio.to_thread(
                            summary_engine.finalize_summary,
                            merged_summary,
                            final_system_prompt,
                        )
                        logger.info(
                            "gerando final_summary.json duration_seconds=%.3f room=%s session=%s",
                            time.monotonic() - final_started_at,
                            room_name,
                            call_session_id,
                        )
                        final_summary = inject_degradation_disclosure(
                            final_summary,
                            missing_minutes,
                            recovery_reasons,
                        )
                except SummaryContractValidationError as exc:
                    logger.warning(
                        "final summary contract invalid room=%s session=%s model=%s kind=%s error=%s",
                        room_name,
                        call_session_id,
                        exc.model,
                        exc.kind,
                        exc,
                    )
                    temp_payload = build_final_summary_temp_payload(
                        room_name=room_name,
                        transcript_session_id=routing.transcript_session_id,
                        call_session_id=routing.call_session_id,
                        model=exc.model,
                        kind=exc.kind,
                        error=str(exc),
                        raw_output=exc.raw_output,
                        updated_at=now_iso,
                    )
                    logger.warning(
                        "enviando ata final temp json para o storage room=%s session=%s path=%s",
                        room_name,
                        call_session_id,
                        final_temp_path,
                    )
                    await asyncio.to_thread(
                        firebase_router.upload_json,
                        routing,
                        final_temp_path,
                        temp_payload,
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
                        final_summary_ready=False,
                        final_transcript_path=final_transcript_path,
                        final_transcript_ready=True,
                        final_summary_text_path=final_summary_text_path,
                        final_summary_text_ready=False,
                    )
                    logger.warning(
                        "final summary temp uploaded room=%s session=%s path=%s",
                        room_name,
                        call_session_id,
                        final_temp_path,
                    )
                    raise RuntimeError(f"falha de contrato na ata final: {exc}") from exc
                except Exception as exc:  # pylint: disable=broad-except
                    if not SUMMARY_FINAL_ENABLE_DETERMINISTIC_FALLBACK:
                        raise
                    logger.warning(
                        "final summary fallback deterministic room=%s session=%s error=%s",
                        room_name,
                        call_session_id,
                        exc,
                    )
                    fallback_reasons = list(recovery_reasons)
                    fallback_reasons.append("falha na geracao final via LLM")
                    final_summary = build_deterministic_final_summary(
                        merged_summary,
                        missing_minutes,
                        fallback_reasons,
                    )

                if final_summary is not None:
                    logger.info(
                        "enviando ata final json para o storage room=%s session=%s path=%s",
                        room_name,
                        call_session_id,
                        final_path,
                    )
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
                    logger.info(
                        "ata final json enviada para o storage room=%s session=%s path=%s",
                        room_name,
                        call_session_id,
                        final_path,
                    )
                    final_summary_text_ready = False
                    final_transcript_path = session_final_transcript_path(routing.storage_base_path)
                    try:
                        final_text_system_prompt = await asyncio.to_thread(
                            firebase_router.fetch_agent_prompt,
                            routing,
                            "stt_finalize_summary_text",
                        )
                        logger.info(
                            "gerando final_summary_text.txt room=%s session=%s model=%s format=%s",
                            room_name,
                            call_session_id,
                            summary_engine.final_text_model,
                            SUMMARY_FINAL_TEXT_FORMAT,
                        )
                        final_text_started_at = time.monotonic()
                        final_summary_text = await asyncio.to_thread(
                            summary_engine.finalize_summary_text,
                            {
                                **final_summary,
                                "updated_at": now_iso,
                                "room_name": room_name,
                                "transcript_session_id": routing.transcript_session_id,
                                "call_session_id": routing.call_session_id,
                            },
                            SUMMARY_FINAL_TEXT_FORMAT,
                            final_text_system_prompt,
                        )
                        logger.info(
                            "gerando final_summary_text.txt duration_seconds=%.3f room=%s session=%s",
                            time.monotonic() - final_text_started_at,
                            room_name,
                            call_session_id,
                        )
                        logger.info(
                            "enviando ata final textual para o storage room=%s session=%s path=%s format=%s",
                            room_name,
                            call_session_id,
                            final_summary_text_path,
                            SUMMARY_FINAL_TEXT_FORMAT,
                        )
                        await asyncio.to_thread(
                            firebase_router.upload_text,
                            routing,
                            final_summary_text_path,
                            final_summary_text,
                            "text/plain; charset=utf-8",
                        )
                        final_summary_text_ready = True
                        logger.info(
                            "final summary text uploaded room=%s session=%s path=%s format=%s",
                            room_name,
                            call_session_id,
                            final_summary_text_path,
                            SUMMARY_FINAL_TEXT_FORMAT,
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            "final summary text generation failed room=%s session=%s model=%s format=%s error=%s",
                            room_name,
                            call_session_id,
                            summary_engine.final_text_model,
                            SUMMARY_FINAL_TEXT_FORMAT,
                            exc,
                        )
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
                            final_summary_text_path=final_summary_text_path,
                            final_summary_text_ready=False,
                        )
                        logger.info(
                            "registrando metadata ata final textual no firestore room=%s session=%s ready=false format=%s",
                            room_name,
                            call_session_id,
                            SUMMARY_FINAL_TEXT_FORMAT,
                        )
                        raise RuntimeError(f"falha ao gerar/enviar ata final textual: {exc}") from exc
                    logger.info(
                        "registrando metadata ata final json no firestore room=%s session=%s",
                        room_name,
                        call_session_id,
                    )
                    logger.info(
                        "registrando metadata ata final textual no firestore room=%s session=%s ready=%s format=%s",
                        room_name,
                        call_session_id,
                        final_summary_text_ready,
                        SUMMARY_FINAL_TEXT_FORMAT,
                    )
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
                        final_summary_text_path=final_summary_text_path,
                        final_summary_text_ready=final_summary_text_ready,
                    )
                    logger.info(
                        "metadata ata final registrada no firestore room=%s session=%s final_summary_ready=%s final_summary_text_ready=%s",
                        room_name,
                        call_session_id,
                        True,
                        final_summary_text_ready,
                    )
                    logger.info(
                        "final summary uploaded room=%s session=%s path=%s partial=%s",
                        room_name,
                        call_session_id,
                        final_path,
                        bool(missing_minutes),
                    )

            await asyncio.to_thread(
                db_mark_summary_task_done, room_name, call_session_id, minute_index, utc_now_iso()
            )
        except Exception as exc:  # pylint: disable=broad-except
            retry_count = retries + 1
            error_now_iso = utc_now_iso()
            next_attempt_iso = (
                parse_iso_datetime(error_now_iso) + timedelta(seconds=15 * retry_count)
            ).isoformat()
            logger.exception(
                "summary task failed room=%s session=%s minute=%s attempt=%s/%s next_attempt_at=%s error=%s",
                room_name,
                call_session_id,
                minute_index,
                retry_count,
                SUMMARY_MAX_RETRIES,
                next_attempt_iso,
                exc,
            )
            await asyncio.to_thread(
                db_mark_summary_task_error,
                room_name,
                call_session_id,
                minute_index,
                retry_count,
                str(exc),
                error_now_iso,
            )
            if retry_count >= SUMMARY_MAX_RETRIES:
                logger.error(
                    "summary task retries exhausted room=%s session=%s minute=%s retries=%s error=%s",
                    room_name,
                    call_session_id,
                    minute_index,
                    retry_count,
                    exc,
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
    if summary_engine.enabled:
        await run_summary_reconciliation_once()
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
        "summary_enabled": runtime.summary_engine.enabled,
        "summary_mode": SUMMARY_MODE,
        "summary_provider": runtime.summary_engine.provider,
        "summary_model_minute": runtime.summary_engine.minute_model,
        "summary_model_accumulated": runtime.summary_engine.accumulated_model,
        "summary_model_final": runtime.summary_engine.final_model,
        "summary_model_final_text": runtime.summary_engine.final_text_model,
        "summary_final_text_format": SUMMARY_FINAL_TEXT_FORMAT,
        "summary_progressive_final_source": SUMMARY_PROGRESSIVE_FINAL_SOURCE,
        "summary_progressive_full_transcript_max_chars": SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS,
        "summary_reconcile_interval_seconds": SUMMARY_RECONCILE_INTERVAL_SECONDS,
        "summary_processing_stale_seconds": SUMMARY_PROCESSING_STALE_SECONDS,
        "summary_finalization_grace_seconds": SUMMARY_FINALIZATION_GRACE_SECONDS,
        "summary_final_reemit_on_late_minutes": SUMMARY_FINAL_REEMIT_ON_LATE_MINUTES,
        "summary_final_enable_deterministic_fallback": SUMMARY_FINAL_ENABLE_DETERMINISTIC_FALLBACK,
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


def build_summary_reprocess_status_payload(
    task_rows: list[sqlite3.Row], minute_exports: list[sqlite3.Row]
) -> dict:
    minute_indexes = [
        int(row["minute_index"]) for row in minute_exports if int(row["minute_index"]) >= 0
    ]
    minute_status_rows = {
        int(row["minute_index"]): row for row in task_rows if int(row["minute_index"]) >= 0
    }
    progress = {
        "total": len(minute_indexes),
        "done": 0,
        "pending": 0,
        "processing": 0,
        "error": 0,
    }
    minute_tasks: list[dict] = []
    for minute_index in minute_indexes:
        row = minute_status_rows.get(minute_index)
        status = str(row["status"] if row is not None else "pending")
        retries = int(row["retries"]) if row is not None and row["retries"] is not None else 0
        error_message = str(row["error_message"] or "") if row is not None else ""
        minute_tasks.append(
            {
                "minute_index": minute_index,
                "status": status,
                "retries": retries,
                "error_message": error_message or None,
            }
        )
        if status in progress:
            progress[status] += 1
        elif status == "done":
            progress["done"] += 1
        else:
            progress["pending"] += 1

    final_row = next((row for row in task_rows if int(row["minute_index"]) < 0), None)
    final_status = str(final_row["status"] if final_row is not None else "pending")
    final_retries = int(final_row["retries"]) if final_row is not None and final_row["retries"] is not None else 0
    final_error = str(final_row["error_message"] or "") if final_row is not None else ""
    final_exhausted = final_status == "error" and final_retries >= SUMMARY_MAX_RETRIES
    final_task = {
        "status": final_status,
        "retries": final_retries,
        "max_retries": SUMMARY_MAX_RETRIES,
        "error_message": final_error or None,
        "exhausted": final_exhausted,
    }
    if final_status == "done":
        overall = "done"
    elif final_exhausted:
        overall = "error_exhausted"
    elif final_status == "processing" or progress["processing"] > 0:
        overall = "processing"
    elif final_status == "error" or progress["error"] > 0:
        overall = "error"
    else:
        overall = "pending"

    return {
        "overall_status": overall,
        "progress": progress,
        "final_task": final_task,
        "minute_tasks": minute_tasks,
    }


@app.post("/v1/admin/summary/reprocess")
async def admin_summary_reprocess(payload: AdminSummaryReprocessTarget) -> JSONResponse:
    runtime: AppState = app.state.runtime
    if not runtime.summary_engine.enabled:
        raise HTTPException(status_code=409, detail="summary_engine_disabled")

    session_row = await asyncio.to_thread(
        db_get_session_row, payload.room_name, payload.call_session_id
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    routing = assert_admin_target_matches_session(payload, session_row)

    now_iso = utc_now_iso()
    if not str(session_row["finalized_at"] or "").strip():
        await asyncio.to_thread(
            db_force_finalize_session_for_reprocess,
            payload.room_name,
            payload.call_session_id,
            now_iso,
        )
        session_row = await asyncio.to_thread(
            db_get_session_row, payload.room_name, payload.call_session_id
        )
        if session_row is None:
            raise HTTPException(status_code=404, detail="session_not_found")
        routing = assert_admin_target_matches_session(payload, session_row)

    await asyncio.to_thread(
        publish_session_minute_exports,
        runtime.firebase_router,
        payload.room_name,
        payload.call_session_id,
        now_iso,
        True,
        True,
    )
    if SUMMARY_MODE == "ata_progressiva":
        cleanup_exports = await asyncio.to_thread(
            db_get_session_minute_exports, payload.room_name, payload.call_session_id
        )
        for export in cleanup_exports:
            summary_path = str(export["summary_json_path"] or "").strip()
            if summary_path:
                await asyncio.to_thread(runtime.firebase_router.delete_object, routing, summary_path)
        await asyncio.to_thread(
            runtime.firebase_router.delete_object,
            routing,
            session_final_summary_text_path(routing.storage_base_path),
        )
    reset_info = await asyncio.to_thread(
        db_reset_summary_reprocess_state,
        payload.room_name,
        payload.call_session_id,
        now_iso,
    )
    if SUMMARY_MODE == "ata_progressiva":
        accumulated_path = session_progressive_ata_accumulated_path(routing.storage_base_path)
        accumulated_meta_path = session_progressive_ata_meta_path(routing.storage_base_path)
        await asyncio.to_thread(runtime.firebase_router.delete_object, routing, accumulated_path)
        await asyncio.to_thread(runtime.firebase_router.delete_object, routing, accumulated_meta_path)
        await asyncio.to_thread(
            runtime.firebase_router.upload_json,
            routing,
            accumulated_meta_path,
            {
                **build_summary_metadata(routing, payload.room_name, now_iso),
                "source": "ata_progressiva",
                "last_minute_index": -1,
                "accumulated_path": accumulated_path,
            },
        )
    else:
        accumulated_path = session_summary_accumulated_path(routing.storage_base_path)
        await asyncio.to_thread(
            runtime.firebase_router.upload_json,
            routing,
            accumulated_path,
            {
                "room_name": routing.room_name,
                "transcript_session_id": routing.transcript_session_id,
                "call_session_id": routing.call_session_id,
                "last_minute_index": -1,
                "summary": default_accumulated_summary_payload(),
                "updated_at": now_iso,
            },
        )
    final_transcript_path = session_final_transcript_path(routing.storage_base_path)
    final_transcript_payload = await asyncio.to_thread(
        build_final_transcript_payload,
        payload.room_name,
        payload.call_session_id,
        routing.transcript_session_id,
        now_iso,
    )
    await asyncio.to_thread(
        runtime.firebase_router.upload_json,
        routing,
        final_transcript_path,
        final_transcript_payload,
    )
    minute_exports = await asyncio.to_thread(
        db_get_session_minute_exports, payload.room_name, payload.call_session_id
    )
    last_minute_index = (
        max(
            (int(row["minute_index"]) for row in minute_exports if int(row["minute_index"]) >= 0),
            default=-1,
        )
        if minute_exports
        else -1
    )
    await asyncio.to_thread(
        runtime.firebase_router.publish_call_index,
        routing=routing,
        status="finalized",
        last_minute_index=last_minute_index,
        finalized=True,
        final_summary_path=(
            None
            if SUMMARY_MODE == "ata_progressiva"
            else session_final_summary_path(routing.storage_base_path)
        ),
        summary_accumulated_path=accumulated_path,
        final_summary_ready=False,
        final_transcript_path=final_transcript_path,
        final_transcript_ready=True,
        final_summary_text_path=session_final_summary_text_path(routing.storage_base_path),
        final_summary_text_ready=False,
    )
    task_rows = await asyncio.to_thread(
        db_get_summary_task_rows, payload.room_name, payload.call_session_id
    )
    snapshot = build_summary_reprocess_status_payload(task_rows, minute_exports)
    snapshot["reset"] = reset_info
    snapshot["queued_at"] = now_iso
    snapshot["session"] = {
        "namespace": payload.namespace,
        "vertical": payload.vertical,
        "slug": payload.slug,
        "room_id": payload.room_id,
        "call_session_id": payload.call_session_id,
        "room_name": payload.room_name,
    }
    return JSONResponse(snapshot, status_code=202)


@app.get("/v1/admin/summary/reprocess/status")
async def admin_summary_reprocess_status(
    namespace: str,
    vertical: str,
    slug: str,
    room_id: str,
    call_session_id: str,
) -> JSONResponse:
    runtime: AppState = app.state.runtime
    try:
        target = AdminSummaryReprocessTarget.model_validate(
            {
                "namespace": namespace,
                "vertical": vertical,
                "slug": slug,
                "room_id": room_id,
                "call_session_id": call_session_id,
            }
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    session_row = await asyncio.to_thread(
        db_get_session_row, target.room_name, target.call_session_id
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    routing = assert_admin_target_matches_session(target, session_row)

    task_rows = await asyncio.to_thread(
        db_get_summary_task_rows, target.room_name, target.call_session_id
    )
    minute_exports = await asyncio.to_thread(
        db_get_session_minute_exports, target.room_name, target.call_session_id
    )
    payload = build_summary_reprocess_status_payload(task_rows, minute_exports)
    payload["session"] = {
        "namespace": target.namespace,
        "vertical": target.vertical,
        "slug": target.slug,
        "room_id": target.room_id,
        "call_session_id": target.call_session_id,
        "room_name": target.room_name,
    }

    final_task = payload["final_task"]
    if final_task["status"] == "done":
        final_payload = await asyncio.to_thread(
            runtime.firebase_router.fetch_json,
            routing,
            session_final_summary_path(routing.storage_base_path),
        )
        if isinstance(final_payload, dict):
            payload["final_summary"] = final_payload.get("summary")

    if final_task["exhausted"]:
        temp_payload = await asyncio.to_thread(
            runtime.firebase_router.fetch_json,
            routing,
            session_final_summary_temp_path(routing.storage_base_path),
        )
        if isinstance(temp_payload, dict):
            payload["final_error_details"] = {
                "error": temp_payload.get("error"),
                "model": temp_payload.get("model"),
                "kind": temp_payload.get("kind"),
                "updated_at": temp_payload.get("updated_at"),
            }

    return JSONResponse(payload, status_code=200)
