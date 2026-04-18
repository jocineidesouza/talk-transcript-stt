# Talk Transcript STT Service (Async HTTP)

Servico de transcricao assíncrona para ingestão de chunks de audio vindos do agente LiveKit.

## Endpoints

- `GET /health`
- `POST /v1/sessions/start`
- `POST /v1/sessions/chunk` (`multipart/form-data` com `meta` + `audio`)
- `POST /v1/sessions/end`

## Fluxo

1. agente envia `start` por participante/sessao
2. agente envia chunks PCM16 mono 16k em `chunk` com `seq` crescente
3. servico grava spool em disco, enfileira no SQLite e responde `202`
4. worker unico consome fila, transcreve e grava incremental em Firestore
5. em `end` de participante/sala, o servico finaliza agregados e salva artefatos textuais no Storage

## Seguranca

Todas as rotas de ingestao exigem HMAC via headers:

- `X-TS-Timestamp`
- `X-TS-Nonce`
- `X-TS-Signature`
- `X-TS-Key-Id`

Canonical string assinada:

```text
METHOD
PATH
TIMESTAMP
NONCE
SHA256_HEX(body)
```

## Banco e fila

- SQLite em WAL mode
- fila unica global em `chunks` (`queued` -> `processing` -> `done/error`)
- idempotencia: chave `(room_name, session_id, participant_identity, seq)`
- anti-replay HMAC: tabela `hmac_nonces`

## Firebase

Quando `FIREBASE_ENABLED=true`:

- Firestore:
  - `calls/{call_key}`
  - `calls/{call_key}/participants/{participant_identity}`
  - `calls/{call_key}/participants/{participant_identity}/segments/{seq}`
- Storage:
  - `calls/{call_key}/participants/{participant}/transcript.txt|json`
  - `calls/{call_key}/transcript.txt|json`

## Principais variaveis de ambiente

- `STT_HMAC_KEY_ID`, `STT_HMAC_SECRET` (ou `STT_HMAC_KEYS`)
- `HMAC_WINDOW_SECONDS` (padrao `300`)
- `SQLITE_PATH` (padrao `/data/queue.db`)
- `SPOOL_DIR` (padrao `/data/spool`)
- `QUEUE_MAX_PENDING` (padrao `2000`)
- `FIREBASE_ENABLED`
- `FIREBASE_SERVICE_ACCOUNT_FILE`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_PROJECT_ID`
