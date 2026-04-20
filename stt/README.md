# Talk Transcript STT Service (Async HTTP)

Servico de transcricao assincrona para ingestao de chunks de audio vindos do agente LiveKit.

## Endpoints

- `GET /health`
- `POST /v1/sessions/start`
- `POST /v1/sessions/chunk` (`multipart/form-data` com `meta` + `audio`)
- `POST /v1/sessions/end`

## Fluxo

1. agente envia `start` por participante/sessao
2. agente envia chunks PCM16 mono 16k em `chunk` com `seq` crescente
3. servico grava spool em disco, enfileira no SQLite e responde `202`
4. worker STT consome fila e grava transcricao no SQLite (`chunks`/`transcripts`)
5. flush configuravel publica minute shards no Storage e indice leve no Firestore
6. worker de resumo (opcional) gera resumo por minuto e resumo acumulado/final
7. fallback de resiliencia: timeout de inatividade marca `room_end` e finaliza apos drenar fila

## Contrato de room_name

- formato obrigatorio: `<namespace>__<room_id>`
- namespaces permitidos:
  - `talk__dev`, `talk__stg`, `talk__prd`
  - `ellevo-connect__dev`, `ellevo-connect__stg`, `ellevo-connect__prd`
- quando a sala nao casar com formato/allowlist, a API responde `202` com status `ignored`

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
- checkpoint de export por minuto: tabela `minute_exports`
- fila de resumo/retry: tabela `summary_tasks`

## Firebase

Quando `FIREBASE_ENABLED=true`:

- destino Firebase e resolvido por namespace da sala
- configuracao por `FIREBASE_NAMESPACE_CONFIG_JSON` (`namespace -> config`)
- cada namespace usa app/credencial propria (sem fallback silencioso)
- nao grava transcricao completa no Firestore por chunk

Firestore:

- `calls/{call_key}`: metadados, status, heartbeat, ponteiros de resumo
- `calls/{call_key}/minute_shards/{minute_index}`: ponteiros dos JSONs por minuto

Storage:

- `calls/{call_key}/minutes/{minute_index}/transcript.json`
- `calls/{call_key}/minutes/{minute_index}/summary.json`
- `calls/{call_key}/summary/accumulated.json`
- `calls/{call_key}/final/final_summary.json`

## OpenAI Summary (opcional)

- habilitado por `OPENAI_SUMMARY_ENABLED=true`
- secret esperado em `OPENAI_APIKEY_FILE` (padrao `/secrets/openai_apikey.json`)
- formato do arquivo: `{"api_key":"..."}` (ver `stt/secrets/openai_apikey.example.json`)
- se secret estiver ausente/invalido, STT continua; resumo fica desabilitado com warning

## Principais variaveis de ambiente

- `STT_HMAC_KEY_ID`, `STT_HMAC_SECRET` (ou `STT_HMAC_KEYS`)
- `HMAC_WINDOW_SECONDS` (padrao `300`)
- `SQLITE_PATH` (padrao `/data/queue.db`)
- `SPOOL_DIR` (padrao `/data/spool`)
- `QUEUE_MAX_PENDING` (padrao `2000`)
- `ROOM_INACTIVITY_TIMEOUT_SECONDS` (padrao `1800`)
- `FIREBASE_ENABLED`
- `FIREBASE_NAMESPACE_CONFIG_JSON`
- `FIREBASE_FLUSH_INTERVAL_SECONDS` (padrao `30`)
- `STORAGE_MINUTE_WINDOW_SECONDS` (padrao `60`)
- `OPENAI_SUMMARY_ENABLED` (padrao `false`)
- `OPENAI_APIKEY_FILE` (padrao `/secrets/openai_apikey.json`)
- `OPENAI_MODEL_MINUTE_SUMMARY` (padrao `gpt-4.1-mini`)
- `OPENAI_MODEL_ACCUMULATED_SUMMARY` (padrao `gpt-4.1`)
- `OPENAI_MODEL_FINAL_SUMMARY` (padrao `gpt-4.1`)
- `OPENAI_REQUEST_TIMEOUT_SECONDS` (padrao `20`)
- `OPENAI_MAX_RETRIES` (padrao `3`)

Exemplo de `FIREBASE_NAMESPACE_CONFIG_JSON`:

```json
{
  "talk__dev": {
    "project_id": "talk-dev",
    "storage_bucket": "talk-dev.appspot.com",
    "credentials_file": "/secrets/talk-dev-firebaseadmin.json"
  },
  "talk__stg": {
    "project_id": "talk-stg",
    "storage_bucket": "talk-stg.appspot.com",
    "credentials_file": "/secrets/talk-stg-firebaseadmin.json"
  },
  "talk__prd": {
    "project_id": "talk-prd",
    "storage_bucket": "talk-prd.appspot.com",
    "credentials_file": "/secrets/talk-prd-firebaseadmin.json"
  },
  "ellevo-connect__dev": {
    "project_id": "ellevo-connect-dev",
    "storage_bucket": "ellevo-connect-dev.appspot.com",
    "credentials_file": "/secrets/ellevo-connect-dev-firebaseadmin.json"
  },
  "ellevo-connect__stg": {
    "project_id": "ellevo-connect-stg",
    "storage_bucket": "ellevo-connect-stg.appspot.com",
    "credentials_file": "/secrets/ellevo-connect-stg-firebaseadmin.json"
  },
  "ellevo-connect__prd": {
    "project_id": "ellevo-connect-prd",
    "storage_bucket": "ellevo-connect-prd.appspot.com",
    "credentials_file": "/secrets/ellevo-connect-prd-firebaseadmin.json"
  }
}
```
