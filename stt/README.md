# Talk Transcript STT Service (Async HTTP)

Servico de transcricao assincrona para ingestao de chunks de audio vindos do agente LiveKit.

## Endpoints

- `GET /health`
- `POST /v1/sessions/start`
- `POST /v1/sessions/chunk` (`multipart/form-data` com `meta` + `audio`)
- `POST /v1/sessions/end`
- `POST /v1/admin/summary/reprocess` (admin, sem HMAC)
- `GET /v1/admin/summary/reprocess/status` (admin, sem HMAC)

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

Observacao:
- Os endpoints administrativos de reprocessamento de summary (`/v1/admin/summary/reprocess*`) nao exigem HMAC.

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
- `POST /v1/sessions/start` exige `LIVEKIT_ROOM_INDEX/{room_name}` com `vertical` e `slug`
- configuracao por `FIREBASE_NAMESPACE_CONFIG_JSON` (`namespace -> config`)
- cada namespace usa app/credencial propria (sem fallback silencioso)
- nao grava transcricao completa no Firestore por chunk

Firestore:

- `VERTICALS/{vertical}/COMPANIES/{slug}/ROOMS/{room_id}/SESSIONS/{call_session_id}`: metadados, status, heartbeat, ponteiros de resumo/transcricao final

Storage:

- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/minutes/{minute_index}/transcript.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/minutes/{minute_index}/summary.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/summary/accumulated.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/final/final_summary.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/final/final_summary_temp.json` (quando houver erro de contrato no final)

## OpenAI Summary (opcional)

- habilitado por `OPENAI_SUMMARY_ENABLED=true`
- secret esperado em `OPENAI_APIKEY_FILE` (padrao `/secrets/openai_apikey.json`)
- formato do arquivo: `{"api_key":"..."}` (ver `stt/secrets/openai_apikey.example.json`)
- se secret estiver ausente/invalido, STT continua; resumo fica desabilitado com warning

## Reprocessamento administrativo de summary

### POST `/v1/admin/summary/reprocess`

Reprocessa do zero os resumos da sessao alvo:
- valida `namespace + vertical + slug + room_id + call_session_id` contra SQLite
- força finalizacao logica se a sessao ainda estiver ativa
- reseta `summary_tasks` e `minute_exports.summary_json_path`
- reseta `summary/accumulated.json`
- reencadeia processamento assíncrono de minutos + final

Body JSON:

```json
{
  "namespace": "talk__dev",
  "vertical": "HEALTH",
  "slug": "acme",
  "room_id": "roomA",
  "call_session_id": "RM_session-1"
}
```

Exemplo curl:

```bash
curl -X POST "http://localhost:8000/v1/admin/summary/reprocess" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace":"talk__dev",
    "vertical":"HEALTH",
    "slug":"acme",
    "room_id":"roomA",
    "call_session_id":"RM_session-1"
  }'
```

### GET `/v1/admin/summary/reprocess/status`

Consulta o progresso e, quando pronto, retorna `final_summary`.

Exemplo curl:

```bash
curl "http://localhost:8000/v1/admin/summary/reprocess/status?namespace=talk__dev&vertical=HEALTH&slug=acme&room_id=roomA&call_session_id=RM_session-1"
```

Campos principais no retorno:
- `overall_status`: `pending | processing | error | error_exhausted | done`
- `progress`: contagem de tarefas de minuto
- `final_task`: estado da tarefa final (`minute_index=-1`)
- `minute_tasks`: estado por minuto
- `final_summary`: presente quando `final_task.status=done`
- `final_error_details`: presente quando `final_task.exhausted=true`

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
- `OPENAI_MODEL_ACCUMULATED_SUMMARY` (padrao `gpt-4.1-mini`)
- `OPENAI_MODEL_FINAL_SUMMARY` (padrao `gpt-4.1-mini`)
- `OPENAI_REQUEST_TIMEOUT_SECONDS` (padrao `300`)
- `OPENAI_REQUEST_RETRIES` (padrao `2`, total de 3 tentativas por request)
- `OPENAI_MAX_RETRIES` (padrao `3`)
- `OPENAI_ACCUMULATED_MAX_ITEMS` (padrao `40`)

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
