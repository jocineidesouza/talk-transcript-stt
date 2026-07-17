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

- `VERTICALS/{vertical}/COMPANIES/{slug}/ROOMS/{room_id}/SESSIONS/{call_session_id}`: metadados, status, heartbeat, ponteiros de resumo/transcricao final e ata textual final

Storage:

- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/minutes/{minute_index}/transcript.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/minutes/{minute_index}/summary.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/minutes/{minute_index}/summary_text.txt` (quando `SUMMARY_MODE=ata_progressiva`)
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/summary/accumulated.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/summary/accumulated.txt` (quando `SUMMARY_MODE=ata_progressiva`)
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/summary/accumulated_meta.json` (quando `SUMMARY_MODE=ata_progressiva`)
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/final/final_summary.json`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/final/final_summary_text.txt`
- `VERTICALS/{vertical}/COMPANIES/{slug}/TRANSCRIPT/{room_id}/{call_session_id}/final/final_summary_temp.json` (quando houver erro de contrato no final)

## Summary LLM (OpenRouter/OpenAI, opcional)

- habilitado por `SUMMARY_ENABLED=true`
- modo definido por `SUMMARY_MODE`: `rolling_json` (padrao atual) ou `ata_progressiva`
- em `ata_progressiva`, a fonte final e definida por `SUMMARY_PROGRESSIVE_FINAL_SOURCE` (`auto`, `delta_only`, `full_transcript`) e limitada por `SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS` no modo `auto`
- provedor definido por `SUMMARY_PROVIDER` (`openrouter` ou `openai`, padrao `openrouter`)
- secrets esperados:
  - `OPENROUTER_APIKEY_FILE` (padrao `/secrets/openrouter_apikey.json`)
  - `OPENAI_APIKEY_FILE` (padrao `/secrets/openai_apikey.json`)
- formato do arquivo de secret: `{"api_key":"..."}`
- para OpenRouter, endpoint padrao `https://openrouter.ai/api/v1` e headers opcionais `OPENROUTER_HTTP_REFERER` / `OPENROUTER_X_TITLE`
- se secret estiver ausente/invalido, STT continua; resumo fica desabilitado com warning

## Reprocessamento administrativo de summary

### POST `/v1/admin/summary/reprocess`

Reprocessa do zero os resumos da sessao alvo:
- valida `namespace + vertical + slug + room_id + call_session_id` contra SQLite
- força finalizacao logica se a sessao ainda estiver ativa
- reseta `summary_tasks` e `minute_exports.summary_json_path`
- reseta `summary/accumulated.json` no modo `rolling_json`; no modo `ata_progressiva`, limpa `summary/accumulated.txt`, `summary/accumulated_meta.json` e snapshots textuais conhecidos
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

## Uso e cobranca da transcricao

Os arquivos `minutes/*/transcript.json` e `final/final_transcript.json` carregam metadados do provedor e uso
agregado no campo `usage`. Cada item de `lines` tambem possui `transcription` e `usage` proprios.

- `provider` e `model` identificam o motor que gerou a linha.
- `operationId` e deterministico por sessao, participante e sequencia, permitindo idempotencia.
- `audioDurationMs` e a duracao do chunk; `processingTimeMs` e o tempo do decode local.
- `costMicros` e custo em microdolares (`1 USD = 1.000.000 micros`), calculado pela duracao do audio.
- Tokens ficam `null` quando o provedor STT nao os fornece; nao sao estimados artificialmente.
- O `usage` do arquivo e a soma dos valores de `lines[].usage`.

No `self_hosted`, `STT_PRICE_USD_PER_HOUR` define o preco de referencia usado para consolidacao. Ao trocar de
provedor, o contrato permanece o mesmo e apenas `STT_PROVIDER`, modelo, identificadores retornados e regra de
preco precisam ser adaptados.

### Custos do resumo via OpenRouter

Enquanto `SUMMARY_PROVIDER=openrouter`, cada chamada ao endpoint `/responses` e registrada no SQLite e o arquivo
`final/summary_costs.json` e enviado ao Storage quando a ata final textual e concluida. O arquivo inclui chamadas
de resumo por minuto, acumulado, JSON final, HTML final, ata progressiva, retries, tokens, cache, reasoning,
generation ID, request ID, modelo e custo retornado pelo OpenRouter.

O custo do OpenRouter e convertido para `costMicros` (`1 USD = 1.000.000 micros`). Cada retry possui uma entrada
separada e seu `operationId` inclui a tentativa logica e a tentativa de rede. Reprocessamentos reutilizam o
registro persistido e nao criam custo novo quando nenhuma chamada e feita.

### Integração futura com OpenAI

A integracao OpenAI ainda nao esta implementada. Quando necessaria, ela deve reutilizar a tabela
`summary_requests` e o mesmo `summary_costs.json`, mapeando `input_tokens`, `output_tokens`,
`input_tokens_details.cached_tokens` e `output_tokens_details.reasoning_tokens`. O custo devera ser calculado
por uma tabela de precos versionada da OpenAI, pois o contrato atual de custos e especifico do OpenRouter.

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
- `SUMMARY_ENABLED` (padrao `false`)
- `SUMMARY_PROVIDER` (padrao `openrouter`)
- `SUMMARY_MODEL_MINUTE` (padrao `openai/gpt-4.1-mini`)
- `SUMMARY_MODEL_ACCUMULATED` (padrao `openai/gpt-4.1-mini`)
- `SUMMARY_MODEL_FINAL` (padrao `openai/gpt-4.1-mini`)
- `SUMMARY_MODEL_FINAL_TEXT` (padrao `openai/gpt-4.1-mini`)
- `SUMMARY_FINAL_TEXT_FORMAT` (`markdown|html|text`, padrao `html`)
- `SUMMARY_REQUEST_TIMEOUT_SECONDS` (padrao `300`)
- `SUMMARY_REQUEST_RETRIES` (padrao `2`, total de 3 tentativas por request)
- `SUMMARY_REQUEST_RETRY_BASE_SECONDS` (padrao `1.5`)
- `SUMMARY_MAX_RETRIES` (padrao `3`)
- `SUMMARY_ACCUMULATED_MAX_ITEMS` (padrao `40`)
- `OPENAI_APIKEY_FILE` (padrao `/secrets/openai_apikey.json`)
- `OPENROUTER_APIKEY_FILE` (padrao `/secrets/openrouter_apikey.json`)
- `OPENAI_BASE_URL` (padrao `https://api.openai.com/v1`)
- `OPENROUTER_BASE_URL` (padrao `https://openrouter.ai/api/v1`)
- `OPENROUTER_HTTP_REFERER` (opcional)
- `OPENROUTER_X_TITLE` (opcional)
- `STT_PROVIDER` (padrao `self_hosted`)
- `STT_MODEL` (padrao derivado de `MODEL_DIR`)
- `STT_PRICE_USD_PER_HOUR` (padrao `1.00`, referencia de cobranca por hora de audio)
- `STT_PRICING_SOURCE` (padrao `azure_speech_standard_reference`)
- `STT_PRICING_VERSION` (padrao `2026-07-17`)

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
