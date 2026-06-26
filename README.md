# Talk Transcript STT (Fila Assíncrona)

Servico STT CPU-only para ingestao assíncrona de chunks enviados pelo agente LiveKit.

## Estrutura

- `docker-compose.yml`: sobe o servico `stt`
- `stt/app.py`: API HTTP (`/health`, `/v1/sessions/*`) + worker
- `stt/Dockerfile`: imagem
- `stt/bootstrap_models.sh`: download automatico do modelo
- `stt/requirements.txt`: dependencias Python

## Endpoints

- `GET /health`
- `POST /v1/sessions/start`
- `POST /v1/sessions/chunk`
- `POST /v1/sessions/end`
- `POST /v1/admin/summary/reprocess`
- `GET /v1/admin/summary/reprocess/status`

## Subir

```bash
docker compose up --build -d stt
docker compose logs -f stt
```


## Notas operacionais

- autenticacao HMAC obrigatoria para ingestao
- fila unica em SQLite (WAL) com worker unico
- respostas de ingestao: `202`, `409` (conflito), `429` (fila cheia)
- suporte multi-ambiente por namespace no `room_name` (`<namespace>__<room_id>`)
- integracao Firestore/Storage opcional por variavel de ambiente
  - Firestore guarda indice/ponteiros
  - Storage guarda `transcript.json`/`summary.json` por janela configuravel
- timeout automatico de sala por inatividade (`ROOM_INACTIVITY_TIMEOUT_SECONDS`, padrao 1800s)
  - a sala recebe `room_end` por timeout quando nao recebe novos `chunk`
  - a finalizacao ocorre apos drenar `queued/processing` (sem descarte imediato)
