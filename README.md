# STT sherpa-onnx (PT-BR)

Servico de Speech-to-Text em CPU com:
- transcricao de arquivo via HTTP
- transcricao em tempo real via WebSocket
- suporte a portugues do Brasil
- VAD interno opcional (habilitado por padrao)

## Estrutura do projeto

- `docker-compose.yml`: sobe o servico `stt`
- `stt/app.py`: API (`/health`, `/transcribe`, `/ws/transcribe`)
- `stt/Dockerfile`: imagem do servico
- `stt/bootstrap_models.sh`: download automatico de modelos
- `stt/ws_client_test.py`: cliente minimo de teste WebSocket
- `stt/test_http_pt.sh`: teste rapido HTTP

## Requisitos

- Docker + Docker Compose
- VPS/host sem GPU (CPU-only)
- Porta `3001` liberada para acesso externo (se necessario)

## Como subir

```bash
cd /opt/stack
docker compose up --build -d stt
docker compose logs -f stt
```

## Endpoints

- `GET /health`: status e metadados do modelo
- `POST /transcribe`: upload de arquivo de audio
- `WS /ws/transcribe`: streaming de audio PCM16 mono

## Testes locais

Health:
```bash
curl -s http://localhost:3001/health
```

HTTP com audio PT-BR:
```bash
/opt/stack/stt/test_http_pt.sh
```

WebSocket realtime:
```bash
docker exec stt python /app/ws_client_test.py \
  --url ws://localhost:3001/ws/transcribe \
  --wav /models/pt_br.wav
```

## VAD interno (opcional)

Variavel:
- `USE_INTERNAL_VAD=true` (padrao): servidor segmenta fala/silencio
- `USE_INTERNAL_VAD=false`: cliente/orquestrador (ex.: LiveKit) controla segmentos

Impacto no WebSocket:
- com VAD interno: pode gerar varios eventos `final` na mesma conexao
- sem VAD interno: normalmente 1 evento `final` ao receber `{"event":"end"}`

## Validacao externa (Windows)

```powershell
curl.exe "http://SEU_DNS_PUBLICO:3001/health"
curl.exe -X POST "http://SEU_DNS_PUBLICO:3001/transcribe" -F "file=@C:\caminho\audio.wav"
```

Para WebSocket externo:
- URL: `ws://SEU_DNS_PUBLICO:3001/ws/transcribe`
- enviar `{"event":"config","sample_rate":16000}` e depois chunks binarios PCM16 mono
- finalizar com `{"event":"end"}`
