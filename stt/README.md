# STT sherpa-onnx (PT-BR + WebSocket)

Servico STT em CPU com:
- `POST /transcribe` (arquivo, compativel com endpoint atual)
- `GET /health`
- `WS /ws/transcribe` (tempo real com VAD e hipoteses parciais/finais)

## Modelo e estrategia

Modelo principal: `sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc-int8`
(portugues, int8).

Observacao tecnica: atualmente nao ha modelo PT-BR de streaming nativo no release com latencia baixa como zipformer streaming. Para manter PT-BR com boa qualidade, o WebSocket usa VAD + decodificacao incremental por segmento (realtime pratico, CPU-only).

## Bootstrap automatico

Ao subir o container, `bootstrap_models.sh` baixa automaticamente (se faltarem):
- modelo PT-BR
- `silero_vad.onnx`
- `pt_br.wav` para teste

Tudo fica em `/opt/stack/stt/models` via volume `./stt/models:/models`.

## Subir o servico

```bash
cd /opt/stack
docker compose up --build -d stt
docker compose logs -f stt
```

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

## Protocolo WebSocket (`/ws/transcribe`)

Entrada:
- mensagens binarias com chunks PCM16 mono (`s16le`, ideal 16 kHz)
- para finalizar envio: `{"event":"end"}` (texto JSON)

Saida:
- `{"type":"partial","segment_id":N,"text":"..."}`
- `{"type":"final","segment_id":N,"text":"...","start_seconds":X}`
- `{"type":"done"}`

## VAD interno opcional (`USE_INTERNAL_VAD`)

Configuracao:
- `USE_INTERNAL_VAD=false` (padrao): desliga segmentacao por VAD interno.
- `USE_INTERNAL_VAD=true`: usa `silero_vad.onnx` no servidor.

Quando usar `true`:
- cliente envia audio bruto continuo e espera finalizacao automatica por fala/silencio.
- melhor para clientes simples sem controle de segmentacao.

Quando usar `false` (ex.: LiveKit controlando segmentos):
- a aplicacao externa decide quando comecar/encerrar segmento.
- no websocket, o servidor passa a tratar toda a sessao como um unico segmento (`segment_id=0`) e emite `final` no evento `{"event":"end"}`.

Impacto no endpoint `/ws/transcribe`:
- com VAD interno: podem existir varios `final` por conexao (segmentos detectados).
- sem VAD interno: normalmente 1 `final` por conexao, ao final do stream.

## Validacao externa da VPS

HTTP:
```bash
curl -X POST "http://SEU_DNS_PUBLICO:3001/transcribe" \
  -F "file=@C:\\caminho\\pt_br.wav"
```

WebSocket:
- URL: `ws://SEU_DNS_PUBLICO:3001/ws/transcribe`
- confirme firewall/security group liberando TCP `3001`.
