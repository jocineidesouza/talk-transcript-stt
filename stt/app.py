import asyncio
import json
import logging
import os
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


MODEL_DIR = Path(
    os.environ.get(
        "MODEL_DIR",
        "/models/sherpa-onnx-nemo-stt_pt_fastconformer_hybrid_large_pc-int8",
    )
)
VAD_MODEL = Path(os.environ.get("VAD_MODEL", "/models/silero_vad.onnx"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
FEATURE_DIM = int(os.environ.get("FEATURE_DIM", "80"))
NUM_THREADS = int(os.environ.get("SHERPA_NUM_THREADS", "3"))
DECODE_MAX_CONCURRENCY = int(os.environ.get("DECODE_MAX_CONCURRENCY", "1"))
VAD_MIN_SILENCE_SECONDS = float(os.environ.get("VAD_MIN_SILENCE_SECONDS", "0.25"))
VAD_MIN_SPEECH_SECONDS = float(os.environ.get("VAD_MIN_SPEECH_SECONDS", "0.05"))
PARTIAL_EMIT_SECONDS = float(os.environ.get("WS_PARTIAL_EMIT_SECONDS", "0.8"))
TAIL_PADDING_SECONDS = float(os.environ.get("TAIL_PADDING_SECONDS", "0.35"))
MODEL_LANGUAGE = os.environ.get("MODEL_LANGUAGE", "pt-BR")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "nemo_ctc_offline_vad_streaming")
STREAMING_ENABLED = True
USE_INTERNAL_VAD = os.environ.get("USE_INTERNAL_VAD", "false").lower() == "true"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stt")

app = FastAPI(title="STT sherpa-onnx")
decode_semaphore = asyncio.Semaphore(DECODE_MAX_CONCURRENCY)


def _require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo ausente: {path}")
    return str(path)


def build_offline_recognizer() -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=_require_file(MODEL_DIR / "model.int8.onnx"),
        tokens=_require_file(MODEL_DIR / "tokens.txt"),
        num_threads=NUM_THREADS,
        provider="cpu",
        decoding_method="greedy_search",
        debug=False,
    )


def create_vad() -> sherpa_onnx.VoiceActivityDetector:
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = _require_file(VAD_MODEL)
    config.silero_vad.min_silence_duration = VAD_MIN_SILENCE_SECONDS
    config.silero_vad.min_speech_duration = VAD_MIN_SPEECH_SECONDS
    config.sample_rate = SAMPLE_RATE
    config.num_threads = max(1, min(NUM_THREADS, 2))
    config.provider = "cpu"
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)


recognizer = build_offline_recognizer()


def convert_to_wav(source: Path, target: Path) -> None:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(target),
    ]
    subprocess.run(cmd, check=True)


def read_wave_file(path: Path) -> tuple[np.ndarray, int, float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Arquivo deve ter 1 canal.")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Arquivo deve ser PCM 16 bits.")
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / sample_rate
        samples = wav_file.readframes(num_frames)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        return samples_float32, sample_rate, duration


def decode_samples(samples: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def resample_linear(
    samples: np.ndarray, input_sample_rate: int, output_sample_rate: int
) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return samples
    if samples.size == 0:
        return samples
    ratio = output_sample_rate / float(input_sample_rate)
    output_size = max(1, int(round(samples.size * ratio)))
    x_old = np.linspace(0, samples.size - 1, num=samples.size, dtype=np.float32)
    x_new = np.linspace(0, samples.size - 1, num=output_size, dtype=np.float32)
    return np.interp(x_new, x_old, samples).astype(np.float32)


async def decode_samples_async(samples: np.ndarray, sample_rate: int) -> str:
    async with decode_semaphore:
        return await asyncio.to_thread(decode_samples, samples, sample_rate)


def decode_with_tail_padding(samples: np.ndarray, sample_rate: int) -> str:
    padded = np.concatenate(
        [samples, np.zeros(int(sample_rate * TAIL_PADDING_SECONDS), dtype=np.float32)]
    )
    return decode_samples(padded, sample_rate)


async def decode_final_async(samples: np.ndarray, sample_rate: int) -> str:
    async with decode_semaphore:
        return await asyncio.to_thread(decode_with_tail_padding, samples, sample_rate)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_dir": str(MODEL_DIR),
        "model_type": MODEL_TYPE,
        "expected_language": MODEL_LANGUAGE,
        "streaming_enabled": STREAMING_ENABLED,
        "use_internal_vad": USE_INTERNAL_VAD,
        "num_threads": NUM_THREADS,
        "decode_max_concurrency": DECODE_MAX_CONCURRENCY,
        "sample_rate": SAMPLE_RATE,
        "feature_dim": FEATURE_DIM,
        "vad_model": str(VAD_MODEL),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        return JSONResponse({"error": "Envie um arquivo no campo 'file'."}, status_code=400)

    with tempfile.TemporaryDirectory(prefix="stt-http-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_path = tmp_path / file.filename
        wav_path = tmp_path / "input.wav"
        data = await file.read()
        source_path.write_bytes(data)

        try:
            convert_to_wav(source_path, wav_path)
            samples, sample_rate, duration = read_wave_file(wav_path)
            text = await decode_final_async(samples, sample_rate)
        except subprocess.CalledProcessError:
            return JSONResponse({"error": "Falha na conversao de audio."}, status_code=400)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(
        {
            "text": text,
            "duration_seconds": round(duration, 3),
            "model": MODEL_DIR.name,
            "language": MODEL_LANGUAGE,
        }
    )


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    connection_id = str(uuid.uuid4())[:8]
    logger.info("ws connected id=%s", connection_id)

    vad = create_vad() if USE_INTERNAL_VAD else None
    chunk_buffer = np.array([], dtype=np.float32)
    window_size = int(vad.config.silero_vad.window_size) if vad else 0
    partial_segment_id = 0
    emitted_partial = ""
    last_partial_len = 0
    partial_emit_samples = int(PARTIAL_EMIT_SECONDS * SAMPLE_RATE)
    input_sample_rate = SAMPLE_RATE
    no_vad_audio = np.array([], dtype=np.float32)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            if text_data is not None:
                if text_data in ("Done", "__END__", '{"event":"end"}'):
                    break
                try:
                    payload = json.loads(text_data)
                    event = payload.get("event")
                    if event == "end":
                        break
                    if event == "config":
                        sr = int(payload.get("sample_rate", SAMPLE_RATE))
                        if sr < 8000 or sr > 48000:
                            await websocket.send_json(
                                {"type": "error", "error": "sample_rate invalido."}
                            )
                            continue
                        input_sample_rate = sr
                        logger.info(
                            "ws config id=%s sample_rate_in=%s",
                            connection_id,
                            input_sample_rate,
                        )
                        await websocket.send_json(
                            {
                                "type": "ack",
                                "sample_rate_in": input_sample_rate,
                                "sample_rate_model": SAMPLE_RATE,
                            }
                        )
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "error": "Mensagem de controle invalida."}
                    )
                    continue

            raw = message.get("bytes")
            if raw is None:
                continue

            # Input protocol: binary PCM s16le mono chunks.
            samples_int16 = np.frombuffer(raw, dtype=np.int16)
            if samples_int16.size == 0:
                continue

            samples = samples_int16.astype(np.float32) / 32768.0
            if input_sample_rate != SAMPLE_RATE:
                samples = resample_linear(samples, input_sample_rate, SAMPLE_RATE)
            if not USE_INTERNAL_VAD:
                no_vad_audio = np.concatenate([no_vad_audio, samples])
                if len(no_vad_audio) - last_partial_len >= partial_emit_samples:
                    partial = await decode_samples_async(no_vad_audio, SAMPLE_RATE)
                    last_partial_len = len(no_vad_audio)
                    if partial and partial != emitted_partial:
                        emitted_partial = partial
                        await websocket.send_json(
                            {"type": "partial", "segment_id": 0, "text": partial}
                        )
                continue

            chunk_buffer = np.concatenate([chunk_buffer, samples])

            while chunk_buffer.size >= window_size:
                frame = chunk_buffer[:window_size]
                chunk_buffer = chunk_buffer[window_size:]
                vad.accept_waveform(frame)

                if vad.is_speech_detected():
                    current = vad.current_segment.samples
                    if len(current) - last_partial_len >= partial_emit_samples:
                        partial = await decode_samples_async(current, SAMPLE_RATE)
                        last_partial_len = len(current)
                        if partial and partial != emitted_partial:
                            emitted_partial = partial
                            await websocket.send_json(
                                {
                                    "type": "partial",
                                    "segment_id": partial_segment_id,
                                    "text": partial,
                                }
                            )
                            logger.debug(
                                "ws partial id=%s segment=%s text=%s",
                                connection_id,
                                partial_segment_id,
                                partial,
                            )

                while not vad.empty():
                    segment = vad.front
                    segment_samples = np.array(segment.samples, copy=True)
                    segment_start = segment.start
                    vad.pop()
                    final_text = await decode_final_async(segment_samples, SAMPLE_RATE)
                    await websocket.send_json(
                        {
                            "type": "final",
                            "segment_id": partial_segment_id,
                            "text": final_text,
                            "start_seconds": round(segment_start / SAMPLE_RATE, 3),
                        }
                    )
                    logger.info(
                        "ws final id=%s segment=%s text=%s",
                        connection_id,
                        partial_segment_id,
                        final_text,
                    )
                    partial_segment_id += 1
                    emitted_partial = ""
                    last_partial_len = 0

        if not USE_INTERNAL_VAD:
            if no_vad_audio.size > 0:
                final_text = await decode_final_async(no_vad_audio, SAMPLE_RATE)
                await websocket.send_json(
                    {"type": "final", "segment_id": 0, "text": final_text, "start_seconds": 0.0}
                )
                logger.info(
                    "ws final id=%s segment=%s text=%s",
                    connection_id,
                    0,
                    final_text,
                )
            await websocket.send_json({"type": "done"})
            return

        if chunk_buffer.size > 0:
            vad.accept_waveform(chunk_buffer)
        vad.flush()

        while not vad.empty():
            segment = vad.front
            segment_samples = np.array(segment.samples, copy=True)
            segment_start = segment.start
            vad.pop()
            final_text = await decode_final_async(segment_samples, SAMPLE_RATE)
            await websocket.send_json(
                {
                    "type": "final",
                    "segment_id": partial_segment_id,
                    "text": final_text,
                    "start_seconds": round(segment_start / SAMPLE_RATE, 3),
                }
            )
            logger.info(
                "ws final id=%s segment=%s text=%s",
                connection_id,
                partial_segment_id,
                final_text,
            )
            partial_segment_id += 1
            last_partial_len = 0

        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("ws disconnected id=%s", connection_id)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ws error id=%s err=%s", connection_id, exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:  # pylint: disable=broad-except
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # pylint: disable=broad-except
            pass
        logger.info("ws closed id=%s", connection_id)
