#!/usr/bin/env python3
import argparse
import asyncio
import json
import wave

import numpy as np
import websockets


def read_wave(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("Use WAV mono PCM16.")
        sample_rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return samples, sample_rate


async def run(url: str, wav_path: str, chunk_ms: int) -> None:
    samples, sr = read_wave(wav_path)

    chunk_size = int(sr * (chunk_ms / 1000.0))
    if chunk_size <= 0:
        raise ValueError("chunk_ms invalido")

    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        recv_task = asyncio.create_task(receive_loop(ws))
        await ws.send(json.dumps({"event": "config", "sample_rate": sr}))

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(chunk_ms / 1000.0)

        await ws.send(json.dumps({"event": "end"}))
        await recv_task


async def receive_loop(ws) -> None:
    async for msg in ws:
        print(msg)
        try:
            payload = json.loads(msg)
            if payload.get("type") == "done":
                break
        except json.JSONDecodeError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="ws://localhost:3001/ws/transcribe",
        help="WebSocket endpoint",
    )
    parser.add_argument(
        "--wav",
        default="/opt/stack/stt/models/pt_br.wav",
        help="Arquivo WAV mono PCM16 16 kHz",
    )
    parser.add_argument("--chunk-ms", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(run(args.url, args.wav, args.chunk_ms))


if __name__ == "__main__":
    main()
