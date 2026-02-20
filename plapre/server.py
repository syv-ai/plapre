"""
FastAPI server for Plapre Danish TTS with chunked PCM streaming.

Start with:
    plapre-serve --port 8000

Or:
    uvicorn plapre.server:app
"""

import asyncio
import logging
import os
import struct
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from plapre.inference import SAMPLE_RATE, Plapre

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tts: Plapre | None = None
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts

    checkpoint = os.environ.get("PLAPRE_CHECKPOINT", "syvai/plapre-nano")
    quant = os.environ.get("PLAPRE_QUANT", "q8_0")
    gpu_mem = float(os.environ.get("PLAPRE_GPU_MEM", "0.5"))
    max_len = int(os.environ.get("PLAPRE_MAX_MODEL_LEN", "512"))
    log.info("Loading model %s (quant=%s, gpu_mem=%.2f, max_len=%d) …", checkpoint, quant, gpu_mem, max_len)
    _tts = Plapre(
        checkpoint=checkpoint,
        quant=quant,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
    )
    log.info("Model ready.")
    yield
    _tts = None


app = FastAPI(title="Plapre TTS", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str
    speaker: str | None = None
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to 16-bit signed LE PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        spk = _tts._resolve_speaker(req.speaker, None, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    gen_kwargs = dict(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    sentences = _tts._split_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="No text provided")

    silence_samples = int(0.3 * SAMPLE_RATE)
    silence_bytes = struct.pack(f"<{silence_samples}h", *([0] * silence_samples))

    prefetch_count = 2  # sequential first N for low latency

    async def generate():
        async with _lock:
            # --- Phase 1: Sequential for fast first audio ---
            seq_limit = min(prefetch_count, len(sentences))
            for i in range(seq_limit):
                log.info("Sequential sentence %d/%d: %s", i + 1, len(sentences), sentences[i])
                audio = await asyncio.to_thread(
                    _tts._generate_audio, sentences[i], spk, **gen_kwargs,
                )
                if audio is not None:
                    yield _float32_to_pcm16(audio)
                    if i < len(sentences) - 1:
                        yield silence_bytes

            # --- Phase 2: Batch remaining via vLLM ---
            remaining = sentences[seq_limit:]
            if remaining:
                log.info("Batch generating %d remaining sentences", len(remaining))
                results = await asyncio.to_thread(
                    _tts._generate_audio_batch, remaining, spk, **gen_kwargs,
                )
                for j, audio in enumerate(results):
                    if audio is not None:
                        yield _float32_to_pcm16(audio)
                        global_idx = seq_limit + j
                        if global_idx < len(sentences) - 1:
                            yield silence_bytes

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
        },
    )


@app.get("/v1/speakers")
async def speakers():
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"speakers": list(_tts.speakers.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Plapre TTS server")
    parser.add_argument(
        "--checkpoint", default="syvai/plapre-nano",
        help="HuggingFace checkpoint (default: syvai/plapre-nano)",
    )
    parser.add_argument("--gpu-mem", type=float, default=0.5, help="GPU memory utilization (default: 0.5)")
    parser.add_argument("--max-model-len", type=int, default=512, help="Max model length (default: 512)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    os.environ["PLAPRE_CHECKPOINT"] = args.checkpoint
    os.environ["PLAPRE_GPU_MEM"] = str(args.gpu_mem)
    os.environ["PLAPRE_MAX_MODEL_LEN"] = str(args.max_model_len)
    uvicorn.run(
        "plapre.server:app",
        host=args.host,
        port=args.port,
        http="httptools",
    )


if __name__ == "__main__":
    main()
