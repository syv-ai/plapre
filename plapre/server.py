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
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from plapre.inference import SAMPLE_RATE, Plapre

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tts: Plapre | None = None
_vocoder_sem: asyncio.Semaphore | None = None
_async_mode: bool = False


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts, _vocoder_sem, _async_mode

    checkpoint = os.environ.get("PLAPRE_CHECKPOINT", "syvai/plapre-nano")
    quant = os.environ.get("PLAPRE_QUANT", "q8_0")
    gpu_mem = float(os.environ.get("PLAPRE_GPU_MEM", "0.5"))
    max_len = int(os.environ.get("PLAPRE_MAX_MODEL_LEN", "512"))
    _async_mode = os.environ.get("PLAPRE_ASYNC", "1") == "1"
    mode_str = "async" if _async_mode else "sync"
    log.info("Loading model %s (quant=%s, gpu_mem=%.2f, max_len=%d, mode=%s) …",
             checkpoint, quant, gpu_mem, max_len, mode_str)
    _tts = Plapre(
        checkpoint=checkpoint,
        quant=quant,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        use_async=_async_mode,
    )
    # Serialize vocoder calls — Vocos cuFFT needs exclusive GPU access to avoid OOM
    _vocoder_sem = asyncio.Semaphore(1)
    log.info("Model ready (%s mode).", mode_str)
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

    async def generate():
        if _async_mode:
            # Async: all sentences submitted concurrently, vLLM batches internally
            log.info("Generating %d sentence(s) via AsyncLLM", len(sentences))
            all_tokens = await _tts.generate_tokens_async(
                sentences, spk, **gen_kwargs
            )
        else:
            # Sync: batch via vLLM sync engine on thread pool
            log.info("Generating %d sentence(s) via sync LLM", len(sentences))
            all_tokens = await asyncio.to_thread(
                _tts._generate_tokens, sentences, spk, **gen_kwargs
            )

        # Vocode sequentially via semaphore — cuFFT needs exclusive GPU access
        for i, tokens in enumerate(all_tokens):
            async with _vocoder_sem:
                try:
                    audio = await asyncio.to_thread(
                        _tts._tokens_to_audio, tokens, spk
                    )
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    log.warning("Vocoder failed for sentence %d: %s", i, e)
                    audio = None
            if audio is not None:
                yield _float32_to_pcm16(audio)
                if i < len(sentences) - 1:
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
    parser.add_argument("--sync", action="store_true", help="Use sync vLLM engine (default: async)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    os.environ["PLAPRE_CHECKPOINT"] = args.checkpoint
    os.environ["PLAPRE_GPU_MEM"] = str(args.gpu_mem)
    os.environ["PLAPRE_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["PLAPRE_ASYNC"] = "0" if args.sync else "1"
    uvicorn.run(
        "plapre.server:app",
        host=args.host,
        port=args.port,
        http="httptools",
    )


if __name__ == "__main__":
    main()
