"""
FastAPI server for Plapre Danish TTS with chunked PCM streaming.

Start with:
    plapre-serve --model plapre-nano-q8_0 --port 8000

Or:
    PLAPRE_MODEL=plapre-nano-q8_0 uvicorn plapre.server:app
"""

import asyncio
import logging
import os
import struct
import tempfile
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from plapre.inference import SAMPLE_RATE, Plapre

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "plapre-pico-f16": ("syvai/plapre-pico", "f16"),
    "plapre-pico-q8_0": ("syvai/plapre-pico", "q8_0"),
    "plapre-pico-q6_k": ("syvai/plapre-pico", "q6_k"),
    "plapre-pico-q4_k_m": ("syvai/plapre-pico", "q4_k_m"),
    "plapre-pico-q4_0": ("syvai/plapre-pico", "q4_0"),
    "plapre-nano-f16": ("syvai/plapre-nano", "f16"),
    "plapre-nano-q8_0": ("syvai/plapre-nano", "q8_0"),
    "plapre-nano-q6_k": ("syvai/plapre-nano", "q6_k"),
    "plapre-nano-q4_k_m": ("syvai/plapre-nano", "q4_k_m"),
    "plapre-nano-q4_0": ("syvai/plapre-nano", "q4_0"),
}
DEFAULT_MODEL = "plapre-pico-q8_0"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_tts: Plapre | None = None
_loaded_model: str = ""
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts, _loaded_model

    model_name = os.environ.get("PLAPRE_MODEL", DEFAULT_MODEL)
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}"
        )
    checkpoint, quant = MODELS[model_name]
    log.info("Loading model %s (checkpoint=%s, quant=%s) …", model_name, checkpoint, quant)
    _tts = Plapre(checkpoint, quant=quant)
    _loaded_model = model_name
    log.info("Model %s ready.", model_name)
    yield
    _tts = None


app = FastAPI(title="Plapre TTS", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str
    speaker: str | None = None
    model: str = DEFAULT_MODEL
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

    if req.model != _loaded_model:
        raise HTTPException(
            status_code=400,
            detail=f"Requested model '{req.model}' but server loaded '{_loaded_model}'. "
                   f"Restart the server with --model {req.model}.",
        )

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

    silence_samples = int(0.1 * SAMPLE_RATE)
    silence_bytes = struct.pack(f"<{silence_samples}h", *([0] * silence_samples))

    async def generate():
        async with _lock:
            for i, sent in enumerate(sentences):
                log.info("Generating sentence %d/%d: %s", i + 1, len(sentences), sent)
                audio = await asyncio.to_thread(
                    _tts._generate_audio, sent, spk, **gen_kwargs,
                )
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


@app.post("/v1/speakers")
async def add_speaker(name: str = Form(...), file: UploadFile = File(...)):
    if _tts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="File must be a .wav file")

    if name in _tts.speakers:
        raise HTTPException(status_code=409, detail=f"Speaker '{name}' already exists")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        emb = await asyncio.to_thread(_tts._extract_speaker_emb, tmp.name)

    _tts.speakers[name] = emb
    log.info("Added speaker '%s', norm=%.3f", name, emb.norm())
    return {"speaker": name, "speakers": list(_tts.speakers.keys())}


@app.get("/v1/models")
async def models():
    return {"models": list(MODELS.keys()), "loaded": _loaded_model}


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
        "--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
        help=f"Model to load (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    os.environ["PLAPRE_MODEL"] = args.model
    uvicorn.run("plapre.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
