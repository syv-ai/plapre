"""
Plapre – Danish TTS inference using vLLM for fast generation.

Usage:
    from plapre import Plapre

    tts = Plapre("syvai/plapre-nano")
    tts.speak("Hej, hvordan har du det?", output="output.wav")

    # Voice cloning
    tts.speak("Hej", output="cloned.wav", speaker_wav="reference.wav")

    # Long text with sentence splitting
    tts.speak("Sætning et. Sætning to.", output="long.wav", split_sentences=True)
"""

import json
import logging
import multiprocessing
import os
import re
from pathlib import Path

# vLLM V1 EngineCore uses multiprocessing; must use spawn to avoid CUDA fork issues
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SPEAKER_DIM = 128
HIDDEN_SIZE = 960
KANADE_MODEL = "frothywater/kanade-25hz-clean"

GGUF_QUANTS = ["f16", "q8_0", "q6_k", "q4_k_m", "q4_0"]
DEFAULT_QUANT = "q8_0"


def _patch_tokenizer_compat():
    """Patch transformers tokenizer for vLLM compatibility (tokenizers 5.x → 4.x)."""
    import transformers.tokenization_utils_base as tub

    _original = tub.PreTrainedTokenizerBase.__init__
    if getattr(_original, "_plapre_patched", False):
        return

    def _patched(self, *args, **kwargs):
        if "extra_special_tokens" in kwargs and isinstance(
            kwargs["extra_special_tokens"], list
        ):
            kwargs["extra_special_tokens"] = {}
        _original(self, *args, **kwargs)

    _patched._plapre_patched = True
    tub.PreTrainedTokenizerBase.__init__ = _patched


class Plapre:
    """Danish text-to-speech synthesis."""

    def __init__(
        self,
        checkpoint: str = "syvai/plapre-nano",
        quant: str = DEFAULT_QUANT,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 512,
        device: str | None = None,
        use_async: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._checkpoint = checkpoint
        self._use_async = use_async

        _patch_tokenizer_compat()

        # --- Tokenizer (CPU) ---
        log.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.audio_token_start = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        self.audio_token_end = self.tokenizer.convert_tokens_to_ids("<audio_12799>")
        self.text_tag = self.tokenizer.convert_tokens_to_ids("<text>")
        self.audio_tag = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.eos_id = self.tokenizer.eos_token_id

        # --- Speaker projection (CPU) ---
        self.speaker_proj = self._load_speaker_proj(checkpoint)

        # --- Embedding layer (CPU) ---
        log.info("Loading embedding layer …")
        self._embed_tokens = self._load_embed_tokens(checkpoint)

        # --- Resolve GGUF model ---
        gguf_path = self._resolve_gguf(checkpoint, quant)
        log.info("Using GGUF model: %s", gguf_path)

        # --- vLLM engine ---
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        if use_async:
            log.info("Initializing async vLLM engine (AsyncLLM) …")
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(
                model=gguf_path,
                tokenizer=checkpoint,
                dtype="auto",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=False,
                enable_prompt_embeds=True,
            )
            self._async_llm = AsyncLLM.from_engine_args(engine_args)
            self._llm = None
            self._request_counter = 0
        else:
            log.info("Initializing vLLM engine …")
            from vllm import LLM

            self._llm = LLM(
                model=gguf_path,
                tokenizer=checkpoint,
                dtype="auto",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=False,
                enable_prompt_embeds=True,
            )
            self._async_llm = None

        # --- Speakers (GPU) ---
        self.speakers = self._load_speakers()
        self.default_speaker = next(iter(self.speakers))
        log.info(
            "Loaded %d speaker(s): %s (default: %s)",
            len(self.speakers),
            list(self.speakers.keys()),
            self.default_speaker,
        )

        # --- Kanade vocoder (GPU) ---
        log.info("Loading Kanade vocoder …")
        from kanade_tokenizer import KanadeModel, load_vocoder

        self.kanade = KanadeModel.from_pretrained(KANADE_MODEL).eval().to(self.device)
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)

        # Cache for projected speaker embeddings
        self._proj_cache: dict[bytes, torch.Tensor] = {}

        log.info("Ready – device=%s, async=%s", self.device, use_async)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        text: str,
        output: str = "output.wav",
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 500,
        split_sentences: bool = False,
        silence_duration: float = 0.3,
    ) -> np.ndarray:
        """Synthesize speech and save to *output*. Returns the audio as a numpy array."""
        spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)

        gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        if split_sentences:
            sentences = self._split_sentences(text)  # normalize included
            log.info("Batch generating %d sentences", len(sentences))
            results = self._generate_audio_batch(
                sentences, spk, **gen_kwargs
            )
            silence = np.zeros(
                int(silence_duration * SAMPLE_RATE), dtype=np.float32
            )
            chunks = []
            for i, audio_chunk in enumerate(results):
                if audio_chunk is not None:
                    chunks.append(audio_chunk)
                    if i < len(sentences) - 1:
                        chunks.append(silence)
            if not chunks:
                log.error("No audio generated for any sentence.")
                return np.array([], dtype=np.float32)
            audio = np.concatenate(chunks)
        else:
            audio = self._generate_audio(self._normalize_text(text), spk, **gen_kwargs)
            if audio is None:
                log.error(
                    "No audio tokens generated. Try different temperature/top_p."
                )
                return np.array([], dtype=np.float32)

        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gguf(checkpoint: str, quant: str) -> str:
        if quant not in GGUF_QUANTS:
            quant = DEFAULT_QUANT
        model_name = checkpoint.rstrip("/").split("/")[-1]
        repo_path = f"gguf/{model_name}.{quant}.gguf"
        local = Path(checkpoint) / repo_path
        if local.exists():
            return str(local)
        return hf_hub_download(checkpoint, repo_path)

    def _load_speakers(self) -> dict[str, torch.Tensor]:
        path = Path(__file__).parent / "speakers.json"
        with open(path) as f:
            raw = json.load(f)
        return {
            name: torch.tensor(emb, dtype=torch.float32, device=self.device)
            for name, emb in raw.items()
        }

    def _load_speaker_proj(self, checkpoint: str) -> nn.Linear:
        proj = nn.Linear(SPEAKER_DIM, HIDDEN_SIZE)
        local = Path(checkpoint) / "speaker_proj.pt"
        if local.exists():
            proj.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            path = hf_hub_download(checkpoint, "speaker_proj.pt")
            proj.load_state_dict(torch.load(path, map_location="cpu"))
        return proj.to(torch.bfloat16).eval()

    def _load_embed_tokens(self, checkpoint: str) -> nn.Embedding:
        """Load just the embedding layer from the model weights."""
        from safetensors.torch import load_file

        local = Path(checkpoint) / "model.safetensors"
        if local.exists():
            st_path = str(local)
        else:
            st_path = hf_hub_download(checkpoint, "model.safetensors")

        state = load_file(st_path)
        weight = state["model.embed_tokens.weight"]
        embed = nn.Embedding(weight.shape[0], weight.shape[1])
        embed.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)
        return embed.eval()

    @torch.no_grad()
    def _project_speaker(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        """Project 128-dim speaker embedding to 960-dim hidden, returns bf16 tensor."""
        key = speaker_emb.cpu().float().numpy().tobytes()
        if key in self._proj_cache:
            return self._proj_cache[key]
        hidden = self.speaker_proj(speaker_emb.cpu().float().to(torch.bfloat16))
        self._proj_cache[key] = hidden
        return hidden

    def _build_prompt(self, text: str) -> list[int]:
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.text_tag] + text_ids + [self.audio_tag]

    @torch.no_grad()
    def _build_embeds_prompt(
        self, prompt_ids: list[int], speaker_hidden: torch.Tensor
    ) -> dict:
        """Build a vLLM EmbedsPrompt with speaker embedding prepended."""
        token_ids = torch.tensor(prompt_ids, dtype=torch.long)
        token_embeds = self._embed_tokens(token_ids)  # (n_tokens, 960)
        full_embeds = torch.cat(
            [speaker_hidden.unsqueeze(0), token_embeds], dim=0
        )  # (n_tokens+1, 960)
        return {"prompt_embeds": full_embeds}

    def _generate_tokens(
        self,
        texts: list[str],
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[list[int]]:
        """Generate audio token IDs for one or more texts using vLLM."""
        from vllm import SamplingParams

        speaker_hidden = self._project_speaker(speaker_emb)

        prompts = []
        for text in texts:
            prompt_ids = self._build_prompt(text)
            prompts.append(self._build_embeds_prompt(prompt_ids, speaker_hidden))

        sampling = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        outputs = self._llm.generate(prompts, sampling, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def _tokens_to_audio(
        self, tokens: list[int], speaker_emb: torch.Tensor
    ) -> np.ndarray | None:
        """Convert generated token IDs to audio waveform via Kanade + Vocos."""
        kanade_indices = [
            tid - self.audio_token_start
            for tid in tokens
            if self.audio_token_start <= tid <= self.audio_token_end
        ]
        if not kanade_indices:
            return None

        tokens_tensor = torch.tensor(
            kanade_indices, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            mel = self.kanade.decode(
                content_token_indices=tokens_tensor,
                global_embedding=speaker_emb.float().to(self.device),
            )
            from kanade_tokenizer import vocode

            waveform = vocode(self.vocoder, mel.unsqueeze(0))
        return waveform.squeeze().cpu().numpy()

    def _generate_audio(
        self,
        text: str,
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> np.ndarray | None:
        all_tokens = self._generate_tokens(
            [text], speaker_emb, temperature, top_p, top_k, max_tokens
        )
        return self._tokens_to_audio(all_tokens[0], speaker_emb)

    def _generate_audio_batch(
        self,
        texts: list[str],
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[np.ndarray | None]:
        """Generate audio for multiple texts in parallel."""
        all_tokens = self._generate_tokens(
            texts, speaker_emb, temperature, top_p, top_k, max_tokens
        )
        return [self._tokens_to_audio(t, speaker_emb) for t in all_tokens]

    # ------------------------------------------------------------------
    # Async generation (for use with AsyncLLM)
    # ------------------------------------------------------------------

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"plapre-{self._request_counter}"

    async def _generate_one_async(
        self,
        prompt: dict,
        sampling,
    ) -> list[int]:
        """Run a single async generate() call, return token IDs."""
        request_id = self._next_request_id()
        final_output = None
        async for out in self._async_llm.generate(
            prompt, sampling, request_id=request_id
        ):
            final_output = out
        return list(final_output.outputs[0].token_ids)

    async def generate_tokens_async(
        self,
        texts: list[str],
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[list[int]]:
        """Generate audio token IDs for multiple texts concurrently via AsyncLLM."""
        import asyncio

        from vllm import SamplingParams

        speaker_hidden = self._project_speaker(speaker_emb)

        prompts = []
        for text in texts:
            prompt_ids = self._build_prompt(text)
            prompts.append(self._build_embeds_prompt(prompt_ids, speaker_hidden))

        sampling = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        tasks = [self._generate_one_async(p, sampling) for p in prompts]
        return await asyncio.gather(*tasks)

    def _extract_speaker_emb(self, wav_path: str) -> torch.Tensor:
        import torchaudio

        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T  # (channels, samples)
        wav = torch.from_numpy(data)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        with torch.no_grad():
            features = self.kanade.encode(wav.to(self.device))
        return features.global_embedding

    def _resolve_speaker(
        self,
        speaker: str | None,
        speaker_wav: str | None,
        speaker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if speaker_emb is not None:
            return speaker_emb.to(self.device)
        if speaker_wav is not None:
            emb = self._extract_speaker_emb(speaker_wav)
            log.info(
                "Speaker embedding from %s, norm=%.3f", speaker_wav, emb.norm()
            )
            return emb
        name = speaker or self.default_speaker
        if name not in self.speakers:
            raise ValueError(
                f"Unknown speaker '{name}'. Available: {list(self.speakers.keys())}"
            )
        return self.speakers[name]

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """Replace numbers with Danish words (e.g. '2,1' → 'to komma et')."""
        from num2words import num2words

        def _replace(m):
            raw = m.group()
            try:
                return num2words(float(raw.replace(",", ".")), lang="da")
            except (ValueError, OverflowError):
                return raw

        return re.sub(r"\d+(?:[,\.]\d+)?", _replace, text)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize raw article/document text for TTS."""
        # Remove trailing separators and image captions (e.g. "--- caption text")
        text = re.sub(r"\s*-{2,}.*$", "", text.strip(), flags=re.DOTALL)
        # Collapse whitespace (newlines, tabs, multiple spaces → single space)
        text = re.sub(r"\s+", " ", text)
        # Numbers → Danish words
        text = Plapre._normalize_numbers(text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Normalize first
        text = Plapre._normalize_text(text)
        # Split on sentence-ending punctuation followed by space
        parts = re.split(r"(?<=[.!?])\s+", text)
        result = []
        for p in parts:
            p = p.strip()
            # Strip leading dialogue dashes (Danish convention: "- quote")
            p = re.sub(r"^[-–—]\s+", "", p)
            if p:
                result.append(p)
        return result
