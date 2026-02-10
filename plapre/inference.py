"""
Plapre – Danish TTS inference using llama.cpp for fast generation.

Usage:
    from plapre import Plapre

    tts = Plapre("syvai/plapre-turbo")
    tts.speak("Hej, hvordan har du det?", output="output.wav")

    # Voice cloning
    tts.speak("Hej", output="cloned.wav", speaker_wav="reference.wav")

    # Long text with sentence splitting
    tts.speak("Sætning et. Sætning to.", output="long.wav", split_sentences=True)
"""

import ctypes
import json
import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from kanade_tokenizer import KanadeModel, load_vocoder, vocode
from phonemizer import phonemize
from transformers import AutoTokenizer

import llama_cpp
from llama_cpp import (
    llama_batch_free,
    llama_batch_init,
    llama_decode,
    llama_kv_self_clear,
    llama_sampler_chain_add,
    llama_sampler_chain_init,
    llama_sampler_chain_default_params,
    llama_sampler_free,
    llama_sampler_init_dist,
    llama_sampler_init_temp,
    llama_sampler_init_top_k,
    llama_sampler_init_top_p,
    llama_sampler_sample,
)

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SPEAKER_DIM = 128
HIDDEN_SIZE = 960
KANADE_MODEL = "frothywater/kanade-25hz-clean"

GGUF_VARIANTS = {
    "f16": "gguf/plapre-turbo.f16.gguf",
    "q8_0": "gguf/plapre-turbo.q8_0.gguf",
    "q6_k": "gguf/plapre-turbo.q6_k.gguf",
    "q4_k_m": "gguf/plapre-turbo.q4_k_m.gguf",
    "q4_0": "gguf/plapre-turbo.q4_0.gguf",
}
DEFAULT_QUANT = "q8_0"


class Plapre:
    """Danish text-to-speech synthesis."""

    def __init__(
        self,
        checkpoint: str = "syvai/plapre-turbo",
        quant: str = DEFAULT_QUANT,
        model_path: str | None = None,
        n_gpu_layers: int = 99,
        flash_attn: bool = True,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._checkpoint = checkpoint

        # --- Tokenizer ---
        log.info("Loading tokenizer from %s …", checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.audio_token_start = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        self.audio_token_end = self.tokenizer.convert_tokens_to_ids("<audio_12799>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("</audio>")
        self.eos_id = self.tokenizer.eos_token_id

        # --- Speaker projection ---
        self.speaker_proj = self._load_speaker_proj(checkpoint)
        self.speakers = self._load_speakers()
        self.default_speaker = next(iter(self.speakers))
        log.info(
            "Loaded %d speaker(s): %s (default: %s)",
            len(self.speakers), list(self.speakers.keys()), self.default_speaker,
        )

        # --- llama.cpp model ---
        gguf_path = model_path or self._resolve_gguf(checkpoint, quant)
        log.info("Loading GGUF model from %s …", gguf_path)

        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        mparams.use_mmap = True

        self._model = llama_cpp.llama_model_load_from_file(
            gguf_path.encode("utf-8"), mparams,
        )
        if not self._model:
            raise RuntimeError(f"Failed to load GGUF model: {gguf_path}")

        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = 8192
        cparams.n_batch = 2048
        cparams.n_ubatch = 512
        cparams.n_threads = 1
        cparams.n_threads_batch = 1
        cparams.flash_attn = flash_attn
        cparams.type_k = 0  # f16
        cparams.type_v = 0
        cparams.n_seq_max = 16

        self._ctx = llama_cpp.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            raise RuntimeError("Failed to create llama context")

        # --- Kanade vocoder ---
        log.info("Loading Kanade vocoder …")
        self.kanade = KanadeModel.from_pretrained(KANADE_MODEL).eval().to(self.device)
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)

        # Cache for projected speaker embeddings
        self._proj_cache: dict[bytes, np.ndarray] = {}

        log.info("Ready – device=%s", self.device)

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            llama_cpp.llama_free(self._ctx)
        if hasattr(self, "_model") and self._model:
            llama_cpp.llama_model_free(self._model)

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

        gen_kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens)

        if split_sentences:
            sentences = self._split_sentences(text)
            log.info("Batched generation for %d sentences", len(sentences))
            audio = self._generate_audio_batch(sentences, spk, silence_duration, **gen_kwargs)
            if audio is None:
                log.error("No audio generated for any sentence.")
                return np.array([], dtype=np.float32)
        else:
            audio = self._generate_audio(text, spk, **gen_kwargs)
            if audio is None:
                log.error("No audio tokens generated. Try different temperature/top_p.")
                return np.array([], dtype=np.float32)

        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

    def extract_speaker(self, wav_path: str) -> torch.Tensor:
        """Extract a 128-dim speaker embedding from a wav file.

        The returned tensor can be passed to ``speak(speaker_emb=...)``
        for consistent voice across multiple generations.
        """
        return self._extract_speaker_emb(wav_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gguf(checkpoint: str, quant: str) -> str:
        local = Path(checkpoint) / GGUF_VARIANTS.get(quant, GGUF_VARIANTS[DEFAULT_QUANT])
        if local.exists():
            return str(local)
        repo_path = GGUF_VARIANTS.get(quant, GGUF_VARIANTS[DEFAULT_QUANT])
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

    @torch.no_grad()
    def _project_speaker(self, speaker_emb: torch.Tensor) -> np.ndarray:
        key = speaker_emb.cpu().float().numpy().tobytes()
        if key in self._proj_cache:
            return self._proj_cache[key]
        hidden = self.speaker_proj(speaker_emb.cpu().float().to(torch.bfloat16))
        result = hidden.float().numpy().copy()
        self._proj_cache[key] = result
        return result

    def _phonemize(self, text: str) -> str:
        result = phonemize(
            [text],
            language="da",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            language_switch="remove-flags",
        )
        return result[0].strip()

    def _build_prompt(self, text: str, phonemes: str) -> list[int]:
        tok = self.tokenizer
        text_ids = tok.encode(text, add_special_tokens=False)

        phone_ids = []
        for c in phonemes:
            tid = tok.convert_tokens_to_ids(f"<phone_{c}>")
            if tid != tok.unk_token_id:
                phone_ids.append(tid)

        text_start = tok.convert_tokens_to_ids("<text>")
        text_end = tok.convert_tokens_to_ids("</text>")
        ph_start = tok.convert_tokens_to_ids("<phonemes>")
        ph_end = tok.convert_tokens_to_ids("</phonemes>")
        audio_start = tok.convert_tokens_to_ids("<audio>")

        return (
            [text_start] + text_ids + [text_end]
            + [ph_start] + phone_ids + [ph_end, audio_start]
        )

    def _create_sampler(self, temperature: float, top_p: float, top_k: int):
        sparams = llama_sampler_chain_default_params()
        smpl = llama_sampler_chain_init(sparams)
        if top_k > 0:
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k))
        if top_p < 1.0:
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1))
        if temperature > 0:
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42))
        return smpl

    def _generate_tokens(
        self,
        prompt_tokens: list[int],
        speaker_hidden: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[int]:
        """Generate audio token IDs using llama.cpp (single sequence)."""
        llama_kv_self_clear(self._ctx)

        n_prompt = len(prompt_tokens)

        # Decode speaker embedding at position 0
        embd_batch = llama_batch_init(1, HIDDEN_SIZE, 1)
        embd_batch.n_tokens = 1
        ctypes.memmove(embd_batch.embd, speaker_hidden.ctypes.data, HIDDEN_SIZE * 4)
        embd_batch.pos[0] = 0
        embd_batch.n_seq_id[0] = 1
        embd_batch.seq_id[0][0] = 0
        embd_batch.logits[0] = 0
        rc = llama_decode(self._ctx, embd_batch)
        llama_batch_free(embd_batch)
        if rc != 0:
            raise RuntimeError(f"Speaker embedding decode failed: {rc}")

        # Decode prompt tokens at positions 1..N
        batch = llama_batch_init(max(n_prompt, 512), 0, 1)
        batch.n_tokens = n_prompt
        for i, tid in enumerate(prompt_tokens):
            batch.token[i] = tid
            batch.pos[i] = i + 1
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if i == n_prompt - 1 else 0

        rc = llama_decode(self._ctx, batch)
        if rc != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"Prompt decode failed: {rc}")

        smpl = self._create_sampler(temperature, top_p, top_k)

        generated = []
        pos = 1 + n_prompt
        for _ in range(max_tokens):
            new_token = llama_sampler_sample(smpl, self._ctx, -1)
            if new_token == self.audio_end_id or new_token == self.eos_id:
                break
            generated.append(new_token)

            batch.n_tokens = 1
            batch.token[0] = new_token
            batch.pos[0] = pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1

            rc = llama_decode(self._ctx, batch)
            if rc != 0:
                break
            pos += 1

        llama_sampler_free(smpl)
        llama_batch_free(batch)
        return generated

    def _generate_tokens_batch(
        self,
        prompt_tokens_list: list[list[int]],
        speaker_hidden: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> list[list[int]]:
        """Generate audio tokens for multiple prompts in parallel."""
        batch_size = len(prompt_tokens_list)
        if batch_size == 1:
            return [self._generate_tokens(
                prompt_tokens_list[0], speaker_hidden, temperature, top_p, top_k, max_tokens,
            )]

        llama_kv_self_clear(self._ctx)
        total_prompt_tokens = sum(len(p) for p in prompt_tokens_list)

        # Decode speaker embeddings for all sequences at position 0
        embd_batch = llama_batch_init(batch_size, HIDDEN_SIZE, 1)
        embd_batch.n_tokens = batch_size
        for seq_id in range(batch_size):
            dst_ptr = ctypes.addressof(embd_batch.embd.contents) + seq_id * HIDDEN_SIZE * 4
            ctypes.memmove(dst_ptr, speaker_hidden.ctypes.data, HIDDEN_SIZE * 4)
            embd_batch.pos[seq_id] = 0
            embd_batch.n_seq_id[seq_id] = 1
            embd_batch.seq_id[seq_id][0] = seq_id
            embd_batch.logits[seq_id] = 0

        rc = llama_decode(self._ctx, embd_batch)
        llama_batch_free(embd_batch)
        if rc != 0:
            raise RuntimeError(f"Batch speaker embedding decode failed: {rc}")

        # Decode all prompt tokens at positions 1..N per sequence
        batch = llama_batch_init(max(total_prompt_tokens + 64, batch_size * 2), 0, 1)
        idx = 0
        for seq_id, prompt in enumerate(prompt_tokens_list):
            for pos, tid in enumerate(prompt):
                batch.token[idx] = tid
                batch.pos[idx] = pos + 1
                batch.n_seq_id[idx] = 1
                batch.seq_id[idx][0] = seq_id
                batch.logits[idx] = 1 if pos == len(prompt) - 1 else 0
                idx += 1
        batch.n_tokens = idx

        rc = llama_decode(self._ctx, batch)
        if rc != 0:
            llama_batch_free(batch)
            raise RuntimeError(f"Batch prompt decode failed: {rc}")

        # Per-sequence samplers
        samplers = [self._create_sampler(temperature, top_p, top_k) for _ in range(batch_size)]

        generated = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        positions = [1 + len(p) for p in prompt_tokens_list]

        # Sample first token for each sequence
        logits_indices = []
        offset = 0
        for seq_id, prompt in enumerate(prompt_tokens_list):
            logits_indices.append(offset + len(prompt) - 1)
            offset += len(prompt)

        for seq_id in range(batch_size):
            new_token = llama_sampler_sample(samplers[seq_id], self._ctx, logits_indices[seq_id])
            if new_token == self.audio_end_id or new_token == self.eos_id:
                finished[seq_id] = True
            else:
                generated[seq_id].append(new_token)

        # Autoregressive loop
        for _ in range(max_tokens - 1):
            if all(finished):
                break

            idx = 0
            active = []
            for seq_id in range(batch_size):
                if finished[seq_id]:
                    continue
                batch.token[idx] = generated[seq_id][-1]
                batch.pos[idx] = positions[seq_id]
                batch.n_seq_id[idx] = 1
                batch.seq_id[idx][0] = seq_id
                batch.logits[idx] = 1
                active.append((seq_id, idx))
                idx += 1
                positions[seq_id] += 1

            if idx == 0:
                break
            batch.n_tokens = idx

            rc = llama_decode(self._ctx, batch)
            if rc != 0:
                break

            for seq_id, batch_idx in active:
                new_token = llama_sampler_sample(samplers[seq_id], self._ctx, batch_idx)
                if new_token == self.audio_end_id or new_token == self.eos_id:
                    finished[seq_id] = True
                else:
                    generated[seq_id].append(new_token)

        for smpl in samplers:
            llama_sampler_free(smpl)
        llama_batch_free(batch)
        return generated

    def _generate_audio(
        self,
        text: str,
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> np.ndarray | None:
        phonemes = self._phonemize(text)
        prompt_ids = self._build_prompt(text, phonemes)
        speaker_hidden = self._project_speaker(speaker_emb)

        generated = self._generate_tokens(
            prompt_ids, speaker_hidden, temperature, top_p, top_k, max_tokens,
        )

        kanade_indices = [
            tid - self.audio_token_start
            for tid in generated
            if self.audio_token_start <= tid <= self.audio_token_end
        ]

        if not kanade_indices:
            return None

        tokens_tensor = torch.tensor(kanade_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            mel = self.kanade.decode(
                content_token_indices=tokens_tensor,
                global_embedding=speaker_emb.float().to(self.device),
            )
            waveform = vocode(self.vocoder, mel.unsqueeze(0))

        return waveform.squeeze().cpu().numpy()

    def _generate_audio_batch(
        self,
        sentences: list[str],
        speaker_emb: torch.Tensor,
        silence_duration: float,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> np.ndarray | None:
        prompt_list = []
        for sent in sentences:
            phonemes = self._phonemize(sent)
            prompt_list.append(self._build_prompt(sent, phonemes))

        speaker_hidden = self._project_speaker(speaker_emb)
        all_generated = self._generate_tokens_batch(
            prompt_list, speaker_hidden, temperature, top_p, top_k, max_tokens,
        )

        silence = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)
        chunks = []
        for i, generated in enumerate(all_generated):
            kanade_indices = [
                tid - self.audio_token_start
                for tid in generated
                if self.audio_token_start <= tid <= self.audio_token_end
            ]
            if not kanade_indices:
                continue
            tokens_tensor = torch.tensor(kanade_indices, dtype=torch.long, device=self.device)
            with torch.no_grad():
                mel = self.kanade.decode(
                    content_token_indices=tokens_tensor,
                    global_embedding=speaker_emb.float().to(self.device),
                )
                waveform = vocode(self.vocoder, mel.unsqueeze(0))
            chunks.append(waveform.squeeze().cpu().numpy())
            if i < len(all_generated) - 1:
                chunks.append(silence)

        if not chunks:
            return None
        return np.concatenate(chunks)

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
        self, speaker: str | None, speaker_wav: str | None, speaker_emb: torch.Tensor | None
    ) -> torch.Tensor:
        if speaker_emb is not None:
            return speaker_emb.to(self.device)
        if speaker_wav is not None:
            emb = self._extract_speaker_emb(speaker_wav)
            log.info("Speaker embedding from %s, norm=%.3f", speaker_wav, emb.norm())
            return emb
        name = speaker or self.default_speaker
        if name not in self.speakers:
            raise ValueError(f"Unknown speaker '{name}'. Available: {list(self.speakers.keys())}")
        return self.speakers[name]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]
