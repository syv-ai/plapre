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

from plapre import tasks
from plapre.tasks import TaskTokens

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


def _patch_vllm_speculators_gguf():
    """Let vLLM load GGUF checkpoints under recent versions.

    vLLM's ``maybe_override_with_speculators`` calls ``get_config_dict(model)``,
    which tries to parse a ``.gguf`` path as a JSON config and raises OSError
    ("is not a valid JSON file"). A GGUF is never a speculators model, so skip
    that probe for ``.gguf`` model paths. No-op on vLLM versions without it.
    """
    try:
        import vllm.transformers_utils.config as _cfg
    except Exception:
        return
    orig = getattr(_cfg, "maybe_override_with_speculators", None)
    if orig is None or getattr(orig, "_plapre_gguf_patched", False):
        return

    def patched(model, tokenizer, trust_remote_code, revision=None,
                vllm_speculative_config=None, hf_token=None, **kwargs):
        if str(model).endswith(".gguf"):
            return model, tokenizer, vllm_speculative_config
        return orig(model, tokenizer, trust_remote_code, revision=revision,
                    vllm_speculative_config=vllm_speculative_config,
                    hf_token=hf_token, **kwargs)

    patched._plapre_gguf_patched = True
    # vLLM binds this symbol via `from … import …`, so rebind on every module
    # that imported the original (plus its home module).
    import sys as _sys

    _cfg.maybe_override_with_speculators = patched
    for _m in list(_sys.modules.values()):
        try:
            if getattr(_m, "maybe_override_with_speculators", None) is orig:
                _m.maybe_override_with_speculators = patched
        except Exception:
            pass


class Plapre:
    """Danish text-to-speech synthesis."""

    def __init__(
        self,
        checkpoint: str = "syvai/plapre-nano-v2",
        quant: str | None = None,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 1024,
        device: str | None = None,
        use_async: bool = False,
        enforce_eager: bool = False,
        dtype: str = "float32",
    ):
        """Load a plapre checkpoint (default: ``syvai/plapre-nano-v2`` safetensors).

        ``quant``: ``None`` (default) loads the HF *safetensors* weights directly — v2
        ships no GGUF. Pass e.g. ``"q8_0"`` for the v1 ``syvai/plapre-nano`` GGUF builds.
        ``dtype``: ``"float32"`` by default — plapre trains fp32 master weights, and lower
        serving precision measurably increases end-of-utterance looping. For the multi-task
        modes with long reference audio raise ``max_model_len`` (~1536)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._checkpoint = checkpoint
        self._use_async = use_async

        _patch_tokenizer_compat()
        _patch_vllm_speculators_gguf()

        # --- Tokenizer (CPU) ---
        log.info("Loading tokenizer …")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        except (ValueError, KeyError, ImportError, OSError):
            # Some transformers builds can't resolve a custom tokenizer_class
            # (e.g. `TokenizersBackend`); load the fast tokenizer directly.
            from transformers import PreTrainedTokenizerFast

            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
        self.audio_token_start = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        self.audio_token_end = self.tokenizer.convert_tokens_to_ids("<audio_12799>")
        self.text_tag = self.tokenizer.convert_tokens_to_ids("<text>")
        self.audio_tag = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.eos_id = self.tokenizer.eos_token_id

        # control-token layout for the multi-task modes (clone/context/pace/edit/pronounce);
        # None-valued on a base checkpoint that lacks these tokens -> mode methods will refuse.
        self.tasks = TaskTokens.from_tokenizer(self.tokenizer)
        self.supports_modes = self.tasks.supports_modes
        # Multi-task checkpoints end generated audio with a trained `</audio>` before
        # `<eos>`; stop on both. On vocabs without `</audio>` this is just [eos].
        self.gen_stop_ids = self.tasks.stop_ids

        # --- Speaker projection (CPU) ---
        self.speaker_proj = self._load_speaker_proj(checkpoint)

        # --- Embedding layer (CPU) ---
        log.info("Loading embedding layer …")
        self._embed_tokens = self._load_embed_tokens(checkpoint)

        # --- Resolve model: full safetensors checkpoint (default), or a v1 GGUF quant ---
        # v2 ships safetensors only. GGUF (v1) also needs the speculators patch on
        # recent vLLM, which rejects a bare `.gguf` file path in ModelConfig.
        if quant in (None, "none", "safetensors"):
            gguf_path = checkpoint
            log.info("Using full (safetensors) model: %s", checkpoint)
        else:
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
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
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
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
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
        silence_duration: float = 0.1,
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
    # Multi-task modes (require a checkpoint with the control tokens,
    # e.g. syvai/plapre-nano-v2 loaded with quant=None)
    # ------------------------------------------------------------------

    def clone(
        self,
        text: str,
        output: str = "clone.wav",
        *,
        reference_wav: str | None = None,
        reference_text: str = "",
        reference_emb: torch.Tensor | None = None,
        reference_tokens: list[int] | None = None,
        durations: list[int] | None = None,
        pronunciations: list | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 500,
    ) -> np.ndarray:
        """Speak *text* in a reference voice. Give ``reference_wav`` (a clip of the target
        voice) or ``reference_tokens`` + ``reference_emb``; ``reference_text`` is what that
        clip says (helps, optional). Optionally layer ``durations`` (pace) and
        ``pronunciations``."""
        self._require_modes("clone")
        ref_tokens, ref_emb = self._resolve_reference(
            reference_wav, reference_emb, reference_tokens
        )
        prompt = tasks.clone_prompt(
            self.tasks, self._encode_target(text), self._encode(reference_text),
            ref_tokens, dur_frames=durations, lex=self._pron_prefix(pronunciations),
        )
        return self._synthesize(prompt, ref_emb, output, temperature, top_p, top_k, max_tokens)

    def continue_context(
        self,
        text: str,
        output: str = "context.wav",
        *,
        prev_text: str,
        prev_wav: str | None = None,
        prev_tokens: list[int] | None = None,
        same_speaker: bool = True,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        durations: list[int] | None = None,
        pronunciations: list | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 500,
    ) -> np.ndarray:
        """Synthesize *text* conditioned on the previous utterance for prosodic continuity.
        ``prev_text`` is required; add ``prev_wav``/``prev_tokens`` for the previous audio.
        Keeps the target voice (``speaker``/``speaker_wav``/``speaker_emb``)."""
        self._require_modes("continue_context")
        spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)
        prev_audio = None
        if prev_wav is not None:
            prev_audio, _ = self._encode_audio(prev_wav)
        elif prev_tokens is not None:
            prev_audio = list(prev_tokens)
        prompt = tasks.context_prompt(
            self.tasks, self._encode_target(text), self._encode(prev_text), same_speaker,
            prev_audio_tokens=prev_audio, dur_frames=durations,
            lex=self._pron_prefix(pronunciations),
        )
        return self._synthesize(prompt, spk, output, temperature, top_p, top_k, max_tokens)

    def edit(
        self,
        new_text: str,
        mask_start: int,
        mask_end: int,
        output: str = "edit.wav",
        *,
        original_wav: str | None = None,
        original_tokens: list[int] | None = None,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        pronunciations: list | None = None,
        min_tokens: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 200,
    ) -> np.ndarray:
        """Edit an existing clip: regenerate the audio for words ``[mask_start:mask_end)``
        (Kanade-frame indices, 25 Hz) while giving the model the *full desired* ``new_text``.
        The fill is spliced back into the original audio. Supply the original clip as
        ``original_wav`` or ``original_tokens``. For insert/delete, set ``min_tokens`` above/
        below the masked span so the fill length follows the new text, not the gap."""
        self._require_modes("edit")
        emb = None
        if original_tokens is not None:
            original_tokens = list(original_tokens)
        elif original_wav is not None:
            original_tokens, emb = self._encode_audio(original_wav)
        else:
            raise ValueError("edit() needs original_wav or original_tokens")
        if speaker is not None or speaker_wav is not None or speaker_emb is not None:
            spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)
        elif emb is not None:
            spk = emb  # keep the edited clip's own voice
        else:
            spk = self.speakers[self.default_speaker]
        prompt = tasks.edit_prompt(
            self.tasks, self._encode(new_text), original_tokens, mask_start, mask_end,
            lex=self._pron_prefix(pronunciations),
        )
        fill_ids = self._generate_ids(
            [prompt], spk, temperature, top_p, top_k, max_tokens,
            stop_token_ids=[self.tasks.edit_end, self.eos_id], min_tokens=min_tokens,
        )[0]
        fill = [
            t - self.audio_token_start
            for t in fill_ids
            if self.audio_token_start <= t <= self.audio_token_end
        ]
        spliced = original_tokens[:mask_start] + fill + original_tokens[mask_end:]
        audio = self._decode_tokens(spliced, spk)
        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

    def speak_paced(
        self,
        text: str,
        durations: list[int],
        output: str = "paced.wav",
        *,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        pronunciations: list | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 700,
    ) -> np.ndarray:
        """Base TTS with explicit per-word length control. ``durations`` = one frame count
        per word (40 ms each); ``len(durations)`` must equal the number of words in *text*."""
        self._require_modes("speak_paced")
        spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)
        prompt = tasks.base_prompt(
            self.tasks, self._encode_target(text), dur_frames=durations,
            lex=self._pron_prefix(pronunciations),
        )
        return self._synthesize(prompt, spk, output, temperature, top_p, top_k, max_tokens)

    def pronounce(
        self,
        text: str,
        pronunciations: list,
        output: str = "pronounced.wav",
        *,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        speaker_emb: torch.Tensor | None = None,
        durations: list[int] | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 500,
    ) -> np.ndarray:
        """Base TTS with pronunciation references pinned. ``pronunciations`` = a list of
        ``(word, reference)`` pairs, where ``reference`` is a wav path or a list of Kanade
        audio tokens rendering that word (ideally from a clean clip that says it)."""
        self._require_modes("pronounce")
        spk = self._resolve_speaker(speaker, speaker_wav, speaker_emb)
        prompt = tasks.base_prompt(
            self.tasks, self._encode_target(text), dur_frames=durations,
            lex=self._pron_prefix(pronunciations),
        )
        return self._synthesize(prompt, spk, output, temperature, top_p, top_k, max_tokens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_modes(self, name: str) -> None:
        if not self.supports_modes:
            raise RuntimeError(
                f"{name}() needs a multi-task checkpoint with the control tokens "
                f"(e.g. Plapre('syvai/plapre-nano-v2', quant=None)); "
                f"'{self._checkpoint}' lacks them."
            )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(self._normalize_text(text), add_special_tokens=False)

    _SENT_END = (".", "?", "!", ":", "…")

    @classmethod
    def _ensure_terminal_punct(cls, text: str) -> str:
        """Append a period when the target text ends mid-sentence (e.g. on a comma).
        The model's stop signal is strongest on sentence-final text — measured loop
        rate on comma-ending fragments drops ~7x with this normalization. Never
        changes the word count (punctuation attaches to the final word)."""
        t = text.rstrip()
        if not t:
            return t
        if t.endswith(","):
            t = t[:-1]
        return t if t.endswith(cls._SENT_END) else t + "."

    def _encode_target(self, text: str) -> list[int]:
        """Encode a TARGET text (the words being synthesized): normalization plus
        terminal punctuation. Reference/context transcripts keep `_encode` verbatim."""
        return self.tokenizer.encode(
            self._ensure_terminal_punct(self._normalize_text(text)), add_special_tokens=False
        )

    def _pron_prefix(self, pronunciations) -> list[int] | None:
        """Build the <lex> prefix from [(word, wav_path | token_list), …], or None."""
        if not pronunciations:
            return None
        pairs = []
        for word, ref in pronunciations:
            ref_tokens = self._encode_audio(ref)[0] if isinstance(ref, str) else list(ref)
            pairs.append((self._encode(word), ref_tokens))
        return tasks.lex_prefix(self.tasks, pairs)

    def _resolve_reference(self, wav, emb, tokens):
        """-> (kanade audio token list, speaker embedding) for the clone reference."""
        if tokens is not None:
            if emb is None:
                raise ValueError("reference_tokens also needs reference_emb")
            return list(tokens), emb.to(self.device)
        if wav is not None:
            toks, e = self._encode_audio(wav)
            return toks, e
        raise ValueError("clone() needs reference_wav or (reference_tokens + reference_emb)")

    @torch.no_grad()
    def _encode_audio(self, wav_path: str):
        """Kanade-encode a wav -> (content token indices as list[int], global_embedding)."""
        import torchaudio

        data, sr = sf.read(wav_path, dtype="float32")
        data = data[np.newaxis, :] if data.ndim == 1 else data.T
        wav = torch.from_numpy(data)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        feats = self.kanade.encode(wav.to(self.device))
        return feats.content_token_indices.cpu().tolist(), feats.global_embedding

    def _generate_ids(
        self,
        prompt_ids_list: list[list[int]],
        speaker_emb: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        stop_token_ids: list[int] | None = None,
        min_tokens: int = 0,
    ) -> list[list[int]]:
        """Generate token ids from ready-made prompt id lists (the mode builders' output)."""
        from vllm import SamplingParams

        speaker_hidden = self._project_speaker(speaker_emb)
        prompts = [self._build_embeds_prompt(ids, speaker_hidden) for ids in prompt_ids_list]
        if stop_token_ids is None:
            stop_token_ids = self.gen_stop_ids
        sampling = SamplingParams(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_tokens=max_tokens, min_tokens=min_tokens, stop_token_ids=stop_token_ids,
        )
        outputs = self._llm.generate(prompts, sampling, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outputs]

    def _decode_tokens(self, kanade_tokens: list[int], speaker_emb: torch.Tensor) -> np.ndarray:
        """Decode raw Kanade content-token indices (0..12799) to a waveform."""
        if not kanade_tokens:
            return np.array([], dtype=np.float32)
        tokens_tensor = torch.tensor(kanade_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            mel = self.kanade.decode(
                content_token_indices=tokens_tensor,
                global_embedding=speaker_emb.float().to(self.device),
            )
            from kanade_tokenizer import vocode

            waveform = vocode(self.vocoder, mel.unsqueeze(0))
        return waveform.squeeze().cpu().numpy()

    def _synthesize(
        self, prompt_ids, speaker_emb, output, temperature, top_p, top_k, max_tokens
    ) -> np.ndarray:
        """Run one mode prompt through generation + Kanade decode and save to *output*."""
        tokens = self._generate_ids(
            [prompt_ids], speaker_emb, temperature, top_p, top_k, max_tokens
        )[0]
        audio = self._tokens_to_audio(tokens, speaker_emb)
        if audio is None:
            log.error("No audio tokens generated. Try different temperature/top_p.")
            return np.array([], dtype=np.float32)
        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

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
        text_ids = self.tokenizer.encode(self._ensure_terminal_punct(text), add_special_tokens=False)
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
            stop_token_ids=self.gen_stop_ids,
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
            stop_token_ids=self.gen_stop_ids,
        )

        tasks = [self._generate_one_async(p, sampling) for p in prompts]
        return await asyncio.gather(*tasks)

    def _extract_speaker_emb(self, wav_path: str) -> torch.Tensor:
        return self._encode_audio(wav_path)[1]

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
        # Split on sentence-ending punctuation followed by space,
        # but require 2+ word chars before the punctuation to avoid
        # splitting on abbreviations like "H.C." or "f."
        parts = re.split(r"(?<=\w{2}[.!?])\s+", text)
        result = []
        for p in parts:
            p = p.strip()
            # Strip leading dialogue dashes (Danish convention: "- quote")
            p = re.sub(r"^[-–—]\s+", "", p)
            if p:
                result.append(p)
        return result
