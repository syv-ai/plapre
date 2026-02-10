"""
Plapre – Danish TTS inference.

Usage:
    from plapre import Plapre

    tts = Plapre("syvai/plapre-turbo")
    tts.speak("Hej, hvordan har du det?", output="output.wav")

    # Voice cloning
    tts.speak("Hej", output="cloned.wav", speaker_wav="reference.wav")

    # Long text with sentence splitting
    tts.speak("Sætning et. Sætning to.", output="long.wav", split_sentences=True)
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SPEAKER_DIM = 128
KANADE_MODEL = "frothywater/kanade-25hz-clean"


class Plapre:
    """Danish text-to-speech synthesis."""

    def __init__(self, checkpoint: str = "syvai/plapre-turbo", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        log.info("Loading model from %s …", checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        self.speaker_proj = self._load_speaker_proj(checkpoint)
        self.speakers = self._load_speakers()
        self.default_speaker = next(iter(self.speakers))
        log.info("Loaded %d speaker(s): %s (default: %s)", len(self.speakers), list(self.speakers.keys()), self.default_speaker)

        self.audio_token_start = self.tokenizer.convert_tokens_to_ids("<audio_0>")
        self.audio_token_end = self.tokenizer.convert_tokens_to_ids("<audio_12799>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("</audio>")
        self.eos_id = self.tokenizer.eos_token_id

        log.info("Loading Kanade vocoder …")
        self.kanade = KanadeModel.from_pretrained(KANADE_MODEL).eval().to(self.device)
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)

        log.info(
            "Ready – %.1fM params, device=%s",
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            self.device,
        )

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
            chunks = []
            silence = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)
            for i, sent in enumerate(sentences):
                log.info("Sentence %d/%d: %s", i + 1, len(sentences), sent)
                audio = self._generate_audio(sent, spk, **gen_kwargs)
                if audio is not None:
                    chunks.append(audio)
                    if i < len(sentences) - 1:
                        chunks.append(silence)
            if not chunks:
                log.error("No audio generated for any sentence.")
                return np.array([], dtype=np.float32)
            audio = np.concatenate(chunks)
        else:
            audio = self._generate_audio(text, spk, **gen_kwargs)
            if audio is None:
                log.error("No audio tokens generated. Try different temperature/top_p.")
                return np.array([], dtype=np.float32)

        sf.write(output, audio, SAMPLE_RATE)
        log.info("Saved %.2fs audio to %s", len(audio) / SAMPLE_RATE, output)
        return audio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_speakers(self) -> dict[str, torch.Tensor]:
        path = Path(__file__).parent / "speakers.json"
        with open(path) as f:
            raw = json.load(f)
        return {
            name: torch.tensor(emb, dtype=torch.float32, device=self.device)
            for name, emb in raw.items()
        }

    def _load_speaker_proj(self, checkpoint: str) -> nn.Linear:
        hidden_size = self.model.config.hidden_size
        proj = nn.Linear(SPEAKER_DIM, hidden_size)
        local = Path(checkpoint) / "speaker_proj.pt"
        if local.exists():
            proj.load_state_dict(torch.load(local, map_location="cpu"))
            log.info("Loaded speaker projection from local path")
        else:
            path = hf_hub_download(checkpoint, "speaker_proj.pt")
            proj.load_state_dict(torch.load(path, map_location="cpu"))
            log.info("Loaded speaker projection from HuggingFace")
        return proj.to(torch.bfloat16).to(self.device).eval()

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
        input_ids = torch.tensor([prompt_ids], device=self.device)

        with torch.no_grad():
            token_embeds = self.model.model.embed_tokens(input_ids)
            spk_hidden = self.speaker_proj(speaker_emb.to(torch.bfloat16)).unsqueeze(0).unsqueeze(0)
            inputs_embeds = torch.cat([spk_hidden, token_embeds], dim=1)

            attention_mask = torch.ones(1, inputs_embeds.shape[1], device=self.device, dtype=torch.long)

            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=[self.audio_end_id, self.eos_id],
                pad_token_id=self.eos_id,
            )

        kanade_indices = [
            tid - self.audio_token_start
            for tid in outputs[0].tolist()
            if self.audio_token_start <= tid <= self.audio_token_end
        ]

        if not kanade_indices:
            return None

        tokens_tensor = torch.tensor(kanade_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            mel = self.kanade.decode(
                content_token_indices=tokens_tensor,
                global_embedding=speaker_emb.float(),
            )
            waveform = vocode(self.vocoder, mel.unsqueeze(0))

        return waveform.squeeze().cpu().numpy()

    def _extract_speaker_emb(self, wav_path: str) -> torch.Tensor:
        import torchaudio

        wav, sr = torchaudio.load(wav_path)
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
