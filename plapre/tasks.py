"""Prompt construction for the plapre multi-task checkpoints (e.g. ``syvai/plapre-nano-v2``).

These models are trained on **2 tasks** — *generate* (text→audio) and *edit* (masked audio
infill) — plus **4 composable controls** that layer onto generate: *voice-reference/clone*,
*context*, *duration/pace*, and *pronunciation references*.

This module holds the token layout only (no model, no GPU): :class:`TaskTokens` resolves the
control-token ids from a tokenizer, and the ``*_prompt`` builders assemble the exact id
sequence each mode was trained on. Keeping it pure makes the layout unit-testable and keeps
:class:`plapre.inference.Plapre` focused on generation. The builders mirror the training
harness (``train_p1_ablate.py`` ``P1MultiTask``); the sequence order is::

    [SPK]  [lex]  [voice-ref]  [context]   <text> BPE  [<dur>…</dur>]  <audio> …audio…  <eos>

Current multi-task checkpoints end generated audio with a trained ``</audio>`` terminator
before ``<eos>`` — generation must stop on BOTH (:attr:`TaskTokens.stop_ids`); the token is
absent (``None``) on older vocabs.

The speaker embedding is prepended by the engine (not a token), so these builders return the
token-id list that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass

# training-time constants (train_p1_ablate.py)
AUDIO_VOCAB = 12800  # <audio_0..12799>
NUM_DUR = 400        # <dur_0..399>, 1 frame = 40 ms @ 25 Hz
REF_AUDIO_CAP = 250  # voice-reference audio tokens
CTX_AUDIO_CAP = 150  # context (previous-utterance) audio tokens
LEX_MAX_REFS = 10    # pronunciation reference pairs
LEX_AUDIO_CAP = 60   # audio tokens per pronunciation reference
LEX_TOTAL_AUDIO_CAP = 200  # audio tokens across all pronunciation references

_CONTROL_TOKENS = {
    "text": "<text>", "audio": "<audio>",
    "audio_end": "</audio>",   # trained audio terminator (None on older vocabs)
    "audio_base": "<audio_0>", "dur_base": "<dur_0>",
    "dur": "<dur>", "dur_end": "</dur>",
    "ref_text": "<ref_text>", "ref_text_end": "</ref_text>",
    "ref_audio": "<ref_audio>", "ref_audio_end": "</ref_audio>",
    "context": "<context>", "context_end": "</context>",
    "spk_same": "<spk_same>", "spk_diff": "<spk_diff>",
    "ctx_audio": "<ctx_audio>", "ctx_audio_end": "</ctx_audio>",
    "edit_mask": "<edit_mask>", "edit_fill": "<edit_fill>", "edit_end": "<edit_end>",
    "lex": "<lex>", "lex_end": "</lex>",
    "lex_word": "<lex_word>", "lex_word_end": "</lex_word>",
    "lex_audio": "<lex_audio>", "lex_audio_end": "</lex_audio>",
}

# controls whose presence marks a mode-capable checkpoint
_MODE_MARKERS = ("ref_text", "context", "dur", "edit_mask", "lex")


@dataclass
class TaskTokens:
    """Resolved token ids for every control block. ``None`` for any token the tokenizer
    lacks (e.g. a base ``plapre-nano`` checkpoint has none of these)."""

    ids: dict[str, int | None]
    eos: int

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "TaskTokens":
        vocab = tokenizer.get_vocab()  # str -> id; .get() returns None for absent tokens
        ids = {key: vocab.get(tok) for key, tok in _CONTROL_TOKENS.items()}
        return cls(ids=ids, eos=tokenizer.eos_token_id)

    def __getattr__(self, name: str) -> int | None:
        # attribute access for every control token: self.ref_text, self.audio_base, …
        try:
            return self.__dict__["ids"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def supports_modes(self) -> bool:
        return all(self.ids.get(k) is not None for k in _MODE_MARKERS)

    @property
    def stop_ids(self) -> list[int]:
        """Generation stop tokens: ``<eos>``, plus the trained ``</audio>`` terminator when
        the vocab has one. Stopping on both is required for correct termination."""
        stops = [self.eos]
        if self.ids.get("audio_end") is not None:
            stops.append(self.ids["audio_end"])
        return stops

    def audio_ids(self, kanade_tokens) -> list[int]:
        base = self.ids["audio_base"]
        return [base + int(k) for k in kanade_tokens]

    def dur_ids(self, frames) -> list[int]:
        base = self.ids["dur_base"]
        return [base + max(0, min(NUM_DUR - 1, int(f))) for f in frames]


# --------------------------------------------------------------------------- #
# Prompt builders. Each takes already-encoded text-id lists (the caller normalizes and
# encodes) and returns the token-id sequence that follows the prepended speaker embedding.
# `lex` is an optional pronunciation prefix from `lex_prefix()`; it composes with any task.
# --------------------------------------------------------------------------- #

def base_prompt(T: TaskTokens, text_ids, dur_frames=None, lex=None) -> list[int]:
    """generate: ``[lex] <text> BPE [<dur>…</dur>] <audio>``."""
    p = list(lex or [])
    p += [T.text] + list(text_ids)
    if dur_frames is not None:
        p += [T.dur] + T.dur_ids(dur_frames) + [T.dur_end]
    p += [T.audio]
    return p


def clone_prompt(T: TaskTokens, text_ids, ref_text_ids, ref_audio_tokens,
                 dur_frames=None, lex=None) -> list[int]:
    """voice-reference/clone: prepend the reference clip; the engine must also use the
    reference clip's speaker embedding (that is what carries the voice)."""
    p = list(lex or [])
    p += ([T.ref_text] + list(ref_text_ids) + [T.ref_text_end]
          + [T.ref_audio] + T.audio_ids(ref_audio_tokens)[:REF_AUDIO_CAP] + [T.ref_audio_end])
    p += [T.text] + list(text_ids)
    if dur_frames is not None:
        p += [T.dur] + T.dur_ids(dur_frames) + [T.dur_end]
    p += [T.audio]
    return p


def context_prompt(T: TaskTokens, text_ids, prev_text_ids, same_speaker,
                   prev_audio_tokens=None, dur_frames=None, lex=None) -> list[int]:
    """context: condition on the previous utterance (text always; audio when given)."""
    p = list(lex or [])
    p += [T.context, T.spk_same if same_speaker else T.spk_diff] + list(prev_text_ids)
    if prev_audio_tokens is not None:
        p += [T.ctx_audio] + T.audio_ids(prev_audio_tokens)[:CTX_AUDIO_CAP] + [T.ctx_audio_end]
    p += [T.context_end] + [T.text] + list(text_ids)
    if dur_frames is not None:
        p += [T.dur] + T.dur_ids(dur_frames) + [T.dur_end]
    p += [T.audio]
    return p


def edit_prompt(T: TaskTokens, new_text_ids, original_tokens, mask_start, mask_end,
                lex=None) -> list[int]:
    """edit: full desired text + original audio with ``[mask_start:mask_end)`` masked out.
    The model then generates the fill audio, stopping at ``<edit_end>``; splice it back:
    ``original[:mask_start] + fill + original[mask_end:]``."""
    a = T.audio_ids(original_tokens)
    p = list(lex or [])
    p += ([T.text] + list(new_text_ids) + [T.audio]
          + a[:mask_start] + [T.edit_mask] + a[mask_end:] + [T.edit_fill, T.edit_mask])
    return p


def lex_prefix(T: TaskTokens, pairs) -> list[int]:
    """pronunciation references: ``pairs`` = ``[(word_text_ids, ref_audio_tokens), …]`` (≤10).
    Each pair binds a word's BPE to a rendition of it (≤60 audio tokens, ≤200 total)."""
    out, total = [T.lex], 0
    for word_ids, ref_tokens in list(pairs)[:LEX_MAX_REFS]:
        cap = min(LEX_AUDIO_CAP, LEX_TOTAL_AUDIO_CAP - total)
        if cap <= 0:
            break
        ref = T.audio_ids(ref_tokens)[:cap]
        total += len(ref)
        out += ([T.lex_word] + list(word_ids) + [T.lex_word_end]
                + [T.lex_audio] + ref + [T.lex_audio_end])
    out += [T.lex_end]
    return out
