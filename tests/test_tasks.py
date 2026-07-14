"""Prompt-layout tests for plapre.tasks — pure, no model/GPU. Runnable via `pytest` or
directly with `python tests/test_tasks.py`. Imports tasks.py in isolation so the package's
heavy deps (torch/vllm/soundfile) aren't required."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "plapre_tasks", Path(__file__).resolve().parent.parent / "plapre" / "tasks.py"
)
tasks = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tasks
_spec.loader.exec_module(tasks)
TaskTokens = tasks.TaskTokens


def _tokens():
    vocab = {tok: 100 + i for i, tok in enumerate(tasks._CONTROL_TOKENS.values())}
    vocab["<audio_0>"] = 1000
    vocab["<dur_0>"] = 2000

    class _Tok:
        def get_vocab(self):
            return vocab

        eos_token_id = 9

    return TaskTokens.from_tokenizer(_Tok()), vocab


def test_supports_modes_detection():
    T, _ = _tokens()
    assert T.supports_modes is True

    class _Bare:
        def get_vocab(self):
            return {"<audio_0>": 1000, "<text>": 1, "<audio>": 2}

        eos_token_id = 9

    assert TaskTokens.from_tokenizer(_Bare()).supports_modes is False


def test_audio_and_dur_offsets():
    T, _ = _tokens()
    assert T.audio_ids([0, 5, 12799]) == [1000, 1005, 1000 + 12799]
    assert T.dur_ids([0, 3, 999]) == [2000, 2003, 2000 + 399]  # clamped to NUM_DUR-1


def test_base_prompt_with_dur_and_lex():
    T, v = _tokens()
    lex = tasks.lex_prefix(T, [([21, 22], [5, 6, 7])])
    assert lex[0] == v["<lex>"] and lex[-1] == v["</lex>"]
    assert lex[6:9] == [1005, 1006, 1007]
    p = tasks.base_prompt(T, [11, 12, 13], dur_frames=[3, 4], lex=lex)
    i = len(lex)
    assert p[:i] == lex
    assert p[i] == v["<text>"] and p[i + 1:i + 4] == [11, 12, 13]
    assert p[i + 4] == v["<dur>"] and p[i + 5:i + 7] == [2003, 2004] and p[i + 7] == v["</dur>"]
    assert p[-1] == v["<audio>"]


def test_clone_prompt_caps_ref_audio():
    T, v = _tokens()
    p = tasks.clone_prompt(T, [11], [31, 32], list(range(300)))
    assert p.index(v["</ref_audio>"]) - p.index(v["<ref_audio>"]) - 1 == 250
    assert p[-1] == v["<audio>"]


def test_context_prompt_flag_and_cap():
    T, v = _tokens()
    p = tasks.context_prompt(T, [11], [41], same_speaker=False, prev_audio_tokens=list(range(200)))
    assert p[0] == v["<context>"] and p[1] == v["<spk_diff>"]
    assert p.index(v["</ctx_audio>"]) - p.index(v["<ctx_audio>"]) - 1 == 150
    assert tasks.context_prompt(T, [11], [41], True)[1] == v["<spk_same>"]
    # text-only context omits the ctx_audio block
    assert v["<ctx_audio>"] not in tasks.context_prompt(T, [11], [41], True)


def test_edit_prompt_mask_structure():
    T, v = _tokens()
    a = list(range(20))
    p = tasks.edit_prompt(T, [11, 12], a, 5, 9)
    seg = p[p.index(v["<audio>"]) + 1:]
    assert seg[:5] == [1000, 1001, 1002, 1003, 1004]
    assert seg[5] == v["<edit_mask>"]
    assert seg[6:17] == [1000 + k for k in range(9, 20)]
    assert seg[-2] == v["<edit_fill>"] and seg[-1] == v["<edit_mask>"]


def test_lex_total_audio_budget():
    T, _ = _tokens()
    big = tasks.lex_prefix(T, [([1], [0] * 60)] * 10)  # 10 refs x 60 = 600 > 200 cap
    assert sum(1 for x in big if 1000 <= x < 2000) == tasks.LEX_TOTAL_AUDIO_CAP


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL TESTS PASSED")
