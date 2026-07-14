"""Run every plapre-nano-v2 inference mode end to end and save a wav for each.

    python examples/run_modes.py --reference target_voice.wav --out out/

`--reference` is any short Danish clip; it's used as the clone target voice, the context
previous-utterance, the edit source, and (its first word) a pronunciation reference. All modes
need the multi-task checkpoint, loaded with quant=None.
"""

import argparse
import os

from plapre import Plapre


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="syvai/plapre-nano-v2")
    ap.add_argument("--reference", required=True, help="a short Danish wav (voice/context/edit source)")
    ap.add_argument("--out", default="modes_out")
    ap.add_argument("--gpu-memory", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tts = Plapre(
        args.checkpoint, quant=None, max_model_len=1536,
        gpu_memory_utilization=args.gpu_memory,
    )
    assert tts.supports_modes, f"{args.checkpoint} has no control tokens — use a multi-task checkpoint"

    o = lambda name: os.path.join(args.out, name)

    # 1) base TTS (built-in speaker)
    tts.speak("Det her er helt almindelig tale.", output=o("01_base.wav"))

    # 2) pace / duration — one frame count per word (40 ms each)
    tts.speak_paced(
        "Det går meget langsomt nu.",
        durations=[16, 14, 26, 22, 14],
        output=o("02_paced.wav"),
    )

    # 3) clone — this text in the reference clip's voice
    tts.clone(
        "Nu taler jeg med en anden stemme.",
        reference_wav=args.reference,
        output=o("03_clone.wav"),
    )

    # 4) clone + pace (controls compose)
    tts.clone(
        "Samme stemme, men langsommere.",
        reference_wav=args.reference,
        durations=[18, 16, 16, 30],
        output=o("04_clone_paced.wav"),
    )

    # 5) context — condition on a previous line
    tts.continue_context(
        "Og derfor blev mødet aflyst.",
        prev_text="Vi havde planlagt at ses i dag.",
        prev_wav=args.reference,
        same_speaker=True,
        output=o("05_context.wav"),
    )

    # 6) pronunciation — pin a hard word using reference audio of it
    tts.pronounce(
        "Ifølge Verdenssundhedsorganisationen stiger tallet fortsat.",
        pronunciations=[("Verdenssundhedsorganisationen", args.reference)],
        output=o("06_pronounce.wav"),
    )

    # 7) edit — regenerate a word span in the reference clip (Kanade frames @ 25 Hz).
    # Pick mask_start/mask_end from your clip's word alignment; here we edit a middle span.
    tts.edit(
        "Det var en rigtig god dag.",
        mask_start=30, mask_end=45,
        original_wav=args.reference,
        min_tokens=12,
        output=o("07_edit.wav"),
    )

    print(f"Wrote 7 mode outputs to {args.out}/")


if __name__ == "__main__":
    main()
