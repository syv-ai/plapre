# Plapre

Danish text-to-speech synthesis. Uses vLLM for fast batched inference.

## Prerequisites

The Plapre models are hosted as **gated models** on Hugging Face. Before using Plapre, you need to:

1. **Accept the model agreement** on the model page:
   - [syvai/plapre-nano](https://huggingface.co/syvai/plapre-nano)
   - [syvai/plapre-pico](https://huggingface.co/syvai/plapre-pico)
2. **Create a Hugging Face token** with `Read access to contents of all public gated repos you can access` permission at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. **Log in** via the CLI:
   ```bash
   huggingface-cli login
   ```

## Installation

Requires Python >= 3.12 and a CUDA GPU.

```bash
uv add git+https://github.com/syv-ai/plapre.git
```

## Models

| Model | Parameters | HuggingFace | Description |
|-------|-----------|-------------|-------------|
| Plapre Nano | ~327M | [syvai/plapre-nano](https://huggingface.co/syvai/plapre-nano) | Larger, higher quality |
| Plapre Pico | ~118M | [syvai/plapre-pico](https://huggingface.co/syvai/plapre-pico) | Smaller, faster inference |

## Usage

### Basic

```python
from plapre import Plapre

tts = Plapre("syvai/plapre-nano")
tts.speak("Hej, hvordan har du det?", output="output.wav")
```

### GPU memory

```python
# Adjust GPU memory allocated to vLLM (default: 0.4)
# Leave headroom for the Kanade vocoder (~400 MiB)
tts = Plapre("syvai/plapre-nano", gpu_memory_utilization=0.5)
```

### Choose a speaker

Built-in speakers are loaded from the package. The first speaker is used by default.

```python
tts.speak("Hej med dig.", output="output.wav", speaker="mic")
```

### Voice cloning

```python
tts.speak("Hej med dig.", output="cloned.wav", speaker_wav="reference.wav")
```

### Long text with sentence splitting

Sentences are batched together in a single vLLM call for higher throughput.

```python
tts.speak(
    "Første sætning. Anden sætning. Tredje sætning!",
    output="long.wav",
    split_sentences=True,
)
```

### Generation parameters

```python
tts.speak(
    "Hej verden.",
    output="output.wav",
    temperature=0.8,     # sampling temperature (default: 0.8)
    top_p=0.95,          # nucleus sampling (default: 0.95)
    top_k=50,            # top-k sampling (default: 50)
    max_tokens=500,      # max audio tokens to generate (default: 500)
)
```

### Extract a speaker embedding

Extract a 128-dim speaker embedding from a wav file, then reuse it across multiple generations:

```python
speaker_emb = tts.extract_speaker("reference.wav")
tts.speak("Hej.", output="a.wav", speaker_emb=speaker_emb)
tts.speak("Farvel.", output="b.wav", speaker_emb=speaker_emb)
```

### Return value

`speak()` returns the audio as a numpy array (24 kHz, float32), in addition to saving the file:

```python
audio = tts.speak("Hej.", output="output.wav")
print(f"Duration: {len(audio) / 24000:.2f}s")
```

## Multi-task inference modes

The multi-task checkpoint [`syvai/plapre-nano-v2`](https://huggingface.co/syvai/plapre-nano-v2)
adds four extra ways to control generation on top of plain TTS: **voice cloning**, **contextual
prosody**, **pace/duration**, **audio editing**, and **pronunciation references**. Load it with
`quant=None` (it ships safetensors, no GGUF) and a larger `max_model_len` (reference audio makes
the prompt longer):

```python
from plapre import Plapre

tts = Plapre("syvai/plapre-nano-v2", quant=None, max_model_len=1536)

# 1) Clone a voice — speak text in the voice of a reference clip
tts.clone("Hej med dig.", output="clone.wav", reference_wav="target_voice.wav")

# 2) Pace / duration — one frame count per word (40 ms each), len == #words
tts.speak_paced("Det går langsomt.", durations=[18, 14, 30], output="slow.wav")

# 3) Context — condition on the previous line for prosodic continuity
tts.continue_context("Og så tog han afsted.", prev_text="Han kiggede på uret.",
                     prev_wav="prev_line.wav", same_speaker=True, output="ctx.wav")

# 4) Edit — regenerate a word span in an existing clip (Kanade-frame indices, 25 Hz)
tts.edit("Det var en god dag.", mask_start=40, mask_end=55,
         original_wav="original.wav", min_tokens=12, output="edit.wav")

# 5) Pronunciation — pin how a hard word should sound with reference audio of it
tts.pronounce("Ifølge Verdenssundhedsorganisationen stiger tallet.",
              pronunciations=[("Verdenssundhedsorganisationen", "who_word.wav")],
              output="fixed.wav")
```

Controls compose: `clone(...)`, `speak_paced(...)`, `continue_context(...)` all accept
`durations=` and `pronunciations=`, and `edit(...)` accepts `pronunciations=`. Reference audio can
be a wav path or a list of Kanade tokens. `plapre/tasks.py` holds the exact prompt layout for each
mode. See `examples/run_modes.py` for an end-to-end script, and the model card for the token
structures.

## API Server

Plapre includes a FastAPI server that streams raw PCM audio with chunked transfer encoding, so clients can start playback before the full response is generated.

### Install

```bash
uv add "plapre[serve] @ git+https://github.com/syv-ai/plapre.git"
```

### Start the server

```bash
plapre-serve --port 8000

# Or with a specific checkpoint and GPU memory settings
plapre-serve --checkpoint syvai/plapre-nano --gpu-mem 0.5 --port 8000
```

Configuration via environment variables is also supported: `PLAPRE_CHECKPOINT`, `PLAPRE_GPU_MEM`, `PLAPRE_MAX_MODEL_LEN`.

### Generate speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hej, hvordan har du det?", "speaker": "mic"}' \
  --output output.pcm

# Convert to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

The response is raw PCM (16-bit signed LE, 24kHz, mono) streamed per-sentence.

### Other endpoints

```bash
# List available speakers
curl http://localhost:8000/v1/speakers

# Health check
curl http://localhost:8000/health
```
