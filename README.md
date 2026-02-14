# Plapre

Danish text-to-speech synthesis. Uses llama.cpp for fast inference.

## Installation

```bash
# Install plapre
uv add git+https://github.com/syv-ai/plapre.git
```

## Usage

### Basic

```python
from plapre import Plapre

tts = Plapre("syvai/plapre-nano")
tts.speak("Hej, hvordan har du det?", output="output.wav")
```

### Quantization

GGUF models are downloaded automatically. Available quants: `f16`, `q8_0` (default), `q6_k`, `q4_k_m`, `q4_0`.

```python
tts = Plapre("syvai/plapre-nano", quant="q4_k_m")
```

Or use a local GGUF file:

```python
tts = Plapre("syvai/plapre-nano", model_path="/path/to/model.gguf")
```

### List available speakers

```python
print(tts.list_speakers())
# ['tor', 'ida', 'liv', 'ask', 'kaj']
```

### Choose a speaker

Built-in speakers are loaded from the package. The first speaker is used by default.

```python
tts.speak("Hej med dig.", output="output.wav", speaker="ida")
```

### Voice cloning

```python
tts.speak("Hej med dig.", output="cloned.wav", speaker_wav="reference.wav")
```

### Long text with sentence splitting

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

## API Server

Plapre includes a FastAPI server that streams raw PCM audio with chunked transfer encoding, so clients can start playback before the full response is generated.

### Install

```bash
pip install "plapre[serve]"
```

### Start the server

```bash
plapre-serve --model plapre-nano-q8_0 --port 8000
```

Available models: `plapre-nano-f16`, `plapre-nano-q8_0` (default), `plapre-nano-q6_k`, `plapre-nano-q4_k_m`, `plapre-nano-q4_0`.

### Generate speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hej, hvordan har du det?", "speaker": "tor"}' \
  --output output.pcm

# Convert to WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

The response is raw PCM (16-bit signed LE, 24kHz, mono) streamed per-sentence.

### Other endpoints

```bash
# List available speakers
curl http://localhost:8000/v1/speakers

# List models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```
