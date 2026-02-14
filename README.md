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

### Choose a speaker

Built-in speakers are loaded from the package. The first speaker is used by default.

```python
tts.speak("Hej med dig.", output="output.wav", speaker="nic")
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
