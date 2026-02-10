# Plapre

Danish text-to-speech synthesis.

## Installation

Requires Python 3.12+ and [espeak-ng](https://github.com/espeak-ng/espeak-ng) installed on your system.

```bash
# Install espeak-ng (required for phonemization)
# macOS
brew install espeak-ng
# Ubuntu/Debian
sudo apt install espeak-ng

# Install plapre
pip install git+https://github.com/syv-ai/plapre.git
```

## Usage

### Basic

```python
from plapre import Plapre

tts = Plapre("syvai/plapre-turbo")
tts.speak("Hej, hvordan har du det?", output="output.wav")
```

### Choose a speaker

Built-in speakers are loaded from the model repo. The first speaker is used by default.

```python
# Use a specific speaker
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

### Using a speaker embedding directly

```python
import torch

# 128-dim speaker embedding from Kanade encoder
speaker_emb = torch.randn(128)
tts.speak("Hej.", output="output.wav", speaker_emb=speaker_emb)
```

### Return value

`speak()` returns the audio as a numpy array (24 kHz, float32), in addition to saving the file:

```python
audio = tts.speak("Hej.", output="output.wav")
print(f"Duration: {len(audio) / 24000:.2f}s")
```
