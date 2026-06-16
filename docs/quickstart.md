# Quickstart

## Install

```bash
uv pip install vadonnx
```

The default `silero` model is bundled in the wheel, so this works with no network.

## Detect speech segments in a file

```python
from vadonnx import load_vad
from vadonnx.audio import read_wav

audio, sr = read_wav("speech.wav")        # mono float32, any PCM WAV
vad = load_vad("silero")
for seg in vad.get_speech_segments(audio, sample_rate=sr):
    print(f"{seg.start:.2f}s -> {seg.end:.2f}s ({seg.duration:.2f}s)")
```

## Per-frame probabilities

```python
probs = vad.probabilities(audio, sample_rate=sr)   # np.ndarray, one value per frame
```

## Streaming

```python
vad = load_vad("silero")
vad.reset()                                # clear state between utterances
while chunk := mic.read():                 # chunk: int16 PCM bytes / numpy array
    prob = vad.process_chunk(chunk, sample_rate=16000)
    if prob >= 0.5:
        ...                                # speech in this chunk
```

Chunks need not align to the model frame size — leftover samples are buffered.

## Tuning

```python
segments = vad.get_speech_segments(
    audio, sample_rate=sr,
    threshold=0.5,              # activation threshold
    min_speech_duration=0.25,   # discard blips shorter than this (s)
    min_silence_duration=0.1,   # silence needed to end a segment (s)
    max_speech_duration=30.0,   # split very long segments (s)
    speech_pad=0.03,            # pad each segment (s)
)
```

## Choosing a model

```python
from vadonnx import list_models
print(list_models())            # ['fsmn', 'fsmn-quant', 'marblenet', 'silero', 'silero-8k', 'ten']

vad = load_vad("silero-8k")     # 8 kHz variant
```

See [backends.md](backends.md) for the trade-offs and parity status of each model.
