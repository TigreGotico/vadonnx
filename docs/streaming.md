# Streaming

`vadonnx` is designed for live, chunk-by-chunk audio as well as whole-buffer
processing.

## The streaming primitive

```python
prob = vad.process_chunk(audio, sample_rate=None)
```

- `audio` may be `int16` PCM **bytes**, or a numpy array (`int16`/`int32`/`float32`).
- `sample_rate` is the rate of *your* audio; `vadonnx` resamples to the model's native
  rate internally. If omitted, the audio is assumed to already be at the model rate.
- Returns the speech probability of the **most recent** completed frame.

Chunks of any size are accepted. Internally a buffer accumulates samples and the model
runs once per `frame_size` samples; a chunk that doesn't complete a frame returns the
previous probability.

```python
vad = load_vad("silero")
vad.reset()
for chunk in mic_stream():                 # arbitrary chunk sizes
    if vad.process_chunk(chunk, sample_rate=16000) >= vad.threshold:
        on_speech()
```

## Resetting between utterances

Stateful models (e.g. Silero) carry recurrent state across frames. Call `reset()` when
a new, independent utterance begins so state from the previous one doesn't leak:

```python
vad.reset()
```

`probabilities()` and `get_speech_segments()` reset automatically, so batch calls are
always independent.

## Boolean convenience

```python
if vad.is_speech(chunk, sample_rate=16000):
    ...
```

## Frame timing

Each frame represents `vad.frame_duration` seconds (`frame_size / sample_rate`). For
Silero at 16 kHz that is `512 / 16000 = 32 ms`.
