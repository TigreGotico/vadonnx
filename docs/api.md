# API reference

## `load_vad(name="silero", **kwargs) -> VADModel`

Load a VAD model behind the unified API.

| arg | meaning |
|-----|---------|
| `name` | model name (`"silero"`, `"silero-op15"`, `"silero-8k"`, `"marblenet"`/`-int8`, `"pyannote"`/`-int8`, `"fsmn"`/`-quant`, `"speechbrain"`, `"ten"`), a local `.onnx` path, or an `http(s)://` URL |
| `signature` | `IOSignature` or dict, required for a raw `.onnx` without a `<model>.signature.json` sidecar |
| `threshold` | activation threshold. When omitted, each backend uses its own default (0.5 for most, 0.7 for `speechbrain`) |
| `neg_threshold` | deactivation threshold (default `threshold - 0.15`) |
| `providers` | ONNX Runtime execution providers (default `["CPUExecutionProvider"]`) |
| `intra_threads`, `inter_threads` | ORT threading (default 1) |
| `revision` | pin a HuggingFace revision |
| `cache_dir` | override the download cache |

## `list_models() -> list[str]`

All known model names (built-in + registered + entry-point plugins).

## `register_model(spec: ModelSpec) -> None`

Register a `ModelSpec` at runtime.

## `class VADModel`

The interface every backend exposes.

| attribute / method | description |
|--------------------|-------------|
| `sample_rate`, `frame_size`, `stateful`, `threshold`, `neg_threshold` | model properties |
| `frame_duration` | seconds per frame (`frame_size / sample_rate`) |
| `process_chunk(audio, sample_rate=None) -> float` | streaming, returns the P(speech) of the last frame |
| `flush() -> float` | process trailing buffered audio at end of stream |
| `__call__(audio, sample_rate=None)` | alias for `process_chunk` |
| `is_speech(audio, sample_rate=None, threshold=None) -> bool` | boolean wrapper |
| `probabilities(audio, sample_rate=None) -> np.ndarray` | per-frame probabilities (resets first) |
| `get_speech_segments(audio, sample_rate=None, **opts) -> list[SpeechSegment]` | detect speech spans |
| `reset()` | clear streaming buffer + recurrent state |

`get_speech_segments` options: `threshold`, `neg_threshold`, `min_speech_duration`,
`min_silence_duration`, `max_speech_duration`, `speech_pad` (all seconds).

Accepted `audio` types everywhere: `int16` PCM **bytes**, or numpy arrays of
`int16` / `int32` / `uint8` / `float32` / `float64`. 2-D arrays are downmixed to mono.

## `class SpeechSegment`

`start`, `end` (seconds), `duration` property, iterable as `(start, end)`.

## `class IOSignature`

Declarative IO wiring for a model. See [custom_models.md](custom_models.md).

## `probs_to_segments(probs, frame_duration, **opts) -> list[SpeechSegment]`

The standalone segmentation function used by `get_speech_segments`.

## `vadonnx.audio`

- `read_wav(path) -> (np.ndarray, int)`: read a PCM WAV to mono float32.
- `to_float32_mono(audio) -> np.ndarray`: normalize any accepted input.
- `resample(x, src_sr, dst_sr) -> np.ndarray`.

---
[← Licensing](licensing.md) · [Home](README.md)
