# vadonnx

Load arbitrary **Voice Activity Detection** models behind a single, unified API — every
model runs through [ONNX Runtime](https://onnxruntime.ai/). One streaming/​batch
interface, one audio-format story, pluggable models.

```python
from vadonnx import load_vad

vad = load_vad("silero")                              # bundled, works fully offline
prob = vad.process_chunk(pcm_bytes)                   # streaming → float in [0, 1]
segments = vad.get_speech_segments(audio, sample_rate=16000)
# -> [SpeechSegment(start=0.32, end=2.27), SpeechSegment(start=3.27, end=4.45), ...]
```

## Why

Every VAD ships its own loader, audio format, feature pipeline and state handling.
`vadonnx` hides that behind one `VADModel` interface: feed it audio (raw `int16`
bytes, numpy arrays, any sample rate) and get back per-frame speech probabilities or
ready-made speech segments. Models are described *declaratively* by an
[`IOSignature`](docs/custom_models.md), so a single generic engine drives most of them
and you can point the same API at any custom `.onnx` file.

- **Lightweight runtime** — only `numpy`, `onnxruntime`, `huggingface_hub`.
- **Offline by default** — a small Silero model is bundled in the wheel.
- **Streaming and batch** — `process_chunk()` for live audio, `get_speech_segments()` /
  `probabilities()` for whole buffers.
- **Bring your own model** — load any ONNX VAD by path/URL with a signature.
- **Extensible** — third parties register backends/models via entry points.

## Install

```bash
uv pip install vadonnx          # runtime (numpy + onnxruntime + huggingface_hub)
uv pip install "vadonnx[mic]"   # + microphone examples
```

## Models

| name | rate | parity vs upstream | notes |
|------|------|--------------------|-------|
| `silero` / `silero-8k` / `silero-op15` | 16k / 8k / 16k | MAE 0 | bundled default, raw PCM |
| `marblenet` / `marblenet-int8` | 16k | MAE 4e-4 | NVIDIA NeMo Frame-VAD, multilingual ([license](docs/licensing.md)) |
| `pyannote` / `pyannote-int8` | 16k | MAE 0 | pyannote segmentation-3.0, windowed |
| `fsmn` / `fsmn-quant` | 16k | tracks upstream | FunASR FSMN-VAD; needs `vadonnx[fsmn]` |
| `speechbrain` | 16k | MAE 0 | SpeechBrain CRDNN, LibriParty-trained |
| `ten` | 16k | — | feature extractor provided by TEN's native library |

See [docs/backends.md](docs/backends.md) for per-model detail and the
[benchmark](benchmark/results/REPORT.md) for measured comparisons across datasets,
including WebRTC and energy baselines.

Models other than the bundled Silero are downloaded on first use from the
[`TigreGotico`](https://huggingface.co/TigreGotico) HuggingFace org and cached under
`$XDG_DATA_HOME/vadonnx`. See [docs/backends.md](docs/backends.md) for per-model detail
and parity notes.

## CLI

```bash
vadonnx list                       # list available models
vadonnx probe silero               # print a model's ONNX input/output signature
vadonnx segment speech.wav         # print detected speech segments of a WAV
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Streaming](docs/streaming.md)
- [Custom models & `IOSignature`](docs/custom_models.md)
- [Backends & parity notes](docs/backends.md)
- [Plugins](docs/plugins.md)
- [Model conversion](docs/conversion.md)
- [Licensing](docs/licensing.md)
- [API reference](docs/api.md)

## License

Apache-2.0. Bundled/downloaded model weights retain their upstream licenses — see
[docs/licensing.md](docs/licensing.md).
