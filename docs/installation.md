# Installation

Always use `uv`.

## Runtime

```bash
uv pip install vadonnx
```

Pulls only `numpy`, `onnxruntime` and `huggingface_hub`. The default `silero` model is
bundled in the wheel, so it works fully offline.

## Extras

| extra | adds | for |
|-------|------|-----|
| `vadonnx[fsmn]` | `kaldi-native-fbank` | the FSMN backend's fbank frontend |
| `vadonnx[mic]` | `sounddevice` | the live-microphone example |
| `vadonnx[resample]` | `scipy` | higher-quality resampling (else numpy linear) |
| `vadonnx[all]` | all runtime optionals | |
| `vadonnx[test]` | `pytest`, `pytest-cov` | running the test suite |
| `vadonnx[convert]` | `onnx`, `huggingface_hub`, `requests` | converting silero/ten/fsmn |
| `vadonnx[convert-marblenet]` | `nemo_toolkit[asr]`, `torch` | exporting MarbleNet |
| `vadonnx[convert-speechbrain]` | `speechbrain`, `torch` | exporting SpeechBrain CRDNN |

## Model cache

Downloaded models are cached under `$XDG_DATA_HOME/vadonnx` (default
`~/.local/share/vadonnx`). Override per call with `load_vad(..., cache_dir=...)`.

## Supported Python

3.10 to 3.14.

---
[Home](README.md) · [Quickstart →](quickstart.md)
