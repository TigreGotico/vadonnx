# Changelog

## 0.1.0 (unreleased)

Initial release.

- Unified ONNX VAD API: `load_vad()`, `VADModel`, `IOSignature`, `SpeechSegment`.
- Generic declarative `OnnxVAD` engine driven by per-model IO signatures.
- Built-in backends: `silero`, `silero-op15`, `silero-8k`, `marblenet` (+int8),
  `fsmn` (+quant), `speechbrain` (first-to-ONNX), `pyannote` (+int8), `ten` (reference).
- Streaming (`process_chunk`) and batch (`probabilities`, `get_speech_segments`) APIs.
- Bundled Silero v6 model for fully offline default usage.
- Model registry with local-path / URL / HuggingFace resolution and entry-point plugins.
- Conversion scripts that fetch/export and publish ONNX models to the `TigreGotico` HF org.
- `vadonnx` CLI (`list`, `probe`, `segment`).
