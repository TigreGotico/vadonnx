# Backends & parity notes

Every model is exposed through the same `VADModel` API. Only `silero` is bundled in the
wheel; the rest download from the [`TigreGotico`](https://huggingface.co/TigreGotico) HF
org on first use and cache under `$XDG_DATA_HOME/vadonnx`.

"Parity" below means how closely `vadonnx`'s output matches the upstream reference
implementation on the same audio.

## `silero` / `silero-8k` — ✅ verified parity

- **Source:** [snakers4/silero-vad](https://github.com/snakers4/silero-vad) v6, MIT.
- **Parity:** **bit-exact** — per-frame probability MAE `0.0000` vs the official
  `silero-vad` ONNX wrapper (both fp32 and dynamic-int8).
- **Frontend:** raw PCM, 32 ms frames (512 samples @ 16k / 256 @ 8k), with a 64/32-sample
  context window carried across frames and a single recurrent `state` tensor.
- **int8:** *not shipped* — the model is already ~2.3 MB and dynamic quantization makes it
  slightly **larger** with no speed benefit.
- Recommended default. Works fully offline.

## `marblenet` — ⚠️ best-effort (near-parity)

- **Source:** [nvidia/frame_vad_multilingual_marblenet_v2.0](https://huggingface.co/nvidia/frame_vad_multilingual_marblenet_v2.0),
  NVIDIA Open Model License (see [licensing.md](licensing.md)).
- **Parity:** end-to-end probability MAE **~4e-4** vs NeMo. The features→logits ONNX is
  bit-exact given features; the 80-mel frontend (preemphasis, centered STFT, NeMo's mel
  filterbank + window, `log(x+5.96e-8)`) is reproduced in numpy. NeMo's `torch.stft`
  preprocessor is not ONNX-exportable, so the frontend runs in numpy with NeMo's exact
  filterbank/window bundled in the wheel.
- **Frontend:** log-mel, 20 ms output frames; multilingual; `P(speech)=softmax(logits)[1]`.
- **int8:** available as part of the same repo (`marblenet_int8.onnx`, ~36% size,
  MAE ~7e-3 vs fp32).
- Good general-purpose multilingual VAD.

## `pyannote` / `pyannote-int8` — ⚠️ community ONNX (strong)

- **Source:** [onnx-community/pyannote-segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)
  (MIT) — an ONNX export of [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  (MIT, gated upstream). Re-hosted (fp32 + int8) on the TigreGotico org, pinned.
- **Architecture:** powerset speaker-segmentation model run as VAD. Takes a raw 10 s
  waveform → per-frame logits over 7 classes (class 0 = non-speech); the backend slides a
  non-overlapping 10 s window over the audio and emits `P(speech) = 1 - softmax[0]` at
  ~17 ms resolution.
- **Quality:** the strongest separation of the bundled models on clean speech (no feature
  reconstruction — raw waveform in). Heavier than Silero (10 s windows). int8 available.
- A great choice when you already use pyannote for diarization.

## `fsmn` / `fsmn-quant` — ⚠️ best-effort

- **Source:** [funasr/fsmn-vad-onnx](https://huggingface.co/funasr/fsmn-vad-onnx), MIT.
- **Parity:** not bit-exact. Frame-level speech/silence separation is clear (on the test
  clip, speech-window mean `0.95` vs silence `0.37`). Uses the FunASR recipe:
  kaldi-style fbank (80-mel) → LFR(`m=5`) → CMVN → FSMN with four recurrent caches;
  `P(speech) = 1 - softmax[..., 0]` (silence PDF). CMVN stats bundled in the wheel.
- **Frontend dependency:** `kaldi-native-fbank` — install the `fsmn` extra:
  `uv pip install "vadonnx[fsmn]"` (or `pip install kaldi-native-fbank`).
- **int8:** `fsmn-quant` (`model_quant.onnx`, ~29% size, MAE ~5e-3 vs fp32).
- Chinese-tuned but generalises; heavier frontend than Silero.

## `speechbrain` — ⚠️ best-effort (first-to-ONNX, high recall)

- **Source:** [speechbrain/vad-crdnn-libriparty](https://huggingface.co/speechbrain/vad-crdnn-libriparty), Apache-2.0.
- **First-to-ONNX:** SpeechBrain ships no ONNX; vadonnx exports the features→posterior
  CRDNN graph and reproduces SpeechBrain's exact Fbank in numpy (mel matrix + Hamming
  window bundled). **End-to-end parity vs SpeechBrain is exact (MAE 0).**
- **Frontend:** 40-mel log-Fbank, 10 ms frames; CNN→RNN→DNN→sigmoid posterior.
- **Caveat:** LibriParty-trained (cocktail-party speech), so it is **high-recall** —
  it over-detects on clean broadcast speech (silence can score ~0.95). Default
  `threshold` is raised to 0.7; still best for noisy/overlapping conditions, not clean
  single-speaker. The benchmark quantifies this.
- No int8 (small model already).

## `ten` — ⚠️ experimental / reference only

- **Source:** [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad), Apache-2.0.
- **Status:** TEN's ONNX expects a precomputed `[B, 3, 41]` mel+pitch feature tensor and
  four recurrent states; **feature extraction lives in TEN's native C library**, which
  `vadonnx` does not reproduce. The ONNX and its discovered signature are published for
  experimentation, but the pure-ONNX path does **not** match upstream and should not be
  relied on for production. Use Silero or MarbleNet instead.

## Choosing

| need | use |
|------|-----|
| best default, offline, fast | `silero` |
| multilingual, general | `marblenet` |
| FunASR-compatible pipeline | `fsmn` |
| smallest footprint | `marblenet` int8 / `fsmn-quant` |
