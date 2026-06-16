# Backends

Every model is exposed through the same `VADModel` API. `silero` is bundled in the
wheel; the others download from the [`TigreGotico`](https://huggingface.co/TigreGotico)
HuggingFace org on first use and cache under `$XDG_DATA_HOME/vadonnx`. Each download is
pinned to a specific revision.

The "parity" column reports the mean absolute error of per-frame speech probability
between `vadonnx` and the upstream reference implementation on the same audio.

| model | rate | frame | parity vs upstream | license |
|-------|------|-------|--------------------|---------|
| `silero` / `silero-8k` / `silero-op15` | 16k / 8k / 16k | 32 ms | MAE 0 | MIT |
| `marblenet` / `marblenet-int8` | 16k | 20 ms | MAE 4e-4 | NVIDIA Open Model License |
| `pyannote` / `pyannote-int8` | 16k | 17 ms | MAE 0 | MIT |
| `fsmn` / `fsmn-quant` | 16k | 10 ms | tracks upstream | MIT |
| `speechbrain` | 16k | 10 ms | MAE 0 | Apache-2.0 |
| `ten` | 16k | 16 ms | mel exact, pitch approximated | Apache-2.0 |

## `silero` / `silero-8k` / `silero-op15`

[Silero VAD](https://github.com/snakers4/silero-vad) v6. Raw PCM, 32 ms frames (512
samples at 16 kHz, 256 at 8 kHz) with a 64/32-sample context window carried across
frames and a single recurrent state tensor. Bundled in the wheel and usable offline.
`silero-op15` is the 16 kHz-only opset-15 build (smaller; for older ONNX Runtimes).
Dynamic int8 quantization is not provided — it does not reduce the model size.

## `marblenet` / `marblenet-int8`

NVIDIA NeMo [Frame-VAD MarbleNet](https://huggingface.co/nvidia/frame_vad_multilingual_marblenet_v2.0),
multilingual. Classifies 80-mel log-spectrogram features (preemphasis, centered STFT,
NeMo's mel filterbank and window, `log(x + 5.96e-8)`); `P(speech) = softmax(logits)[1]`
per 20 ms frame. The mel filterbank and window are bundled in the wheel. `marblenet-int8`
is the dynamically quantized graph (~36% of the size). License obligations: see
[licensing.md](licensing.md).

## `pyannote` / `pyannote-int8`

[pyannote segmentation-3.0](https://huggingface.co/onnx-community/pyannote-segmentation-3.0),
a powerset speaker-segmentation model used for voice activity. Takes a raw 10 s waveform
and emits per-frame logits over 7 classes (class 0 = non-speech). The backend slides a
non-overlapping 10 s window over the audio and returns `P(speech) = 1 - softmax[0]` at
~17 ms resolution. `pyannote-int8` is the quantized graph. Integrates naturally with
pyannote diarization pipelines.

## `fsmn` / `fsmn-quant`

FunASR [FSMN-VAD](https://huggingface.co/funasr/fsmn-vad-onnx). Classifies kaldi-style
fbank features (80-mel) stacked at low frame rate (`lfr_m=5`) and CMVN-normalized to a
400-dim input, through a streaming FSMN with four recurrent caches;
`P(speech) = 1 - softmax[0]` (silence is PDF id 0). The CMVN statistics are bundled in
the wheel. Requires the `fsmn` extra (`kaldi-native-fbank`) for the fbank frontend:
`uv pip install "vadonnx[fsmn]"`. `fsmn-quant` is the int8 graph.

## `speechbrain`

[SpeechBrain CRDNN VAD](https://huggingface.co/speechbrain/vad-crdnn-libriparty),
trained on LibriParty. Classifies 40-mel log-Fbank features (CNN → RNN → DNN → sigmoid)
into a per-frame posterior at 10 ms resolution. The mel filter matrix and window are
bundled in the wheel. Trained on overlapping cocktail-party speech; on clean
single-speaker audio it labels low-level ambient sound as active, so the default
`threshold` is 0.7.

## `ten`

[TEN VAD](https://github.com/TEN-framework/ten-vad). The ONNX graph consumes a
`[B, 3, 41]` mel+pitch feature tensor and four recurrent states, fed every 256-sample
hop. The frontend is computed in numpy: preemphasis (0.97), Hann-768 STFT zero-padded to
1024, power spectrum, 40-band HTK-triangular mel with `log(sum(power·filter)/32768² +
1e-20)`, per-feature (mean, std) normalization, and a pitch estimate as the 41st feature.
The mel branch reproduces the upstream C implementation exactly; the pitch feature uses
an autocorrelation estimate in place of TEN's native pitch tracker. The mel mean/std and
STFT window are bundled in the wheel.

## Choosing

| need | model |
|------|-------|
| offline default, fast | `silero` |
| multilingual | `marblenet` |
| diarization-aligned | `pyannote` |
| FunASR-compatible pipeline | `fsmn` |
| smallest footprint | `marblenet-int8` / `fsmn-quant` |

Measured comparisons across datasets are in [the benchmark report](../benchmark/results/REPORT.md).
