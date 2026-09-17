# Licensing

The `vadonnx` library code is **Apache-2.0**. Model weights downloaded or bundled keep
their upstream licenses. Your obligations depend on which model you load.

| model | license | redistribution |
|-------|---------|----------------|
| `silero` / `silero-8k` / `silero-op15` | MIT | freely redistributable (bundled in the wheel) |
| `ten` | Apache-2.0 | freely redistributable |
| `fsmn` / `fsmn-quant` | MIT (FunASR) | freely redistributable |
| `speechbrain` | Apache-2.0 (SpeechBrain) | freely redistributable |
| `pyannote` / `pyannote-int8` | MIT | community ONNX of a gated upstream, see below |
| `marblenet` / `marblenet-int8` | **NVIDIA Open Model License** | see below |
| `pulsevad` / `pulsevad-fp32` / `pulsevad-81k` | MIT | freely redistributable, see below |

## Silero / FSMN (MIT), TEN / SpeechBrain (Apache-2.0)

Permissive. The ONNX files are mirrored on the `TigreGotico` HF org with model cards
attributing the upstream source and license.

## pyannote (MIT, gated upstream)

`pyannote` uses the community ONNX (onnx-community, MIT), itself an export of
`pyannote/segmentation-3.0` (MIT but **gated**: upstream requires accepting conditions
and providing contact info). The weights are MIT-licensed. The gate is only an access
formality. The re-hosted card attributes both upstream sources.

## MarbleNet: NVIDIA Open Model License

`marblenet` derives from
[nvidia/frame_vad_multilingual_marblenet_v2.0](https://huggingface.co/nvidia/frame_vad_multilingual_marblenet_v2.0),
distributed under the **NVIDIA Open Model License**. Commercial use is permitted, but
you are responsible for complying with that license, including its attribution terms.
The mirrored repo's model card reproduces the license and links upstream. Review the
license before redistributing the model or using it commercially.

## PulseVAD (MIT)

The PulseVAD weights are MIT-licensed, Copyright (c) 2026 Aydin Adnan. `vadonnx` does not
bundle or mirror them. It downloads them from the upstream repository at a pinned
commit. If you redistribute the files, include the upstream
[LICENSE](https://github.com/AydinAdnan/PulseVAD/blob/af25e79d66830a3fee74541812721f6158fc92b5/LICENSE).

The upstream
[ATTRIBUTION.md](https://github.com/AydinAdnan/PulseVAD/blob/af25e79d66830a3fee74541812721f6158fc92b5/ATTRIBUTION.md)
lists the training data and tools:

- LibriSpeech train-clean-100 (CC BY 4.0), Vassil Panayotov, Guoguo Chen, Daniel Povey,
  Sanjeev Khudanpur.
- Common Voice (CC0), Mozilla.
- Multilingual LibriSpeech (CC BY 4.0).
- VoxLingua107 (CC BY 4.0).
- MUSAN noise and music (CC BY 4.0), David Snyder, Guoguo Chen, Daniel Povey.
- DNS Challenge noise, Interspeech 2020 (CC BY 4.0 / CC0 subset), Chandan K. A. Reddy et al.
- Synthetic wind noise (Mirabilii 2022 algorithm), synthesized locally.
- Silero VAD v5/v6 (MIT), used only to label the training audio.

Upstream states that it did not use the Silero labeled dataset, LibriVAD or the kiloVAD
checkpoints, which have non-commercial licenses.

## Bundled assets

The wheel bundles only:
- `silero_vad.onnx` (MIT): the offline default model.
- `fsmn_cmvn.npz`: CMVN statistics derived from FunASR (MIT).
- `marblenet_mel_fb.npy` / `marblenet_window.npy`: the mel filterbank and window extracted
  from the NeMo MarbleNet preprocessor (NVIDIA Open Model License).

No model weights other than Silero ship in the wheel. Everything else is fetched
on demand from HuggingFace.

---
[← Conversion](conversion.md) · [Home](README.md) · [API reference →](api.md)
