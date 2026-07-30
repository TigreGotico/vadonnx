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
