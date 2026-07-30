# Model conversion

Conversion is a maintainer task: fetch or export each upstream VAD, verify it, and
publish the ONNX to the [`TigreGotico`](https://huggingface.co/TigreGotico) HF org.
Runtime users never need any of this, because models download on demand.

## Install the conversion extras

```bash
uv pip install "vadonnx[convert]"              # silero, ten, fsmn, pyannote (light)
uv pip install "vadonnx[convert-marblenet]"    # + nemo_toolkit[asr] + torch (heavy)
uv pip install "vadonnx[convert-speechbrain]"  # + speechbrain + torch (heavy)
```

## Convert locally (no upload)

```bash
python scripts/convert_all.py                 # all
python scripts/convert_all.py silero fsmn     # selected
```

Each converter downloads/exports the model, runs `verify_onnx()` (loads the graph,
checks the declared `IOSignature`, runs a dummy inference), writes a `signature.json`
and a license-compliant model card, and for FSMN/MarbleNet bakes the feature assets
(`fsmn_cmvn.npz`, `marblenet_mel_fb.npy`, `marblenet_window.npy`) into `vadonnx/data/`.

## Publish to HuggingFace

```bash
huggingface-cli login        # once
python scripts/publish.py    # or: python scripts/publish.py marblenet
```

The script prints the resulting commit SHAs. Pin them into the matching
`ModelSpec.revision` fields in `vadonnx/registry.py` so downloads are reproducible.

## Per-model notes

| model | how | int8 |
|-------|-----|------|
| `silero` | download official ONNX, verify | n/a (already tiny) |
| `ten` | download official ONNX, discover signature | n/a |
| `fsmn` | download `model.onnx` + `model_quant.onnx` + `am.mvn`, bake CMVN | `model_quant.onnx` |
| `marblenet` | NeMo `EncDecFrameClassificationModel.export()`, bundle mel fb+window | `marblenet_int8.onnx` (dynamic) |
| `speechbrain` | export CRDNN graph from SpeechBrain, bundle mel matrix+window | n/a |
| `pyannote` | mirror the community ONNX (fp32 + int8) | `model_int8.onnx` |

**Why MarbleNet ships a features input and a numpy frontend:** NeMo's preprocessor uses
`torch.stft`, which neither ONNX exporter (dynamo or TorchScript) can emit. So the
features-to-logits graph is exported, and the mel frontend is reproduced in numpy using
NeMo's exact filterbank and window (bundled), giving end-to-end MAE of about 4e-4 against NeMo.

---
[← Plugins](plugins.md) · [Home](README.md) · [Licensing →](licensing.md)
