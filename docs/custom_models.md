# Custom models & `IOSignature`

`vadonnx` can drive almost any VAD ONNX graph without new code. You describe the
graph's inputs/outputs with an `IOSignature`, and the generic engine handles framing,
state, features and probability extraction.

## Loading a custom ONNX

```python
from vadonnx import load_vad, IOSignature

vad = load_vad("/path/to/my_vad.onnx", signature=IOSignature(
    sample_rate=16000,
    frame_size=512,
    audio_input="input",
    audio_layout="BT",          # (1, frame_size)
    prob_output="output",
    prob_extract="scalar",
))
```

A URL works too (`load_vad("https://.../model.onnx", signature=...)`), cached on first
use. If a `<model>.signature.json` sidecar sits next to the file, the signature is
loaded automatically and you can omit it.

## `IOSignature` fields

| field | meaning |
|-------|---------|
| `sample_rate` | rate the model expects (Hz) |
| `frame_size` | audio samples consumed per step (the hop) |
| `context_size` | trailing samples of the previous frame to prepend (Silero uses 64) |
| `stateful` | whether the model carries recurrent state |
| `feature` | `None` (raw PCM) / `"logmel"` / `"fbank"` — runs `vadonnx.features` first |
| `feature_params` | kwargs for the feature extractor |
| `audio_input` | name of the audio/feature input tensor |
| `audio_layout` | `"BT"` `(1,T)`, `"T"` `(T,)`, `"BFT"` `(1,feat,frames)`, `"BTF"` `(1,frames,feat)` |
| `state_inputs` | `{input_name: shape}` recurrent-state tensors (zeroed on reset) |
| `extra_inputs` | `{name: (kind, value)}` constant inputs, e.g. `{"sr": ("int64_scalar", 16000)}` |
| `state_output_map` | `{state_input: output_name}` next-state wiring |
| `prob_output` | output name or index holding the speech score |
| `prob_extract` | `"scalar"`, `"last"`, `"mean"`, `"index:N"`, `"1-minus:N"` |
| `multiclass_collapse` | class indices that mean *speech* (summed) |

## Examples

**Stateful, with a state tensor and a constant input (Silero-style):**

```python
IOSignature(
    sample_rate=16000, frame_size=512, context_size=64, stateful=True,
    state_inputs={"state": (2, 1, 128)},
    extra_inputs={"sr": ("int64_scalar", 16000)},
    state_output_map={"state": "stateN"},
    prob_output="output", prob_extract="scalar",
)
```

**Multi-class output where class 0 is non-speech:**

```python
IOSignature(sample_rate=16000, frame_size=400, feature="fbank",
            audio_layout="BTF", prob_output="logits", prob_extract="1-minus:0")
```

## Registering a custom model by name

```python
from vadonnx import register_model, ModelSpec, IOSignature
register_model(ModelSpec(
    name="my-vad",
    signature=IOSignature(sample_rate=16000, frame_size=512, audio_layout="BT"),
    hf_repo="me/my-vad-onnx", filename="model.onnx",
))
vad = load_vad("my-vad")
```
