# Plugins

Third-party packages can add models and backends to `vadonnx` without forking, via
Python entry points. They are discovered lazily the first time you call `list_models()`
or load an unknown model.

## Register a model

Expose a `ModelSpec` (or a zero-arg callable returning one) in the `vadonnx.models`
group:

```toml
# in your package's pyproject.toml
[project.entry-points."vadonnx.models"]
my-vad = "my_pkg.registry:SPEC"
```

```python
# my_pkg/registry.py
from vadonnx import ModelSpec, IOSignature
SPEC = ModelSpec(
    name="my-vad",
    signature=IOSignature(sample_rate=16000, frame_size=512, audio_layout="BT"),
    hf_repo="me/my-vad-onnx", filename="model.onnx",
)
```

## Register a backend

If your model needs custom Python glue (beyond a declarative signature), expose a
`VADModel` subclass in the `vadonnx.backends` group and reference it from a spec's
`backend` field:

```toml
[project.entry-points."vadonnx.backends"]
mybackend = "my_pkg.backend:MyVAD"
```

```python
# spec referencing it
ModelSpec(name="my-vad", backend="mybackend", signature=..., hf_repo=...)
```

The `backend` field also accepts a dotted path (`"my_pkg.backend:MyVAD"`) directly.

## Precedence

Built-in models win over registered ones, which win over plugin-provided ones, on a
name collision. `list_models()` includes everything discovered.
