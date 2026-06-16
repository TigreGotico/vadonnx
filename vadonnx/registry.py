"""Built-in model registry, programmatic registration and plugin discovery.

A :class:`ModelSpec` ties a friendly name (``"silero"``) to where its ONNX file lives
(bundled in the wheel, or a file on the ``TigreGotico`` HuggingFace org pinned by
revision) and to the :class:`~vadonnx.signature.IOSignature` that drives inference.
Third parties can add models/backends without forking via entry points in the
``vadonnx.models`` / ``vadonnx.backends`` groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .signature import IOSignature


@dataclass
class ModelSpec:
    name: str
    signature: IOSignature
    backend: str = "onnx"            # backend key (see BACKENDS) or dotted import path
    hf_repo: Optional[str] = None    # e.g. "TigreGotico/silero-vad-onnx"
    filename: str = "model.onnx"
    revision: Optional[str] = None   # pinned commit sha for reproducibility
    bundled: Optional[str] = None    # filename under vadonnx/data/ shipped in the wheel
    license: str = ""
    extras: List[str] = field(default_factory=list)
    description: str = ""


# Silero v5/v6 share one graph for both rates; the `sr` input selects 8k vs 16k and the
# frame size differs (256 @ 8k, 512 @ 16k). Outputs: `output` (B,1) + `stateN` (2,B,128).
_SILERO_KW = dict(
    stateful=True,
    audio_input="input",
    audio_layout="BT",
    state_inputs={"state": (2, 1, 128)},
    state_output_map={"state": "stateN"},
    prob_output="output",
    prob_extract="scalar",
    license="MIT",
)

BUILTIN: Dict[str, ModelSpec] = {
    "silero": ModelSpec(
        name="silero",
        signature=IOSignature(
            sample_rate=16000, frame_size=512, context_size=64,
            extra_inputs={"sr": ("int64_scalar", 16000)}, **_SILERO_KW,
        ),
        hf_repo="TigreGotico/silero-vad-onnx",
        filename="silero_vad.onnx",
        revision="fbcc454c02e349042822de4a9c500df0179c080b",
        bundled="silero_vad.onnx",
        license="MIT",
        description="Silero VAD v6 — fast, multilingual, the recommended default.",
    ),
    "silero-op15": ModelSpec(
        name="silero-op15",
        signature=IOSignature(
            sample_rate=16000, frame_size=512, context_size=64,
            extra_inputs={"sr": ("int64_scalar", 16000)}, **_SILERO_KW,
        ),
        hf_repo="TigreGotico/silero-vad-onnx",
        filename="silero_vad_16k_op15.onnx",
        revision="60821c708b09ff913bde53744fc6cc1c2891d133",
        license="MIT",
        description="Silero VAD v6, 16 kHz-only, opset 15 (smaller; older ONNX runtimes).",
    ),
    "silero-8k": ModelSpec(
        name="silero-8k",
        signature=IOSignature(
            sample_rate=8000, frame_size=256, context_size=32,
            extra_inputs={"sr": ("int64_scalar", 8000)}, **_SILERO_KW,
        ),
        hf_repo="TigreGotico/silero-vad-onnx",
        filename="silero_vad.onnx",
        revision="fbcc454c02e349042822de4a9c500df0179c080b",
        bundled="silero_vad.onnx",
        license="MIT",
        description="Silero VAD v6 driven at an 8 kHz sample rate.",
    ),
    "ten": ModelSpec(
        name="ten",
        signature=IOSignature(
            sample_rate=16000, frame_size=256, stateful=True,
            audio_input="input", audio_layout="BT",
            prob_output=0, prob_extract="scalar", license="Apache-2.0",
        ),
        backend="ten",
        hf_repo="TigreGotico/ten-vad-onnx",
        filename="ten-vad.onnx",
        revision="225df9e0b79788bb5ee037e62e7fb22f7993c882",
        license="Apache-2.0",
        description="TEN VAD — experimental/reference only (needs native feature extractor).",
    ),
    "fsmn": ModelSpec(
        name="fsmn",
        signature=IOSignature(
            sample_rate=16000, frame_size=160, stateful=True,
            feature="fbank", audio_input="speech", audio_layout="BTF",
            prob_output="logits", prob_extract="1-minus:0", license="MIT",
        ),
        backend="fsmn",
        hf_repo="TigreGotico/fsmn-vad-onnx",
        filename="model.onnx",
        revision="2216bade214cd80e818bce2c823c878dba117898",
        license="MIT",
        extras=["kaldi-native-fbank"],
        description="FunASR FSMN-VAD (fbank+LFR+CMVN frontend; best-effort parity).",
    ),
    "fsmn-quant": ModelSpec(
        name="fsmn-quant",
        signature=IOSignature(
            sample_rate=16000, frame_size=160, stateful=True,
            feature="fbank", audio_input="speech", audio_layout="BTF",
            prob_output="logits", prob_extract="1-minus:0", license="MIT",
        ),
        backend="fsmn",
        hf_repo="TigreGotico/fsmn-vad-onnx",
        filename="model_quant.onnx",
        revision="2216bade214cd80e818bce2c823c878dba117898",
        license="MIT",
        extras=["kaldi-native-fbank"],
        description="Quantized (int8) FunASR FSMN-VAD.",
    ),
    "marblenet": ModelSpec(
        name="marblenet",
        signature=IOSignature(
            sample_rate=16000, frame_size=320, stateful=False,
            feature="logmel", audio_input="audio_signal", audio_layout="BFT",
            prob_output=0, prob_extract="index:1", license="NVIDIA-OpenModelLicense",
        ),
        backend="marblenet",
        hf_repo="TigreGotico/frame-vad-marblenet-onnx",
        filename="marblenet.onnx",
        revision="e8786fe74e055954901eb553cc9c3145323981cc",
        license="NVIDIA-OpenModelLicense",
        description="NVIDIA NeMo Frame-VAD MarbleNet (numpy mel frontend, ~parity).",
    ),
    "marblenet-int8": ModelSpec(
        name="marblenet-int8",
        signature=IOSignature(
            sample_rate=16000, frame_size=320, stateful=False,
            feature="logmel", audio_input="audio_signal", audio_layout="BFT",
            prob_output=0, prob_extract="index:1", license="NVIDIA-OpenModelLicense",
        ),
        backend="marblenet",
        hf_repo="TigreGotico/frame-vad-marblenet-onnx",
        filename="marblenet_int8.onnx",
        revision="e8786fe74e055954901eb553cc9c3145323981cc",
        license="NVIDIA-OpenModelLicense",
        description="Quantized (int8) NVIDIA NeMo Frame-VAD MarbleNet.",
    ),
    "speechbrain": ModelSpec(
        name="speechbrain",
        signature=IOSignature(
            sample_rate=16000, frame_size=160, stateful=False,
            feature="logmel", audio_input="feats", audio_layout="BTF",
            prob_output="probs", prob_extract="last", license="Apache-2.0",
        ),
        backend="speechbrain",
        hf_repo="TigreGotico/sb-vad-crdnn-onnx",
        filename="sb_vad_crdnn.onnx",
        revision="61e5b4cbcffb6eac7770292ba84609164ea66b3d",
        license="Apache-2.0",
        description="SpeechBrain CRDNN VAD (first-to-ONNX; LibriParty-tuned, high recall).",
    ),
    "pyannote": ModelSpec(
        name="pyannote",
        signature=IOSignature(
            sample_rate=16000, frame_size=272, stateful=False,
            audio_input="input_values", audio_layout="BT",
            prob_output="logits", prob_extract="1-minus:0", license="MIT",
        ),
        backend="pyannote",
        hf_repo="TigreGotico/pyannote-segmentation-3.0-onnx",
        filename="model.onnx",
        revision="dd19a60106a77a38aa2eea2ab07338ce2e21634f",
        license="MIT",
        description="pyannote segmentation-3.0 VAD (community ONNX; strong, windowed).",
    ),
    "pyannote-int8": ModelSpec(
        name="pyannote-int8",
        signature=IOSignature(
            sample_rate=16000, frame_size=272, stateful=False,
            audio_input="input_values", audio_layout="BT",
            prob_output="logits", prob_extract="1-minus:0", license="MIT",
        ),
        backend="pyannote",
        hf_repo="TigreGotico/pyannote-segmentation-3.0-onnx",
        filename="model_int8.onnx",
        revision="dd19a60106a77a38aa2eea2ab07338ce2e21634f",
        license="MIT",
        description="Quantized (int8) pyannote segmentation-3.0 VAD.",
    ),
}


# backend key -> class. Populated lazily to avoid importing onnxruntime at import time.
def _backends() -> Dict[str, type]:
    from .onnx_backend import OnnxVAD
    from .backends.ten import TenVAD
    from .backends.fsmn import FsmnVAD
    from .backends.marblenet import MarbleVAD
    from .backends.speechbrain import SbVAD
    from .backends.pyannote import PyannoteVAD

    return {"onnx": OnnxVAD, "ten": TenVAD, "fsmn": FsmnVAD,
            "marblenet": MarbleVAD, "speechbrain": SbVAD, "pyannote": PyannoteVAD}


_REGISTERED: Dict[str, ModelSpec] = {}
_PLUGINS_LOADED = False
_PLUGIN_SPECS: Dict[str, ModelSpec] = {}
_PLUGIN_BACKENDS: Dict[str, type] = {}


def register_model(spec: ModelSpec) -> None:
    """Register a :class:`ModelSpec` at runtime (highest precedence after built-ins)."""
    _REGISTERED[spec.name] = spec


def discover_plugins() -> None:
    """Load models/backends advertised by other installed packages (idempotent)."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover
        return
    for ep in _iter_eps(entry_points, "vadonnx.models"):
        try:
            obj = ep.load()
            spec = obj() if callable(obj) and not isinstance(obj, ModelSpec) else obj
            if isinstance(spec, ModelSpec):
                _PLUGIN_SPECS[spec.name] = spec
        except Exception:  # pragma: no cover - third party errors must not break us
            continue
    for ep in _iter_eps(entry_points, "vadonnx.backends"):
        try:
            _PLUGIN_BACKENDS[ep.name] = ep.load()
        except Exception:  # pragma: no cover
            continue


def _iter_eps(entry_points, group):
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - py<3.10 style
        eps = entry_points().get(group, [])
    return list(eps)


def get_spec(name: str) -> Optional[ModelSpec]:
    """Resolve a model name to its spec across built-ins, registered and plugins."""
    if name in _REGISTERED:
        return _REGISTERED[name]
    if name in BUILTIN:
        return BUILTIN[name]
    discover_plugins()
    return _PLUGIN_SPECS.get(name)


def get_backend(key: str) -> type:
    """Resolve a backend key (or dotted ``module:Class`` path) to a class."""
    builtin = _backends()
    if key in builtin:
        return builtin[key]
    discover_plugins()
    if key in _PLUGIN_BACKENDS:
        return _PLUGIN_BACKENDS[key]
    if ":" in key or "." in key:
        from importlib import import_module

        mod, _, cls = key.replace(":", ".").rpartition(".")
        return getattr(import_module(mod), cls)
    raise KeyError(f"unknown backend {key!r}")


def list_models() -> List[str]:
    """Return all known model names (built-in + registered + discovered plugins)."""
    discover_plugins()
    names = set(BUILTIN) | set(_REGISTERED) | set(_PLUGIN_SPECS)
    return sorted(names)
