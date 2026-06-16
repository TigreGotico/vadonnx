"""Public entry points: :func:`load_vad`, :func:`list_models`, :func:`register_model`."""
from __future__ import annotations

from typing import List, Optional, Union

from .base import VADModel
from .registry import (
    ModelSpec,
    get_backend,
    get_spec,
    list_models,
    register_model,
)
from .resolver import (
    download_url,
    is_onnx_path,
    is_url,
    resolve_spec_file,
    sidecar_signature,
)
from .signature import IOSignature, coerce_signature

__all__ = ["load_vad", "list_models", "register_model"]


def load_vad(
    name: str = "silero",
    *,
    signature: Union[IOSignature, dict, None] = None,
    threshold: Optional[float] = None,
    neg_threshold: Optional[float] = None,
    providers: Optional[List[str]] = None,
    intra_threads: int = 1,
    inter_threads: int = 1,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **backend_kwargs,
) -> VADModel:
    """Load a VAD model behind the unified :class:`~vadonnx.base.VADModel` API.

    ``name`` may be a built-in/registered/plugin model name (e.g. ``"silero"``,
    ``"ten"``, ``"fsmn"``, ``"marblenet"``), a local ``.onnx`` path, or an
    ``http(s)://`` URL. For an arbitrary custom ONNX file, pass ``signature`` (an
    :class:`IOSignature` or dict) describing its IO; if omitted, a
    ``<model>.signature.json`` sidecar is used when present.

    Args:
        name: model name, ``.onnx`` path, or URL.
        signature: required for custom ONNX without a sidecar; ignored for known models
            unless you want to override the registry signature.
        threshold: activation threshold for speech. When omitted, each backend's own
            default applies (0.5 for most models, 0.7 for ``speechbrain``).
        neg_threshold: deactivation threshold (defaults to ``threshold - 0.15``).
        providers: ONNX Runtime execution providers (default CPU).
        intra_threads / inter_threads: ORT threading (default 1 each, low latency).
        revision: pin a HuggingFace revision (overrides the registry default).
        cache_dir: override the download cache directory.
        **backend_kwargs: forwarded to the backend constructor.

    Returns:
        A ready-to-use :class:`~vadonnx.base.VADModel`.
    """
    sig = coerce_signature(signature)

    # common backend kwargs; pass threshold only when given so each backend keeps its
    # own default otherwise.
    common = dict(providers=providers, intra_threads=intra_threads,
                  inter_threads=inter_threads, neg_threshold=neg_threshold,
                  **backend_kwargs)
    if threshold is not None:
        common["threshold"] = threshold

    # 1. explicit local .onnx file
    if is_onnx_path(name) and not is_url(name):
        import os

        if not os.path.isfile(name):
            raise FileNotFoundError(f"no such ONNX file: {name}")
        sig = sig or sidecar_signature(name)
        if sig is None:
            raise ValueError(
                f"loading a raw .onnx ({name}) requires a `signature` "
                "or a <model>.signature.json sidecar"
            )
        from .onnx_backend import OnnxVAD

        return OnnxVAD(name, sig, **common)

    # 2. URL
    if is_url(name):
        path = download_url(name, cache_dir)
        sig = sig or sidecar_signature(path)
        if sig is None:
            raise ValueError("loading a URL model requires a `signature`")
        from .onnx_backend import OnnxVAD

        return OnnxVAD(path, sig, **common)

    # 3. registry name
    spec: Optional[ModelSpec] = get_spec(name)
    if spec is None:
        raise KeyError(
            f"unknown model {name!r}; known: {list_models()} "
            "(or pass a .onnx path / URL with a signature)"
        )
    path = resolve_spec_file(spec, revision, cache_dir)
    sig = sig or spec.signature
    backend_cls = get_backend(spec.backend)
    return backend_cls(path, sig, **common)
