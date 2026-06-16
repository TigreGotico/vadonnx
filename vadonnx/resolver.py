"""Resolve a model reference to a local ``.onnx`` file path.

Resolution order used by :func:`vadonnx.api.load_vad`:

1. an explicit local ``.onnx`` path (used as-is);
2. an ``http(s)://`` URL (downloaded once and cached);
3. a registry name -> bundled file in the wheel, else a HuggingFace download pinned
   by revision.

Downloads are cached under ``$XDG_DATA_HOME/vadonnx`` (``~/.local/share/vadonnx``).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from .registry import ModelSpec
from .signature import IOSignature


def xdg_data_home() -> str:
    return os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    path = cache_dir or os.path.join(xdg_data_home(), "vadonnx")
    os.makedirs(path, exist_ok=True)
    return path


def data_dir() -> str:
    """Directory of models bundled inside the wheel (``vadonnx/data``)."""
    return os.path.join(os.path.dirname(__file__), "data")


def bundled_path(filename: str) -> Optional[str]:
    path = os.path.join(data_dir(), filename)
    return path if os.path.isfile(path) else None


def is_onnx_path(name: str) -> bool:
    return name.lower().endswith(".onnx")


def is_url(name: str) -> bool:
    return name.startswith("http://") or name.startswith("https://")


def download_url(url: str, cache_dir: Optional[str] = None) -> str:
    """Download an ONNX file from a URL into the cache (skipped if already present)."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    folder = os.path.join(get_cache_dir(cache_dir), "url", digest)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, os.path.basename(url.split("?")[0]) or "model.onnx")
    if not os.path.isfile(dest):
        import urllib.request

        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def hf_download(
    repo: str, filename: str, revision: Optional[str], cache_dir: Optional[str] = None
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        cache_dir=os.path.join(get_cache_dir(cache_dir), "hf"),
    )


def resolve_spec_file(spec: ModelSpec, revision: Optional[str], cache_dir: Optional[str]) -> str:
    """Return a local path for a registry spec (bundled file preferred, else HF)."""
    if spec.bundled:
        local = bundled_path(spec.bundled)
        if local:
            return local
    if not spec.hf_repo:
        raise FileNotFoundError(
            f"model {spec.name!r} has no bundled file and no hf_repo to download from"
        )
    return hf_download(spec.hf_repo, spec.filename, revision or spec.revision, cache_dir)


def sidecar_signature(model_path: str) -> Optional[IOSignature]:
    """Load ``<model>.signature.json`` (or ``signature.json``) next to a model file."""
    base, _ = os.path.splitext(model_path)
    for cand in (base + ".signature.json", os.path.join(os.path.dirname(model_path), "signature.json")):
        if os.path.isfile(cand):
            return IOSignature.from_json(cand)
    return None
