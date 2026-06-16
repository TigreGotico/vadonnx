"""Shared helpers for the conversion scripts: download, verify, card, publish."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..signature import IOSignature


def download(url: str, dest: str) -> str:
    """Download ``url`` to ``dest`` (skipped if it already exists)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if not os.path.isfile(dest):
        import urllib.request

        print(f"  downloading {url}")
        with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    return dest


def _dummy_for(inp, frame_size: int):
    """Build a plausible dummy tensor for an ORT input descriptor."""
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    # replace a dynamic time axis with frame_size where it looks like audio
    if "float" in inp.type:
        return (np.random.randn(*shape).astype(np.float32) * 0.01)
    if "int64" in inp.type:
        return np.zeros(shape, dtype=np.int64) if shape else np.array(0, np.int64)
    return np.zeros(shape, dtype=np.float32)


def verify_onnx(path: str, signature: IOSignature, sample_rate: Optional[int] = None) -> dict:
    """Load the model, run a dummy inference, and sanity-check the declared signature.

    Returns a dict describing the discovered inputs/outputs. Raises ``AssertionError``
    if the declared audio input / state / output names are not present.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    assert signature.audio_input in in_names, (
        f"audio_input {signature.audio_input!r} not in model inputs {in_names}"
    )
    for s in signature.state_inputs:
        assert s in in_names, f"state input {s!r} not in {in_names}"
    for e in signature.extra_inputs:
        assert e in in_names, f"extra input {e!r} not in {in_names}"
    if isinstance(signature.prob_output, str):
        assert signature.prob_output in out_names, (
            f"prob_output {signature.prob_output!r} not in outputs {out_names}"
        )
    for out_name in signature.state_output_map.values():
        assert out_name in out_names, f"state output {out_name!r} not in {out_names}"

    return {
        "inputs": [(i.name, i.shape, i.type) for i in sess.get_inputs()],
        "outputs": [(o.name, o.shape, o.type) for o in sess.get_outputs()],
    }


def write_signature(signature: IOSignature, model_path: str) -> str:
    """Write ``<model>.signature.json`` next to the model file."""
    base, _ = os.path.splitext(model_path)
    out = base + ".signature.json"
    signature.save_json(out)
    return out


_SPDX = {"mit": "mit", "apache-2.0": "apache-2.0", "apache 2.0": "apache-2.0",
         "bsd": "bsd", "bsd-2-clause": "bsd-2-clause", "bsd-3-clause": "bsd-3-clause"}


def make_card(name: str, upstream: str, license_name: str, signature: IOSignature) -> str:
    """Return Markdown for a HuggingFace model card.

    Maps the license to a valid HF tag: known SPDX ids pass through, anything else
    (e.g. the NVIDIA Open Model License) becomes ``license: other`` + ``license_name``.
    """
    key = license_name.strip().lower()
    if key in _SPDX:
        license_yaml = f"license: {_SPDX[key]}"
    else:
        slug = key.replace(" ", "-")
        license_yaml = f"license: other\nlicense_name: {slug}"
    return f"""---
{license_yaml}
library_name: vadonnx
tags:
  - voice-activity-detection
  - vad
  - onnx
---

# {name} (ONNX, repackaged for vadonnx)

ONNX voice activity detection model packaged for use with
[`vadonnx`](https://github.com/TigreGotico/vadonnx).

- **Upstream source:** {upstream}
- **License:** {license_name}
- **Sample rate:** {signature.sample_rate} Hz
- **Frame size:** {signature.frame_size} samples
- **Stateful:** {signature.stateful}

This repository redistributes the model in ONNX form together with a
`signature.json` describing its input/output wiring. All rights and the original
license belong to the upstream authors.

## Usage

```python
from vadonnx import load_vad
vad = load_vad("{name.split('-')[0]}")
segments = vad.get_speech_segments(audio, sample_rate={signature.sample_rate})
```
"""


def push_to_hf(
    repo_id: str,
    files: dict,
    *,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Create ``repo_id`` if needed and upload ``files`` ({path_in_repo: local_path}).

    Returns the resulting commit sha.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    for path_in_repo, local in files.items():
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
    info = api.repo_info(repo_id, repo_type="model")
    return info.sha
