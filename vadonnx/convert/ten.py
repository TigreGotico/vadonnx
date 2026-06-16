"""Fetch, verify and package the TEN VAD ONNX for publishing.

TEN VAD (Apache-2.0) ships an official ONNX in its repo. The graph expects precomputed
features (a `[B, 3, 41]` mel+pitch tensor) plus four recurrent state tensors — feature
extraction lives in TEN's native C library, which `vadonnx` does not reproduce. This
converter downloads, inspects and republishes the ONNX with a signature describing its
true IO so it is available for experimentation; see docs/backends.md for the caveat.
"""
from __future__ import annotations

import os

from ..signature import IOSignature
from . import common

TEN_URL = "https://github.com/TEN-framework/ten-vad/raw/main/src/onnx_model/ten-vad.onnx"
UPSTREAM = "https://github.com/TEN-framework/ten-vad (Apache-2.0)"
REPO = "TigreGotico/ten-vad-onnx"


def discover_signature(model_path: str) -> IOSignature:
    """Inspect the graph and build a signature reflecting its real inputs/outputs."""
    import onnxruntime as ort

    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in s.get_inputs()]
    outs = [o.name for o in s.get_outputs()]
    # feature input is the first; remaining float inputs are recurrent state.
    feat_name, feat_shape = ins[0]
    feat_dim = int(feat_shape[-1]) if isinstance(feat_shape[-1], int) else 41
    state_inputs = {n: tuple(d if isinstance(d, int) else 1 for d in shp) for n, shp in ins[1:]}
    state_map = {n: outs[i + 1] for i, n in enumerate(state_inputs) if i + 1 < len(outs)}
    return IOSignature(
        sample_rate=16000, frame_size=256, stateful=True,
        feature="logmel", feature_params={"n_mels": feat_dim, "hop": 256, "n_fft": 1024},
        audio_input=feat_name, audio_layout="BFT",
        state_inputs=state_inputs, state_output_map=state_map,
        prob_output=0, prob_extract="last", license="Apache-2.0",
    )


def convert(out_dir: str = "scratch/ten") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model = os.path.join(out_dir, "ten-vad.onnx")
    common.download(TEN_URL, model)
    sig = discover_signature(model)
    info = common.verify_onnx(model, sig)
    print("verified ten:", info["inputs"], "->", info["outputs"])
    sig_path = common.write_signature(sig, model)
    card = common.make_card("ten-vad", UPSTREAM, "Apache-2.0", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card + "\n> **Note:** feature extraction uses TEN's native library; "
                "the pure-ONNX path in vadonnx is experimental.\n")
    return {"model": model, "signature": sig_path, "card": card_path, "repo": REPO}


def publish(out_dir: str = "scratch/ten", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {"ten-vad.onnx": files["model"], "ten-vad.signature.json": files["signature"],
         "README.md": files["card"]},
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
