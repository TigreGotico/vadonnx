"""Mirror the community pyannote segmentation-3.0 ONNX to the TigreGotico org.

Source: onnx-community/pyannote-segmentation-3.0 (MIT), itself an ONNX export of
pyannote/segmentation-3.0 (MIT, gated). We re-host the fp32 + int8 graphs with a
signature sidecar and an attributing model card, pinned by revision. Light deps only.
"""
from __future__ import annotations

import os

from ..registry import BUILTIN
from . import common

SRC = "onnx-community/pyannote-segmentation-3.0"
UPSTREAM = (f"https://huggingface.co/{SRC} (MIT) — export of "
            "pyannote/segmentation-3.0 (MIT, gated)")
REPO = "TigreGotico/pyannote-segmentation-3.0-onnx"


def convert(out_dir: str = "scratch/pyannote") -> dict:
    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    model = hf_hub_download(SRC, "onnx/model.onnx", cache_dir=os.path.join(out_dir, "hf"))
    quant = hf_hub_download(SRC, "onnx/model_int8.onnx", cache_dir=os.path.join(out_dir, "hf"))

    sig = BUILTIN["pyannote"].signature
    common.verify_onnx(model, sig)
    sig_path = os.path.join(out_dir, "pyannote.signature.json")
    sig.save_json(sig_path)
    card = common.make_card("pyannote-segmentation-3.0", UPSTREAM, "MIT", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card + "\n> Powerset segmentation; VAD = 1 - P(non-speech class 0).\n")
    return {"model": model, "quant": quant, "signature": sig_path,
            "card": card_path, "repo": REPO}


def publish(out_dir: str = "scratch/pyannote", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {"model.onnx": files["model"], "model_int8.onnx": files["quant"],
         "pyannote.signature.json": files["signature"], "README.md": files["card"]},
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
