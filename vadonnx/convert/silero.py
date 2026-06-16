"""Fetch, verify and package the Silero VAD ONNX for publishing.

Silero ships an official ONNX (MIT). We download it, verify it matches the registry
signature, write a signature sidecar + model card, and optionally publish to the
TigreGotico HF org. No heavy dependencies.
"""
from __future__ import annotations

import os

from ..registry import BUILTIN
from . import common

SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
UPSTREAM = "https://github.com/snakers4/silero-vad (MIT)"
REPO = "TigreGotico/silero-vad-onnx"


def convert(out_dir: str = "scratch/silero") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model = os.path.join(out_dir, "silero_vad.onnx")
    common.download(SILERO_URL, model)

    sig = BUILTIN["silero"].signature
    info = common.verify_onnx(model, sig)
    print("verified silero:", info["inputs"], "->", info["outputs"])

    sig_path = common.write_signature(sig, model)
    card = common.make_card("silero-vad", UPSTREAM, "MIT", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)
    return {"model": model, "signature": sig_path, "card": card_path, "repo": REPO}


def publish(out_dir: str = "scratch/silero", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {
            "silero_vad.onnx": files["model"],
            "silero_vad.signature.json": files["signature"],
            "README.md": files["card"],
        },
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
