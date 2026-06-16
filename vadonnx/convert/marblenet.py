"""Export NVIDIA NeMo Frame-VAD MarbleNet to ONNX (fp32 + int8) and package it.

Requires the ``convert-marblenet`` extra (``nemo_toolkit[asr]`` + ``torch``); these are
imported lazily. NeMo's STFT preprocessor cannot be folded into ONNX (`torch.stft` is
not ONNX-exportable), so we export the features→logits graph and extract NeMo's mel
filterbank + window into the bundled ``marblenet_mel_fb.npy`` / ``marblenet_window.npy``
that the runtime :mod:`vadonnx.backends.marblenet` numpy frontend reproduces.
"""
from __future__ import annotations

import os

import numpy as np

from ..registry import BUILTIN
from . import common

MODEL = "nvidia/frame_vad_multilingual_marblenet_v2.0"
UPSTREAM = f"https://huggingface.co/{MODEL} (NVIDIA Open Model License)"
REPO = "TigreGotico/frame-vad-marblenet-onnx"


def _load():
    import nemo.collections.asr as nemo_asr

    for strict in (True, False):
        try:
            m = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                MODEL, strict=strict
            )
            return m.to("cpu").eval()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not load {MODEL}: {last}")


def convert(out_dir: str = "scratch/marble") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    m = _load()

    # export features -> logits graph
    model = os.path.join(out_dir, "marblenet.onnx")
    m.export(model)

    # bundle NeMo's mel filterbank + window for the numpy frontend
    f = m.preprocessor.featurizer
    data = os.path.join(os.path.dirname(__file__), "..", "data")
    np.save(os.path.join(data, "marblenet_mel_fb.npy"), np.squeeze(f.fb.numpy()).astype(np.float32))
    np.save(os.path.join(data, "marblenet_window.npy"), f.window.numpy().astype(np.float32))

    # int8 (≈36% size, MAE ~7e-3 vs fp32 — worth shipping)
    quant = os.path.join(out_dir, "marblenet_int8.onnx")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model, quant, weight_type=QuantType.QInt8)

    sig = BUILTIN["marblenet"].signature
    card = common.make_card("frame-vad-marblenet", UPSTREAM, "NVIDIA Open Model License", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as fh:
        fh.write(card)
    sig.save_json(os.path.join(out_dir, "marblenet.signature.json"))
    return {"model": model, "quant": quant, "card": card_path, "repo": REPO}


def publish(out_dir: str = "scratch/marble", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {
            "marblenet.onnx": files["model"],
            "marblenet_int8.onnx": files["quant"],
            "marblenet.signature.json": os.path.join(out_dir, "marblenet.signature.json"),
            "README.md": files["card"],
        },
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
