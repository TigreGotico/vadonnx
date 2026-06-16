"""Fetch and package FunASR FSMN-VAD ONNX (fp32 + int8) for publishing.

FSMN-VAD (MIT) ships an official ONNX via FunASR/ModelScope. We fetch ``model.onnx``
and ``model_quant.onnx`` plus the ``am.mvn`` CMVN statistics, bake the CMVN into the
bundled ``fsmn_cmvn.npz`` (used by the runtime backend), write a card, and publish.
The fbank+LFR+CMVN frontend lives in :mod:`vadonnx.backends.fsmn`. No heavy deps.
"""
from __future__ import annotations

import os

import numpy as np

from ..registry import BUILTIN
from . import common

HF = "https://huggingface.co/funasr/fsmn-vad-onnx/resolve/main"
MS = "https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx/resolve/master"
UPSTREAM = "https://huggingface.co/funasr/fsmn-vad-onnx (MIT)"
REPO = "TigreGotico/fsmn-vad-onnx"


def parse_cmvn(path: str):
    lines = open(path, encoding="utf-8").readlines()
    means, vars = [], []
    for i, line in enumerate(lines):
        it = line.split()
        if it and it[0] == "<AddShift>":
            nx = lines[i + 1].split()
            if nx[0] == "<LearnRateCoef>":
                means = nx[3 : len(nx) - 1]
        elif it and it[0] == "<Rescale>":
            nx = lines[i + 1].split()
            if nx[0] == "<LearnRateCoef>":
                vars = nx[3 : len(nx) - 1]
    return np.array(means, np.float32), np.array(vars, np.float32)


def convert(out_dir: str = "scratch/fsmn") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model = common.download(f"{HF}/model.onnx", os.path.join(out_dir, "model.onnx"))
    quant = common.download(f"{HF}/model_quant.onnx", os.path.join(out_dir, "model_quant.onnx"))
    mvn = common.download(f"{MS}/am.mvn", os.path.join(out_dir, "am.mvn"))

    # bake CMVN into the bundled npz consumed by the runtime backend
    means, vars = parse_cmvn(mvn)
    data_npz = os.path.join(os.path.dirname(__file__), "..", "data", "fsmn_cmvn.npz")
    np.savez(data_npz, means=means, vars=vars)
    print(f"wrote {os.path.normpath(data_npz)} (cmvn dim {means.shape[0]})")

    sig = BUILTIN["fsmn"].signature
    card = common.make_card("fsmn-vad", UPSTREAM, "MIT", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card)
    sig.save_json(os.path.join(out_dir, "fsmn.signature.json"))
    return {"model": model, "quant": quant, "mvn": mvn, "card": card_path, "repo": REPO}


def publish(out_dir: str = "scratch/fsmn", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {
            "model.onnx": files["model"],
            "model_quant.onnx": files["quant"],
            "am.mvn": files["mvn"],
            "fsmn.signature.json": os.path.join(out_dir, "fsmn.signature.json"),
            "README.md": files["card"],
        },
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
