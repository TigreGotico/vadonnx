"""Export SpeechBrain CRDNN VAD to ONNX (first-to-ONNX) and package it.

Requires the ``convert-speechbrain`` extra (``speechbrain`` + ``torch``), imported
lazily. Exports the features→posterior graph (mean-var-norm + CNN + RNN + DNN + sigmoid)
and bundles SpeechBrain's exact Fbank mel matrix + window into the wheel so the runtime
numpy frontend reproduces it bit-exactly. No official ONNX exists upstream.
"""
from __future__ import annotations

import os

import numpy as np

from ..registry import BUILTIN
from . import common

SOURCE = "speechbrain/vad-crdnn-libriparty"
UPSTREAM = f"https://huggingface.co/{SOURCE} (Apache-2.0)"
REPO = "TigreGotico/sb-vad-crdnn-onnx"


def convert(out_dir: str = "scratch/sb") -> dict:
    import torch
    import torch.nn as nn
    from speechbrain.inference.VAD import VAD

    os.makedirs(out_dir, exist_ok=True)
    vad = VAD.from_hparams(source=SOURCE, savedir=os.path.join(out_dir, "hf"))
    cf = vad.mods["compute_features"]
    fb = cf.compute_fbanks

    # bundle SpeechBrain's exact mel filter matrix + window
    cap = {}
    orig = fb._create_fbank_matrix
    fb._create_fbank_matrix = lambda a, b: cap.setdefault("M", orig(a, b))
    with torch.no_grad():
        feats = cf(torch.zeros(1, 16000))
    data = os.path.join(os.path.dirname(__file__), "..", "data")
    np.save(os.path.join(data, "sb_vad_mel_fb.npy"), cap["M"].numpy().astype(np.float32))
    np.save(os.path.join(data, "sb_vad_window.npy"),
            torch.hamming_window(400, periodic=True).numpy().astype(np.float32))

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mvn, self.cnn, self.rnn, self.dnn = (
                m.mods.mean_var_norm, m.mods.cnn, m.mods.rnn, m.mods.dnn)

        def forward(self, feats, lens):
            x = self.mvn(feats, lens)
            x = self.cnn(x)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            x, _ = self.rnn(x)
            return torch.sigmoid(self.dnn(x))

    w = Wrap(vad).eval()
    model = os.path.join(out_dir, "sb_vad_crdnn.onnx")
    dummy = torch.zeros(1, 100, feats.shape[-1])
    with torch.no_grad():
        torch.onnx.export(
            w, (dummy, torch.ones(1)), model,
            input_names=["feats", "lens"], output_names=["probs"],
            dynamic_axes={"feats": {1: "T"}, "probs": {1: "T"}},
            opset_version=17, dynamo=False,
        )

    sig = BUILTIN["speechbrain"].signature
    common.verify_onnx(model, sig)
    sig.save_json(os.path.join(out_dir, "sb_vad_crdnn.signature.json"))
    card = common.make_card("sb-vad-crdnn", UPSTREAM, "Apache-2.0", sig)
    card_path = os.path.join(out_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card + "\n> First-to-ONNX export of SpeechBrain CRDNN VAD. "
                "LibriParty-tuned (high recall).\n")
    return {"model": model, "card": card_path, "repo": REPO,
            "signature": os.path.join(out_dir, "sb_vad_crdnn.signature.json")}


def publish(out_dir: str = "scratch/sb", token: str | None = None) -> str:
    files = convert(out_dir)
    sha = common.push_to_hf(
        REPO,
        {"sb_vad_crdnn.onnx": files["model"],
         "sb_vad_crdnn.signature.json": files["signature"],
         "README.md": files["card"]},
        token=token,
    )
    print(f"published {REPO} @ {sha}")
    return sha


if __name__ == "__main__":  # pragma: no cover
    convert()
