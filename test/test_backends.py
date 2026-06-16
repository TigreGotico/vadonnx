"""Offline structural tests for the non-bundled backends.

Functional inference for these models requires their downloaded ONNX (see
test_download.py, network-gated). These tests verify the wiring offline.
"""
import os

from vadonnx.registry import BUILTIN, get_backend
from vadonnx.resolver import data_dir


def test_backend_classes_resolve():
    from vadonnx.backends.fsmn import FsmnVAD
    from vadonnx.backends.marblenet import MarbleVAD
    from vadonnx.backends.speechbrain import SbVAD
    from vadonnx.backends.pyannote import PyannoteVAD
    from vadonnx.backends.ten import TenVAD
    from vadonnx.onnx_backend import OnnxVAD

    assert get_backend("onnx") is OnnxVAD
    assert get_backend("ten") is TenVAD
    assert get_backend("fsmn") is FsmnVAD
    assert get_backend("marblenet") is MarbleVAD
    assert get_backend("speechbrain") is SbVAD
    assert get_backend("pyannote") is PyannoteVAD


def test_specs_have_repo_and_revision():
    for name in ("silero", "silero-8k", "ten", "fsmn", "fsmn-quant",
                 "marblenet", "marblenet-int8", "speechbrain", "pyannote", "pyannote-int8"):
        spec = BUILTIN[name]
        assert spec.hf_repo, f"{name} missing hf_repo"
        assert spec.revision, f"{name} missing pinned revision"
        assert spec.signature.sample_rate in (8000, 16000)


def test_bundled_feature_assets_present():
    for fn in ("silero_vad.onnx", "fsmn_cmvn.npz",
               "marblenet_mel_fb.npy", "marblenet_window.npy", "ten_vad_coeff.npz",
               "sb_vad_mel_fb.npy", "sb_vad_window.npy"):
        assert os.path.isfile(os.path.join(data_dir(), fn)), fn


def test_marblenet_frontend_shapes():
    """The MarbleNet numpy mel frontend runs offline (no ONNX needed)."""
    import numpy as np

    from vadonnx.backends.marblenet import MarbleVAD

    # call the frontend without constructing a session
    self = MarbleVAD.__new__(MarbleVAD)
    self._fb = np.load(os.path.join(data_dir(), "marblenet_mel_fb.npy")).astype(np.float64)
    win = np.load(os.path.join(data_dir(), "marblenet_window.npy")).astype(np.float64)
    self._w512 = np.zeros(512)
    off = (512 - win.shape[0]) // 2
    self._w512[off:off + win.shape[0]] = win
    feat = MarbleVAD._features(self, np.random.default_rng(0).standard_normal(16000).astype(np.float32))
    assert feat.shape[0] == 80
    assert feat.shape[1] > 1
    assert np.all(np.isfinite(feat))
