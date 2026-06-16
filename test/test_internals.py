"""Offline unit tests for internal helpers and numpy feature frontends."""
import sys

import numpy as np

from vadonnx.onnx_backend import OnnxVAD
from vadonnx.signature import IOSignature


# ---- prob_extract reducers (OnnxVAD._reduce) ----
def _reducer(mode, arr, collapse=None):
    o = OnnxVAD.__new__(OnnxVAD)
    o.signature = IOSignature(16000, 512, prob_extract=mode, multiclass_collapse=collapse)
    return o._reduce(np.asarray(arr, dtype=np.float32))


def test_reduce_modes():
    assert abs(_reducer("scalar", [[0.7]]) - 0.7) < 1e-5
    assert abs(_reducer("last", [0.1, 0.2, 0.9]) - 0.9) < 1e-5
    assert abs(_reducer("mean", [0.0, 1.0]) - 0.5) < 1e-6
    assert abs(_reducer("index:1", [[0.2, 0.8]]) - 0.8) < 1e-5
    assert abs(_reducer("1-minus:0", [[0.3, 0.7]]) - 0.7) < 1e-5
    assert _reducer("scalar", []) == 0.0


def test_reduce_multiclass_collapse():
    # classes 1+2 are speech; sum then mean
    assert abs(_reducer("mean", [[0.1, 0.3, 0.6]], collapse=[1, 2]) - 0.9) < 1e-6


# ---- TEN frontend ----
def test_ten_mel_filterbank():
    from vadonnx.backends.ten import _mel_filterbank

    fb = _mel_filterbank()
    assert fb.shape == (40, 513)
    assert np.all(fb >= 0) and fb.sum() > 0


def test_ten_autocorr_pitch():
    from vadonnx.backends.ten import _autocorr_pitch

    assert _autocorr_pitch(np.zeros(768, dtype=np.float32)) == 0.0
    t = np.arange(768) / 16000.0
    f0 = _autocorr_pitch(np.sin(2 * np.pi * 150 * t).astype(np.float32))
    assert 130 < f0 < 170  # recovers ~150 Hz


# ---- FSMN LFR ----
def test_fsmn_apply_lfr():
    from vadonnx.backends.fsmn import _apply_lfr

    feat = np.random.default_rng(0).standard_normal((20, 80)).astype(np.float32)
    out = _apply_lfr(feat)  # lfr_m=5, lfr_n=1 -> 400-dim rows
    assert out.shape[1] == 400
    assert out.shape[0] == 20


# ---- resample numpy fallback (no scipy) ----
def test_resample_numpy_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.signal", None)
    from vadonnx.audio import resample

    x = np.sin(np.linspace(0, 20, 16000)).astype(np.float32)
    y = resample(x, 16000, 8000)
    assert abs(len(y) - 8000) <= 1
    assert y.dtype == np.float32


# ---- empty audio ----
def test_empty_audio(silero):
    assert silero.probabilities(np.zeros(0, dtype=np.float32)).size == 0
    assert silero.get_speech_segments(np.zeros(0, dtype=np.float32)) == []
    silero.reset()
    assert silero.process_chunk(np.zeros(0, dtype=np.float32)) == 0.0
