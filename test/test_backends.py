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


def test_pulsevad_specs_pin_url_and_digest():
    for name in ("pulsevad", "pulsevad-fp32", "pulsevad-81k"):
        spec = BUILTIN[name]
        assert spec.backend == "pulsevad"
        assert spec.url and "/af25e79d66830a3fee74541812721f6158fc92b5/" in spec.url
        assert spec.url.endswith("/" + spec.filename)
        assert spec.sha256 and len(spec.sha256) == 64
        assert spec.license == "MIT"
    from vadonnx.backends.pulsevad import PulseVAD
    assert get_backend("pulsevad") is PulseVAD


def test_pulsevad_mel_filterbank():
    """HTK triangles with Slaney area normalization over 257 bins and 64 bands."""
    import numpy as np

    from vadonnx.backends.pulsevad import mel_filterbank

    fb = mel_filterbank()
    assert fb.shape == (257, 64)
    assert fb.dtype == np.float32
    assert fb.min() >= 0
    # every band has one peak and a non-empty support
    assert np.all(fb.max(axis=0) > 0)
    # Slaney normalization: each triangle area is 1 on a 31.25 Hz bin grid
    np.testing.assert_allclose(fb.sum(axis=0) * 31.25, 1.0, rtol=0.2)


def test_pulsevad_frontend_shapes():
    import numpy as np

    from vadonnx.backends.pulsevad import PulseVAD, mel_filterbank

    self = PulseVAD.__new__(PulseVAD)
    self._fb = mel_filterbank()
    n = np.arange(400)
    self._w512 = np.zeros(512, dtype=np.float32)
    self._w512[56:456] = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / 400)
    x = np.random.default_rng(0).standard_normal(3200).astype(np.float32)
    feat = PulseVAD._features(self, x)
    assert feat.shape == (64, 21)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))
    np.testing.assert_allclose(feat.mean(axis=1), 0.0, atol=1e-4)


def test_restricted_weights_are_not_labelled_permissive():
    # ten-vad is Apache-2.0 plus Agora's conditions, and the FSMN weights are under the
    # FunASR Model Open Source License, not MIT. A bare permissive label here misleads
    # anyone who ships on these models (T-2488).
    assert BUILTIN["ten"].license == "Apache-2.0 with TEN additional conditions"
    assert BUILTIN["ten"].signature.license == BUILTIN["ten"].license
    for name in ("fsmn", "fsmn-quant"):
        assert BUILTIN[name].license == "FunASR Model License 1.1"
        assert BUILTIN[name].signature.license == BUILTIN[name].license
    for name in ("ten", "fsmn", "fsmn-quant"):
        assert BUILTIN[name].license not in ("MIT", "Apache-2.0")


def test_every_spec_license_matches_its_signature():
    # the spec and its IOSignature carry the same string; a find-and-replace on
    # one line and not the other once relabelled pyannote as a FunASR model
    for name, spec in BUILTIN.items():
        assert spec.license, name
        assert spec.signature.license == spec.license, name

def test_fsmn_default_threshold_is_funasr_rule():
    # FunASR: speech when exp(log p_speech) >= exp(log p_sil) + speech_noise_thres (0.6),
    # one silence pdf, so p_speech >= 0.8. Below that the model's answer on digital
    # silence (0.635, T-2490) is silence, as in FunASR's own pipeline.
    from vadonnx.backends.fsmn import FUNASR_SPEECH_THRESHOLD, FsmnVAD
    import inspect
    assert FUNASR_SPEECH_THRESHOLD == 0.8
    assert inspect.signature(FsmnVAD.__init__).parameters["threshold"].default == 0.8
    assert 0.635 < FUNASR_SPEECH_THRESHOLD - 0.15, "the deactivation threshold must also sit above the silence answer"
