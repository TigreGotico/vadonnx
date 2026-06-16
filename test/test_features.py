import numpy as np

from vadonnx import features


def test_mel_filterbank_shape():
    fb = features.mel_filterbank(16000, n_fft=400, n_mels=80)
    assert fb.shape == (80, 201)
    assert np.all(fb >= 0)
    assert fb.sum() > 0


def test_log_mel_shape_and_finite():
    audio = np.sin(np.linspace(0, 50, 16000)).astype(np.float32)
    lm = features.log_mel(audio, 16000, n_fft=400, hop=160, n_mels=80)
    assert lm.shape[1] == 80
    assert lm.shape[0] > 1
    assert np.all(np.isfinite(lm))


def test_hz_mel_inverse():
    hz = np.array([0.0, 100.0, 1000.0, 4000.0])
    np.testing.assert_allclose(features.mel_to_hz(features.hz_to_mel(hz)), hz, atol=1e-3)


def test_cmvn():
    feat = np.ones((3, 4), dtype=np.float32)
    out = features.apply_cmvn(feat, mean=-1.0, istd=2.0)
    np.testing.assert_allclose(out, np.zeros((3, 4)), atol=1e-6)


def test_extract_dispatch():
    audio = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    lm = features.extract("logmel", audio, 16000, n_mels=40)
    assert lm.shape[1] == 40
