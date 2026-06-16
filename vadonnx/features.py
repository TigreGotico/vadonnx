"""Numpy-only acoustic features (log-mel / fbank + CMVN).

Some VAD models (FSMN, MarbleNet) classify acoustic features rather than raw PCM.
The preferred approach is to *fold* the feature extractor into the ONNX graph at
conversion time so the runtime stays raw-PCM-in / prob-out (``feature=None``). These
helpers exist as a dependency-light fallback for models whose graph expects features,
and are exercised directly by the test-suite. Implemented with numpy FFT only — no
librosa/torchaudio at runtime.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def hz_to_mel(hz: np.ndarray, htk: bool = True) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    # Slaney
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (hz - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = hz >= min_log_hz
    mels[log_t] = min_log_mel + np.log(hz[log_t] / min_log_hz) / logstep
    return mels


def mel_to_hz(mel: np.ndarray, htk: bool = True) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mel
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mel >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mel[log_t] - min_log_mel))
    return freqs


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = True,
) -> np.ndarray:
    """Triangular mel filterbank of shape ``(n_mels, n_fft // 2 + 1)``."""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)
    mel_pts = np.linspace(hz_to_mel(np.array([fmin]), htk)[0],
                          hz_to_mel(np.array([fmax]), htk)[0], n_mels + 2)
    hz_pts = mel_to_hz(mel_pts, htk)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        if center == left or right == center:
            continue
        lo = (fft_freqs - left) / (center - left)
        hi = (right - fft_freqs) / (right - center)
        fb[m - 1] = np.clip(np.minimum(lo, hi), 0.0, None)
    return fb.astype(np.float32)


def _frame(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if x.shape[0] < win:
        x = np.concatenate([x, np.zeros(win - x.shape[0], dtype=x.dtype)])
    n = 1 + (x.shape[0] - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def log_mel(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 400,
    hop: int = 160,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    log_floor: float = 1e-10,
    htk: bool = True,
    window: str = "hann",
) -> np.ndarray:
    """Log-mel spectrogram of shape ``(n_frames, n_mels)``."""
    x = np.asarray(audio, dtype=np.float32)
    frames = _frame(x, n_fft, hop).astype(np.float64)
    if window == "hann":
        frames = frames * np.hanning(n_fft)[None, :]
    elif window == "hamming":
        frames = frames * np.hamming(n_fft)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, htk).astype(np.float64)
    mel = spec @ fb.T
    return np.log(np.maximum(mel, log_floor)).astype(np.float32)


def fbank(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """Alias for :func:`log_mel` (kaldi-style log mel filterbank energies)."""
    return log_mel(audio, sample_rate, **kwargs)


def apply_cmvn(
    feat: np.ndarray, mean: np.ndarray, istd: np.ndarray
) -> np.ndarray:
    """Apply cepstral mean and (inverse) variance normalization."""
    return ((feat + np.asarray(mean, dtype=np.float32))
            * np.asarray(istd, dtype=np.float32)).astype(np.float32)


_FEATURES = {"logmel": log_mel, "fbank": fbank, "fbank_cmvn": fbank}


def extract(name: str, audio: np.ndarray, sample_rate: int, **params) -> np.ndarray:
    """Dispatch to a named feature extractor (used by the generic backend)."""
    if name not in _FEATURES:
        raise ValueError(f"unknown feature {name!r}; known: {sorted(_FEATURES)}")
    cmvn = params.pop("cmvn", None)
    feat = _FEATURES[name](audio, sample_rate, **params)
    if cmvn is not None:
        feat = apply_cmvn(feat, cmvn.get("mean", 0.0), cmvn.get("istd", 1.0))
    return feat
