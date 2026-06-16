"""Audio normalization, resampling and WAV reading helpers (numpy-only core).

Everything in :mod:`vadonnx` works internally with mono ``float32`` PCM in the range
``[-1, 1]``. These helpers convert the various shapes users pass in (raw ``int16``
bytes, numpy int/float arrays, stereo) into that canonical form and resample between
sample rates.
"""
from __future__ import annotations

import wave
from typing import Tuple, Union

import numpy as np

AudioLike = Union[bytes, bytearray, memoryview, np.ndarray]


def to_float32_mono(audio: AudioLike) -> np.ndarray:
    """Convert audio of various dtypes/containers to mono float32 in [-1, 1].

    Accepts raw little-endian ``int16`` PCM bytes, or a numpy array of dtype
    ``int16`` / ``int32`` / ``float32`` / ``float64``. Multi-channel arrays
    (shape ``(n, channels)``) are downmixed by averaging channels.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(audio), dtype="<i2").astype(np.float32) / 32768.0
        return arr

    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be bytes or np.ndarray, got {type(audio).__name__}"
        )

    arr = audio
    # downmix to mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        raise ValueError(f"audio array must be 1-D or 2-D, got {arr.ndim}-D")

    if arr.dtype == np.float32:
        out = arr
    elif arr.dtype == np.float64:
        out = arr.astype(np.float32)
    elif arr.dtype == np.int16:
        out = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        out = arr.astype(np.float32) / 2147483648.0
    elif arr.dtype == np.uint8:  # unsigned 8-bit PCM, midpoint 128
        out = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        raise TypeError(f"unsupported audio dtype: {arr.dtype}")

    return np.ascontiguousarray(out, dtype=np.float32)


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono float32 audio from ``src_sr`` to ``dst_sr``.

    Uses ``scipy.signal.resample_poly`` when SciPy is installed (the ``[resample]``
    extra), otherwise falls back to linear interpolation via :func:`numpy.interp`,
    which is adequate for voice activity detection framing.
    """
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly  # type: ignore

        from math import gcd

        g = gcd(int(src_sr), int(dst_sr))
        up, down = int(dst_sr) // g, int(src_sr) // g
        return resample_poly(x, up, down).astype(np.float32)
    except Exception:
        # linear interpolation fallback (no extra deps)
        n_out = int(round(x.shape[0] * dst_sr / src_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        src_idx = np.linspace(0.0, x.shape[0] - 1, num=n_out, dtype=np.float64)
        return np.interp(src_idx, np.arange(x.shape[0]), x).astype(np.float32)


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV file into mono float32 audio. Returns ``(audio, sample_rate)``.

    Uses the standard-library :mod:`wave` module so no third-party audio dependency
    is required for examples and tests.
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), sr
