"""PulseVAD backend.

The ONNX graph classifies one 200 ms window (3200 samples at 16 kHz) into
``[non_speech, speech]`` logits, so ``P(speech) = sigmoid(logits[1] - logits[0])``.
The frontend is computed in numpy and follows the upstream ``frontend_np`` module:
preemphasis (0.97), waveform z-norm, centered reflect-padded STFT with a periodic
Hann-400 window inside ``n_fft=512``, a 64-band HTK mel filterbank with Slaney area
normalization, ``log(x + 1e-5)`` and a per-band z-norm over the 21 frames. Windows do
not overlap and carry no state, so each 200 ms frame is independent.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..signature import IOSignature

_SR = 16000
_WINDOW = 3200
_N_FFT = 512
_WIN = 400
_HOP = 160
_N_MELS = 64
_N_FRAMES = 21
_PREEMPH = 0.97
_EPS = 1e-5


def mel_filterbank() -> np.ndarray:
    """HTK-scale triangular filters with Slaney area normalization, shape ``[257, 64]``.

    Matches ``torchaudio.functional.melscale_fbanks(257, 0, 8000, 64, 16000,
    norm="slaney", mel_scale="htk")`` and the ``mel_filterbank.npy`` shipped upstream.
    """
    freqs = np.linspace(0, _SR // 2, _N_FFT // 2 + 1)
    hi = 2595.0 * np.log10(1.0 + (_SR / 2) / 700.0)
    pts = 700.0 * (10 ** (np.linspace(0.0, hi, _N_MELS + 2) / 2595.0) - 1.0)
    diff = pts[1:] - pts[:-1]
    slopes = pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    fb *= (2.0 / (pts[2:] - pts[:-2]))[None, :]
    return fb.astype(np.float32)


class PulseVAD(VADModel):
    """PulseVAD tiny causal CNN backend."""

    def __init__(
        self,
        model_path: str,
        signature: IOSignature,
        *,
        providers: Optional[List[str]] = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        threshold: float = 0.5,
        neg_threshold: Optional[float] = None,
    ):
        import onnxruntime as ort

        self.signature = signature
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = inter_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._in_name = self.session.get_inputs()[0].name
        self._fb = mel_filterbank()
        n = np.arange(_WIN)
        win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / _WIN)).astype(np.float32)
        off = (_N_FFT - _WIN) // 2
        self._w512 = np.zeros(_N_FFT, dtype=np.float32)
        self._w512[off:off + _WIN] = win
        super().__init__(
            sample_rate=_SR, frame_size=_WINDOW, stateful=False,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    def _features(self, window: np.ndarray) -> np.ndarray:
        """Normalized log-mel of one 3200-sample window; returns ``[64, 21]``."""
        x = window.astype(np.float32)
        x = np.concatenate([x[:1], x[1:] - _PREEMPH * x[:-1]])
        x = (x - x.mean()) / (x.std() + _EPS)
        xp = np.pad(x, _N_FFT // 2, mode="reflect")
        idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(_N_FRAMES)[:, None]
        frames = xp[idx] * self._w512
        spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
        # keep float64 up to the log, as upstream does; the int8 graph amplifies
        # a float32 rounding gap here into a visible probability change
        power = np.abs(spec) ** 2
        logmel = np.log((power @ self._fb).T + _EPS)
        mean = logmel.mean(axis=-1, keepdims=True)
        std = logmel.std(axis=-1, keepdims=True)
        return ((logmel - mean) / (std + _EPS)).astype(np.float32)

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        logits = self.session.run(None, {self._in_name: self._features(frame_f32)[None]})[0]
        diff = float(logits[0, 1] - logits[0, 0])
        return float(1.0 / (1.0 + np.exp(-diff)))
