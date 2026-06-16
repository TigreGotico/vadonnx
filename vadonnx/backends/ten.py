"""TEN VAD backend.

Reproduces TEN VAD's feature pipeline in numpy and drives the TEN ONNX graph. Per
256-sample hop: pre-emphasis (0.97), Hann-768 STFT zero-padded to 1024, power spectrum,
40-band HTK-triangular mel (``log(sum(power·filter)/32768² + 1e-20)``), per-feature
(mean, std) normalization, and a pitch estimate as the 41st feature. Three frames are
stacked into the ``[1, 3, 41]`` model input; four recurrent states are carried across
hops. The mel mean/std and STFT window are bundled in the wheel.

The mel branch matches the upstream C implementation; the pitch feature uses an
autocorrelation estimate in place of TEN's native pitch tracker.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..resolver import data_dir
from ..signature import IOSignature

_FS, _FFT, _HOP, _WIN, _NMEL = 16000, 1024, 256, 768, 40
_NB = _FFT // 2 + 1
_POWER_SCALE = float(_FFT * _FFT)   # matches TEN's FFT output rescaling
_STATE_INPUTS = ("input_2", "input_3", "input_6", "input_7")
_STATE_OUTPUTS = ("output_2", "output_3", "output_6", "output_7")
_HIDDEN = 64


def _mel_filterbank() -> np.ndarray:
    high = 2595.0 * np.log10(1 + 8000.0 / 700.0)
    hz = 700.0 * (10 ** (np.linspace(0.0, high, _NMEL + 2) / 2595.0) - 1)
    edges = np.floor((_FFT + 1) * hz / _FS).astype(int)
    fb = np.zeros((_NMEL, _NB), dtype=np.float64)
    for j in range(_NMEL):
        a, b, c = edges[j], edges[j + 1], edges[j + 2]
        if b > a:
            fb[j, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[j, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


def _autocorr_pitch(frame: np.ndarray, fmin: float = 60.0, fmax: float = 400.0,
                    thr: float = 0.3) -> float:
    x = frame - frame.mean()
    if np.sqrt((x ** 2).mean()) < 1e-4:
        return 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = int(_FS / fmax), int(_FS / fmin)
    seg = ac[lo:hi]
    if seg.size == 0:
        return 0.0
    lag = lo + int(np.argmax(seg))
    return float(_FS / lag) if ac[lag] / (ac[0] + 1e-9) > thr else 0.0


class TenVAD(VADModel):
    """TEN VAD backend (frame-by-frame, 16 ms hops)."""

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
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._fb = _mel_filterbank()
        coeff = np.load(os.path.join(data_dir(), "ten_vad_coeff.npz"))
        self._win = coeff["window"].astype(np.float64)
        self._means = coeff["means"].astype(np.float64)
        self._stds = coeff["stds"].astype(np.float64)
        super().__init__(
            sample_rate=_FS, frame_size=_HOP, stateful=True,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    def _reset_state(self) -> None:
        self._state = {n: np.zeros((1, _HIDDEN), dtype=np.float32) for n in _STATE_INPUTS}
        self._stack = np.zeros((3, 41), dtype=np.float32)
        self._emph_q = np.zeros(_WIN, dtype=np.float64)
        self._raw_q = np.zeros(_WIN, dtype=np.float64)
        self._preemph_prev = 0.0

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        frame = frame_f32.astype(np.float64)
        # pre-emphasis y[i] = x[i] - 0.97*x[i-1], continuous across hops
        emph = np.empty(_HOP, dtype=np.float64)
        emph[0] = frame[0] - 0.97 * self._preemph_prev
        emph[1:] = frame[1:] - 0.97 * frame[:-1]
        self._preemph_prev = frame[-1]
        self._emph_q = np.concatenate([self._emph_q[_HOP:], emph])
        self._raw_q = np.concatenate([self._raw_q[_HOP:], frame])

        spec = np.fft.rfft(np.concatenate([self._emph_q * self._win,
                                           np.zeros(_FFT - _WIN)]), n=_FFT)
        power = (spec.real ** 2 + spec.imag ** 2) * _POWER_SCALE
        mel = np.log((power @ self._fb.T) / (32768.0 ** 2) + 1e-20)
        feat = np.empty(41, dtype=np.float32)
        feat[:40] = (mel - self._means[:40]) / (self._stds[:40] + 1e-20)
        pitch = _autocorr_pitch(self._raw_q)
        feat[40] = (pitch - self._means[40]) / (self._stds[40] + 1e-20)

        self._stack = np.vstack([self._stack[1:], feat[None]])
        feed = {"input_1": self._stack[None].astype(np.float32), **self._state}
        outs = self.session.run(None, feed)
        named = dict(zip(self._out_names, outs))
        for si, so in zip(_STATE_INPUTS, _STATE_OUTPUTS):
            self._state[si] = named[so]
        return float(np.asarray(named["output_1"]).reshape(-1)[-1])
