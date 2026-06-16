"""NVIDIA NeMo Frame-VAD MarbleNet backend.

MarbleNet classifies 80-dim log-mel features (preemphasised, centered STFT, power
spectrum, NeMo's mel filterbank, ``log(x + 5.96e-8)``) and emits 2-class logits per
20 ms output frame; ``P(speech) = softmax(logits)[..., 1]``. The mel filterbank and
window are extracted from NeMo and bundled in the wheel, so the numpy frontend
reproduces NeMo's preprocessor closely (end-to-end MAE ~4e-4 vs NeMo).

The ONNX graph takes features (NeMo's STFT preprocessor cannot be folded into ONNX —
`torch.stft` is not exportable), so this backend computes the mel frontend in numpy.
MarbleNet is a CNN over the whole sequence (stateless across calls); streaming
processes audio in blocks.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..resolver import data_dir
from ..signature import IOSignature

_N_FFT = 512
_HOP = 160
_GUARD = 5.960464477539063e-08
_PREEMPH = 0.97
_OUT_FRAME = 320  # 20 ms @ 16 kHz (model downsamples mel frames by 2)


class MarbleVAD(VADModel):
    """Best-effort NeMo Frame-VAD MarbleNet backend (near-parity numpy frontend)."""

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
        self._fb = np.load(os.path.join(data_dir(), "marblenet_mel_fb.npy")).astype(np.float64)
        win = np.load(os.path.join(data_dir(), "marblenet_window.npy")).astype(np.float64)
        self._w512 = np.zeros(_N_FFT, dtype=np.float64)
        off = (_N_FFT - win.shape[0]) // 2
        self._w512[off:off + win.shape[0]] = win

        super().__init__(
            sample_rate=16000, frame_size=_OUT_FRAME, stateful=False,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    # ------------------------------------------------------------------ #
    def _features(self, audio: np.ndarray) -> np.ndarray:
        """NeMo-matching log-mel; returns [80, T]."""
        x = audio.astype(np.float64)
        if x.shape[0] < 2:
            return np.zeros((self._fb.shape[0], 0), dtype=np.float32)
        x = np.concatenate([x[:1], x[1:] - _PREEMPH * x[:-1]])
        xp = np.pad(x, _N_FFT // 2, mode="reflect")
        n = 1 + (len(xp) - _N_FFT) // _HOP
        if n <= 0:
            return np.zeros((self._fb.shape[0], 0), dtype=np.float32)
        idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n)[:, None]
        frames = xp[idx] * self._w512
        spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
        power = spec.real ** 2 + spec.imag ** 2
        mel = power @ self._fb.T            # [n,257]@[257,80]
        logmel = np.log(mel + _GUARD)
        return logmel.T.astype(np.float32)  # [80, n]

    def _forward(self, audio: np.ndarray) -> np.ndarray:
        feat = self._features(audio)
        if feat.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.session.run(None, {self._in_name: feat[None]})[0][0]  # [T, 2]
        out = out - out.max(axis=-1, keepdims=True)
        e = np.exp(out)
        return (e[:, 1] / e.sum(axis=-1)).astype(np.float32)

    # batch path (near-parity vs NeMo)
    def _run_all(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x) if x.shape[0] else np.zeros(0, dtype=np.float32)

    # streaming path — MarbleNet is stateless across calls; process in blocks
    def _reset_state(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._stream_prob = 0.0

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        self._audio_buf = np.concatenate([self._audio_buf, frame_f32])
        if self._audio_buf.shape[0] < 25 * _OUT_FRAME:  # ~0.5 s blocks
            return self._stream_prob
        p = self._forward(self._audio_buf)
        self._audio_buf = np.zeros(0, dtype=np.float32)
        if p.size:
            self._stream_prob = float(p[-1])
        return self._stream_prob
