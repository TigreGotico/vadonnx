"""SpeechBrain CRDNN VAD backend (first-to-ONNX export).

`speechbrain/vad-crdnn-libriparty` ships no ONNX upstream; vadonnx exports the
features→posterior graph (Fbank → mean-var-norm → CNN → RNN → DNN → sigmoid) and
reproduces SpeechBrain's exact Fbank in numpy (STFT center-padded, Hamming window,
power spectrum, SpeechBrain's mel filter matrix, dB with top-dB clamp) — end-to-end
parity vs SpeechBrain is exact (MAE 0). The mel matrix + window are bundled in the wheel.

Note: this model is LibriParty-trained and tends toward high recall (it over-detects on
clean broadcast speech); see docs/backends.md.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..resolver import data_dir
from ..signature import IOSignature

_N_FFT = 400
_HOP = 160
_AMIN = 1e-10
_TOP_DB = 80.0


class SbVAD(VADModel):
    """SpeechBrain CRDNN VAD (10 ms frames, posterior in [0, 1])."""

    def __init__(
        self,
        model_path: str,
        signature: IOSignature,
        *,
        providers: Optional[List[str]] = None,
        intra_threads: int = 1,
        inter_threads: int = 1,
        threshold: float = 0.7,
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
        self._in = self.session.get_inputs()[0].name
        self._fb = np.load(os.path.join(data_dir(), "sb_vad_mel_fb.npy")).astype(np.float64)
        self._win = np.load(os.path.join(data_dir(), "sb_vad_window.npy")).astype(np.float64)
        super().__init__(
            sample_rate=16000, frame_size=_HOP, stateful=False,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    def _features(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float64)
        if x.shape[0] < _N_FFT:
            x = np.pad(x, (0, _N_FFT - x.shape[0]))
        xp = np.pad(x, _N_FFT // 2, mode="constant")
        n = 1 + (len(xp) - _N_FFT) // _HOP
        idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n)[:, None]
        frames = xp[idx] * self._win
        spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
        power = spec.real ** 2 + spec.imag ** 2
        mel = power @ self._fb                      # [n,201]@[201,40]
        db = 10.0 * np.log10(np.clip(mel, _AMIN, None))
        db = np.maximum(db, db.max() - _TOP_DB)
        return db.astype(np.float32)                # [n, 40]

    def _forward(self, audio: np.ndarray) -> np.ndarray:
        feats = self._features(audio)
        if feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.session.run(None, {self._in: feats[None],
                                      "lens": np.ones(1, dtype=np.float32)})[0]
        return out[0, :, 0].astype(np.float32)

    def _run_all(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x) if x.shape[0] else np.zeros(0, dtype=np.float32)

    def _reset_state(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._stream_prob = 0.0

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        self._audio_buf = np.concatenate([self._audio_buf, frame_f32])
        if self._audio_buf.shape[0] < 50 * _HOP:  # ~0.5 s blocks
            return self._stream_prob
        p = self._forward(self._audio_buf)
        self._audio_buf = np.zeros(0, dtype=np.float32)
        if p.size:
            self._stream_prob = float(p[-1])
        return self._stream_prob
