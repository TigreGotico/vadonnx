"""pyannote segmentation-3.0 VAD backend (community ONNX).

pyannote's segmentation-3.0 is a powerset speaker-segmentation model: it takes a raw
10 s waveform and emits per-frame logits over 7 classes (class 0 = non-speech, 1-3 =
single speakers, 4-6 = speaker pairs). Voice activity is ``P(speech) = 1 - softmax[0]``.

The model operates on fixed ~10 s windows, so this backend slides a non-overlapping
window over the audio (padding the tail), runs each, and concatenates the per-frame
speech probabilities (~17 ms resolution). Uses the community ONNX export
(onnx-community/pyannote-segmentation-3.0, MIT).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..signature import IOSignature

_WINDOW = 160000  # 10 s @ 16 kHz


class PyannoteVAD(VADModel):
    """Windowed voice activity detection from pyannote segmentation-3.0."""

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
        self._in = self.session.get_inputs()[0].name
        self._fpw = 589  # frames per 10 s window (refined after first inference)
        super().__init__(
            sample_rate=16000, frame_size=272, stateful=False,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    @property
    def frame_duration(self) -> float:
        return (_WINDOW / self.sample_rate) / self._fpw

    def _window_probs(self, window: np.ndarray) -> np.ndarray:
        logits = self.session.run(
            None, {self._in: window.reshape(1, 1, _WINDOW).astype(np.float32)}
        )[0][0]                                   # [F, 7]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        sm = e / e.sum(axis=-1, keepdims=True)
        return (1.0 - sm[:, 0]).astype(np.float32)  # P(speech)

    def _run_all(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        n = int(np.ceil(x.shape[0] / _WINDOW))
        xp = np.pad(x, (0, n * _WINDOW - x.shape[0]))
        probs = []
        for i in range(n):
            probs.append(self._window_probs(xp[i * _WINDOW : (i + 1) * _WINDOW]))
        self._fpw = len(probs[0])
        full = np.concatenate(probs)
        keep = int(round(x.shape[0] / self.sample_rate / self.frame_duration))
        return full[:keep] if keep else full

    def _reset_state(self) -> None:
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._stream_prob = 0.0

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        self._audio_buf = np.concatenate([self._audio_buf, frame_f32])
        if self._audio_buf.shape[0] < _WINDOW:
            return self._stream_prob
        p = self._window_probs(self._audio_buf[:_WINDOW])
        self._audio_buf = self._audio_buf[_WINDOW:]
        if p.size:
            self._stream_prob = float(p[-1])
        return self._stream_prob

    def _flush(self) -> float:
        if self._audio_buf.size:
            pad = np.pad(self._audio_buf, (0, _WINDOW - self._audio_buf.shape[0]))
            p = self._window_probs(pad)
            # keep only frames covering the real (unpadded) residual
            keep = max(1, int(round(self._audio_buf.shape[0] / self.sample_rate
                                    / self.frame_duration)))
            self._audio_buf = np.zeros(0, dtype=np.float32)
            if p.size:
                self._stream_prob = float(p[:keep][-1])
        return self._stream_prob
