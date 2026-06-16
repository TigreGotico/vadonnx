"""Classical (non-ONNX) VAD baselines for the benchmark — a reference floor.

These are deliberately *not* part of the vadonnx library (which is ONNX-only); they live
here so the benchmark can show how much the neural models actually buy over a classical
energy gate and the canonical WebRTC GMM VAD. Both subclass ``vadonnx.base.VADModel`` so
they plug into the existing runner/metrics unchanged.
"""
from __future__ import annotations

import numpy as np

from vadonnx.base import VADModel
from vadonnx import load_vad


class EnergyVAD(VADModel):
    """Adaptive log-energy gate. Continuous score (per-frame log-energy min/max
    normalized over the clip), so it yields a meaningful ROC-AUC as an absolute floor."""

    def __init__(self, frame_ms: int = 20, **kw):
        fs = int(16000 * frame_ms / 1000)
        super().__init__(sample_rate=16000, frame_size=fs, stateful=False, **kw)

    def _infer_frame(self, frame: np.ndarray) -> float:  # streaming (crude, fixed dBFS)
        db = 20 * np.log10(np.sqrt((frame ** 2).mean()) + 1e-10)
        return float(np.clip((db + 60.0) / 50.0, 0.0, 1.0))

    def _run_all(self, x: np.ndarray) -> np.ndarray:  # batch (clip-normalized)
        fs = self.frame_size
        n = x.shape[0] // fs
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        e = np.sqrt((x[: n * fs].reshape(n, fs) ** 2).mean(axis=1) + 1e-10)
        db = 20 * np.log10(e + 1e-10)
        lo, hi = np.percentile(db, 10), np.percentile(db, 95)
        return np.clip((db - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


class WebRTCVAD(VADModel):
    """Google WebRTC VAD (GMM, in C via the `webrtcvad` package). Binary per-frame
    decision — ROC-AUC is degenerate; report its operating point (F1/DER/FA/Miss)."""

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 20, **kw):
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness
        fs = int(16000 * frame_ms / 1000)  # 20 ms -> 320 samples (valid webrtc frame)
        super().__init__(sample_rate=16000, frame_size=fs, stateful=False, **kw)

    def _infer_frame(self, frame: np.ndarray) -> float:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        try:
            return 1.0 if self._vad.is_speech(pcm, 16000) else 0.0
        except Exception:
            return 0.0


def load_any(name: str) -> VADModel:
    """Resolve a benchmark model name: classical baselines or a vadonnx backend."""
    if name == "energy":
        return EnergyVAD()
    if name == "webrtc":
        return WebRTCVAD(aggressiveness=2)
    if name.startswith("webrtc-"):
        return WebRTCVAD(aggressiveness=int(name.rsplit("-", 1)[1]))
    return load_vad(name)
