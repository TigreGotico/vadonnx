"""FunASR FSMN-VAD backend.

FSMN-VAD classifies kaldi-style fbank features (80-mel), low-frame-rate stacked
(``lfr_m=5``) and CMVN-normalised to a 400-dim input, through a streaming FSMN with
four recurrent caches. The graph emits a 248-way softmax per 10 ms frame; the speech
probability is ``1 - p(silence)`` where silence is PDF id 0
(``P(speech) = 1 - softmax[..., 0]``).

Feature extraction uses `kaldi-native-fbank` (the ``fsmn`` extra); the CMVN statistics
are bundled in the wheel.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from ..base import VADModel
from ..resolver import data_dir
from ..signature import IOSignature

_LFR_M = 5
_LFR_N = 1
_FBANK_SHIFT = 160  # 10 ms @ 16 kHz


def _apply_lfr(feat: np.ndarray, m: int = _LFR_M, n: int = _LFR_N) -> np.ndarray:
    T = feat.shape[0]
    t_lfr = int(np.ceil(T / n))
    left = np.tile(feat[0], ((m - 1) // 2, 1))
    feat = np.vstack((left, feat))
    T = T + (m - 1) // 2
    rows = []
    for i in range(t_lfr):
        if m <= T - i * n:
            rows.append(feat[i * n : i * n + m].reshape(1, -1))
        else:
            num_pad = m - (T - i * n)
            row = feat[i * n :].reshape(-1)
            for _ in range(num_pad):
                row = np.hstack((row, feat[-1]))
            rows.append(row)
    return np.vstack(rows).astype(np.float32)


class FsmnVAD(VADModel):
    """FSMN-VAD backend (batch + streaming)."""

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
        cmvn_file: Optional[str] = None,
    ):
        import onnxruntime as ort

        self.signature = signature
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = inter_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._state_in = [i.name for i in self.session.get_inputs()][1:]
        self._out_names = [o.name for o in self.session.get_outputs()]

        cmvn_path = cmvn_file or os.path.join(data_dir(), "fsmn_cmvn.npz")
        cmvn = np.load(cmvn_path)
        self._means, self._vars = cmvn["means"], cmvn["vars"]

        # 10 ms frame timing for segmentation
        super().__init__(
            sample_rate=16000, frame_size=_FBANK_SHIFT, stateful=True,
            threshold=threshold, neg_threshold=neg_threshold,
        )

    # ------------------------------------------------------------------ #
    def _fbank_opts(self):
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.mel_opts.num_bins = 80
        opts.energy_floor = 0
        opts.frame_opts.snip_edges = True
        return opts

    def _features(self, audio_f32: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        fb = knf.OnlineFbank(self._fbank_opts())
        fb.accept_waveform(16000, (audio_f32 * (1 << 15)).tolist())
        n = fb.num_frames_ready
        if n == 0:
            return np.zeros((0, 400), dtype=np.float32)
        mat = np.array([fb.get_frame(i) for i in range(n)], dtype=np.float32)
        feat = _apply_lfr(mat)
        return ((feat + self._means) * self._vars).astype(np.float32)

    def _zero_caches(self):
        return [np.zeros((1, 128, 19, 1), dtype=np.float32) for _ in self._state_in]

    def _forward(self, feats: np.ndarray, caches):
        feed = {"speech": feats[None]}
        for name, c in zip(self._state_in, caches):
            feed[name] = c
        outs = self.session.run(None, feed)
        logits = outs[0][0]  # (T, 248) softmax
        p_speech = (1.0 - logits[:, 0]).astype(np.float32)
        new_caches = list(outs[1:])
        return p_speech, new_caches

    # batch path — full sequence in one shot (verified sane vs upstream)
    def _run_all(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        feats = self._features(x)
        if feats.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        p_speech, _ = self._forward(feats, self._zero_caches())
        return p_speech

    # streaming path
    def _reset_state(self) -> None:
        self._caches = self._zero_caches() if hasattr(self, "session") else []
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._stream_prob = 0.0

    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        # accumulate a small block before running the model (fbank needs a 25 ms window)
        self._audio_buf = np.concatenate([self._audio_buf, frame_f32])
        if self._audio_buf.shape[0] < 16 * _FBANK_SHIFT:  # ~160 ms blocks
            return self._stream_prob
        feats = self._features(self._audio_buf)
        self._audio_buf = np.zeros(0, dtype=np.float32)
        if feats.shape[0]:
            p_speech, self._caches = self._forward(feats, self._caches)
            if p_speech.size:
                self._stream_prob = float(p_speech[-1])
        return self._stream_prob
