"""Base class shared by every VAD backend.

Backends only implement :meth:`VADModel._infer_frame` (run one fixed-size frame
through the model and return a speech probability) and, if stateful, override
:meth:`VADModel._reset_state`. Everything users actually call — streaming
``process_chunk``, batch ``probabilities`` and ``get_speech_segments``, audio
normalization, resampling and framing — is implemented once here.
"""
from __future__ import annotations

import abc
from math import inf
from typing import List, Optional

import numpy as np

from .audio import AudioLike, resample, to_float32_mono
from .segment import SpeechSegment, probs_to_segments


class VADModel(abc.ABC):
    """Abstract base for all voice activity detection backends."""

    def __init__(
        self,
        *,
        sample_rate: int,
        frame_size: int,
        stateful: bool = False,
        threshold: float = 0.5,
        neg_threshold: Optional[float] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.stateful = bool(stateful)
        self.threshold = float(threshold)
        self.neg_threshold = neg_threshold
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._reset_state()

    # ------------------------------------------------------------------ #
    # backend hooks
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _infer_frame(self, frame_f32: np.ndarray) -> float:
        """Run one ``(frame_size,)`` float32 frame and return P(speech) in [0, 1]."""

    def _reset_state(self) -> None:
        """Reset recurrent state. No-op for stateless models; overridden otherwise."""

    @property
    def frame_duration(self) -> float:
        """Seconds of audio represented by one frame at the model's native rate."""
        return self.frame_size / self.sample_rate

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear streaming buffer and recurrent state. Call between utterances."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._reset_state()

    def _prepare(self, audio: AudioLike, sample_rate: Optional[int]) -> np.ndarray:
        x = to_float32_mono(audio)
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = resample(x, sample_rate, self.sample_rate)
        return x

    def process_chunk(
        self, audio: AudioLike, sample_rate: Optional[int] = None
    ) -> float:
        """Feed one streaming chunk; return the most recent frame's speech probability.

        Chunks need not align to the model frame size — leftover samples are buffered
        and combined with the next call. If the chunk does not yet complete a frame,
        the previously computed probability is returned.
        """
        x = self._prepare(audio, sample_rate)
        if x.size:
            self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        fs = self.frame_size
        n_full = self._buf.shape[0] // fs
        for i in range(n_full):
            frame = self._buf[i * fs : (i + 1) * fs]
            self._last_prob = float(self._infer_frame(frame))
        if n_full:
            self._buf = self._buf[n_full * fs :].copy()
        return self._last_prob

    def __call__(self, audio: AudioLike, sample_rate: Optional[int] = None) -> float:
        return self.process_chunk(audio, sample_rate)

    def is_speech(
        self,
        audio: AudioLike,
        sample_rate: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Convenience boolean wrapper around :meth:`process_chunk`."""
        prob = self.process_chunk(audio, sample_rate)
        return prob >= (self.threshold if threshold is None else threshold)

    def _run_all(self, x: np.ndarray) -> np.ndarray:
        """Reset, then run every (zero-padded) frame of ``x`` and return all probs."""
        self.reset()
        fs = self.frame_size
        if x.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        n_frames = -(-x.shape[0] // fs)  # ceil
        pad = n_frames * fs - x.shape[0]
        if pad:
            x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
        out = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            out[i] = self._infer_frame(x[i * fs : (i + 1) * fs])
        self.reset()
        return out

    def probabilities(
        self, audio: AudioLike, sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Return the per-frame speech probability array for a whole audio buffer."""
        x = self._prepare(audio, sample_rate)
        return self._run_all(x)

    def get_speech_segments(
        self,
        audio: AudioLike,
        sample_rate: Optional[int] = None,
        *,
        threshold: Optional[float] = None,
        neg_threshold: Optional[float] = None,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
        max_speech_duration: float = inf,
        speech_pad: float = 0.03,
    ) -> List[SpeechSegment]:
        """Detect speech spans over a whole audio buffer.

        Returns a list of :class:`~vadonnx.segment.SpeechSegment` with ``start``/``end``
        in seconds, computed by running :meth:`probabilities` then applying the shared
        double-threshold segmentation.
        """
        x = self._prepare(audio, sample_rate)
        probs = self._run_all(x)
        return probs_to_segments(
            probs,
            self.frame_duration,
            threshold=self.threshold if threshold is None else threshold,
            neg_threshold=self.neg_threshold if neg_threshold is None else neg_threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            max_speech_duration=max_speech_duration,
            speech_pad=speech_pad,
            audio_duration=x.shape[0] / self.sample_rate,
        )
