"""Turn a sequence of per-frame speech probabilities into speech segments.

This is the one place the segmentation policy lives; every backend reuses it via
:meth:`vadonnx.base.VADModel.get_speech_segments`. It implements the standard
double-threshold (hysteresis) state machine used by Silero/sherpa-onnx: speech is
*entered* when the probability rises above ``threshold`` and *left* only after it
stays below ``neg_threshold`` for at least ``min_silence_duration``. Short blips are
discarded and over-long segments are split.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import List, Optional

import numpy as np


@dataclass
class SpeechSegment:
    """A detected span of speech, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple:
        return (self.start, self.end)

    def __iter__(self):
        yield self.start
        yield self.end


def probs_to_segments(
    probs: np.ndarray,
    frame_duration: float,
    *,
    threshold: float = 0.5,
    neg_threshold: Optional[float] = None,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.1,
    max_speech_duration: float = inf,
    speech_pad: float = 0.03,
    audio_duration: Optional[float] = None,
) -> List[SpeechSegment]:
    """Convert per-frame probabilities to a list of :class:`SpeechSegment`.

    Args:
        probs: 1-D array of per-frame speech probabilities in ``[0, 1]``.
        frame_duration: seconds represented by each probability (``frame_size / sr``).
        threshold: activation threshold; speech starts when ``prob >= threshold``.
        neg_threshold: deactivation threshold; defaults to ``threshold - 0.15``
            (floored at 0.01). Hysteresis prevents rapid on/off chatter.
        min_speech_duration: discard speech spans shorter than this (seconds).
        min_silence_duration: silence must last this long to end a speech span.
        max_speech_duration: split spans longer than this (seconds).
        speech_pad: pad each side of every segment by this many seconds.
        audio_duration: total audio length (seconds) used to clamp padding; if
            omitted it is inferred from ``len(probs) * frame_duration``.
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    n = probs.shape[0]
    if n == 0:
        return []
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    if audio_duration is None:
        audio_duration = n * frame_duration

    min_speech_frames = max(1, int(round(min_speech_duration / frame_duration)))
    min_silence_frames = max(1, int(round(min_silence_duration / frame_duration)))

    raw: List[List[int]] = []  # [start_frame, end_frame)
    triggered = False
    current_start = 0
    temp_end = 0  # frame index where the current candidate silence began

    for i, p in enumerate(probs):
        if p >= threshold and temp_end:
            temp_end = 0
        if p >= threshold and not triggered:
            triggered = True
            current_start = i
            continue
        if triggered and p < neg_threshold:
            if not temp_end:
                temp_end = i
            if (i - temp_end) < min_silence_frames:
                continue
            if (temp_end - current_start) >= min_speech_frames:
                raw.append([current_start, temp_end])
            temp_end = 0
            triggered = False

    if triggered and (n - current_start) >= min_speech_frames:
        raw.append([current_start, n])

    # split overly long segments
    if max_speech_duration != inf:
        max_frames = max(1, int(round(max_speech_duration / frame_duration)))
        split: List[List[int]] = []
        for s, e in raw:
            while e - s > max_frames:
                split.append([s, s + max_frames])
                s += max_frames
            split.append([s, e])
        raw = split

    # frames -> seconds, apply padding, clamp
    segments: List[SpeechSegment] = []
    for s, e in raw:
        start = max(0.0, s * frame_duration - speech_pad)
        end = min(audio_duration, e * frame_duration + speech_pad)
        if end > start:
            segments.append(SpeechSegment(round(start, 4), round(end, 4)))

    # merge segments that genuinely overlap after padding (strict: touching spans,
    # e.g. those produced by max_speech_duration splitting, stay separate)
    merged: List[SpeechSegment] = []
    for seg in segments:
        if merged and seg.start < merged[-1].end:
            merged[-1] = SpeechSegment(merged[-1].start, max(merged[-1].end, seg.end))
        else:
            merged.append(seg)
    return merged
