"""Deterministic synthetic audio generators for tests (no network, no fixtures).

These let segmentation/probability tests assert on controlled signals without needing
external audio. Everything is seeded so runs are reproducible.
"""
from __future__ import annotations

import numpy as np


def silence(duration: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def tone(
    duration: float, freq: float = 220.0, sample_rate: int = 16000, amp: float = 0.3
) -> np.ndarray:
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def white_noise(
    duration: float, sample_rate: int = 16000, amp: float = 0.3, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def speech_like(
    duration: float, sample_rate: int = 16000, seed: int = 0, amp: float = 0.5
) -> np.ndarray:
    """A crude voiced-sound surrogate: a ~120 Hz pulse train shaped by formant-ish
    sinusoids with a slow amplitude envelope. Not real speech, but energetic and
    structured — useful for exercising segmentation logic deterministically.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 120.0
    sig = np.zeros(n, dtype=np.float64)
    for fmt in (500.0, 1500.0, 2500.0):
        sig += np.sin(2 * np.pi * fmt * t) * np.exp(-((fmt / 3000.0)))
    # voicing pulse train
    sig *= 0.5 + 0.5 * (np.sin(2 * np.pi * f0 * t) > 0.7)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    sig = sig * env + 0.02 * rng.standard_normal(n)
    sig = sig / (np.abs(sig).max() + 1e-9) * amp
    return sig.astype(np.float32)
