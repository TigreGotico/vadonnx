"""Frame-grid VAD metrics: ROC-AUC plus operating-point precision/recall/F1 and the
diarization-style detection error rate (false-alarm + miss over speech time).

Everything is resampled onto a common 10 ms grid so models with different native frame
rates (Silero 32 ms, FSMN 10 ms, MarbleNet 20 ms) are compared fairly against the same
reference labels.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

GRID = 0.01  # 10 ms reference grid


def intervals_to_labels(intervals: Sequence[Tuple[float, float]], duration: float,
                        grid: float = GRID) -> np.ndarray:
    """Binary speech label per grid frame (1 = speech), by frame-center membership."""
    n = int(duration / grid)
    centers = (np.arange(n) + 0.5) * grid
    labels = np.zeros(n, dtype=np.int8)
    for a, b in intervals:
        labels[(centers >= a) & (centers < b)] = 1
    return labels


def probs_to_grid(probs: np.ndarray, frame_duration: float, n_grid: int,
                  grid: float = GRID) -> np.ndarray:
    """Resample a per-frame probability array onto the common grid (nearest frame)."""
    if probs.size == 0:
        return np.zeros(n_grid, dtype=np.float32)
    grid_centers = (np.arange(n_grid) + 0.5) * grid
    src_idx = np.clip((grid_centers / frame_duration).astype(int), 0, probs.size - 1)
    return probs[src_idx].astype(np.float32)


def roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if labels.min() == labels.max():  # single class — AUC undefined
        return float("nan")
    return float(roc_auc_score(labels, probs))


def roc_curve(labels: np.ndarray, probs: np.ndarray):
    from sklearn.metrics import roc_curve as _rc

    if labels.min() == labels.max():
        return np.array([0, 1]), np.array([0, 1])
    fpr, tpr, _ = _rc(labels, probs)
    return fpr, tpr


def operating_point(labels: np.ndarray, pred: np.ndarray) -> dict:
    """Frame-level precision/recall/F1 + false-alarm/miss rates + detection error rate.

    ``pred`` is a binary per-frame decision (already thresholded / segmented).
    DER follows the detection convention: (false_alarm + miss) / total_speech_frames.
    """
    labels = labels.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    tn = int(np.sum(~pred & ~labels))
    n_speech = tp + fn
    n_nonspeech = fp + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_speech if n_speech else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fa_rate = fp / n_nonspeech if n_nonspeech else 0.0       # false alarm (on non-speech)
    miss_rate = fn / n_speech if n_speech else 0.0           # miss (on speech)
    der = (fp + fn) / n_speech if n_speech else float("nan")  # detection error rate
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "false_alarm": fa_rate, "miss": miss_rate, "der": der, "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def segments_to_pred(segments, n_grid: int, grid: float = GRID) -> np.ndarray:
    """Binary per-grid-frame prediction from detected SpeechSegments."""
    pred = np.zeros(n_grid, dtype=np.int8)
    centers = (np.arange(n_grid) + 0.5) * grid
    for seg in segments:
        pred[(centers >= seg.start) & (centers < seg.end)] = 1
    return pred
