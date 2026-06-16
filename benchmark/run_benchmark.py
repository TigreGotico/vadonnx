#!/usr/bin/env python3
"""Run the VAD benchmark across vadonnx backends and write results.json.

    python benchmark/run_benchmark.py [--n-utts 80] [--models silero marblenet fsmn ...]

Builds a real-speech (LibriSpeech) + real-noise (ESC-50) SNR sweep, scores each backend
frame-by-frame against exact reference labels, and records ROC-AUC plus segmented
operating-point metrics (F1 / false-alarm / miss / detection-error-rate).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import datasets as D
import metrics as M
from baselines import load_any

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
SNRS = [None, 20, 15, 10, 5, 0]          # None = clean
DEFAULT_MODELS = ["silero", "marblenet", "marblenet-int8", "fsmn", "fsmn-quant", "speechbrain", "pyannote", "ten",
                  "webrtc", "energy"]


def condition_name(snr):
    return "clean" if snr is None else f"{snr}dB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=80)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    libri_root = os.path.join(DATA, "LibriSpeech", "dev-clean")
    esc_root = os.path.join(DATA, "ESC-50-master")
    print("loading speech + noise ...")
    utts = D.load_librispeech_utts(libri_root, args.n_utts, seed=args.seed)
    noises = D.load_esc50_noises(esc_root, n=80, seed=args.seed + 1)
    total_speech = sum(len(u) for u in utts) / D.SR
    print(f"  {len(utts)} utterances ({total_speech:.0f}s speech), {len(noises)} noise clips")

    # pre-build each SNR condition once (shared across models)
    conditions = {}
    for snr in SNRS:
        audio, intervals = D.build_synthetic(utts, noises, snr, seed=args.seed)
        labels = M.intervals_to_labels(intervals, len(audio) / D.SR)
        conditions[condition_name(snr)] = (audio, labels)
    dur = len(next(iter(conditions.values()))[0]) / D.SR
    print(f"  built {len(conditions)} conditions, {dur:.0f}s each, grid={M.GRID*1000:.0f}ms")

    results = {"meta": {"n_utts": len(utts), "duration_s": dur, "grid_ms": M.GRID * 1000,
                        "speech_frac": float(conditions["clean"][1].mean()),
                        "dataset": "LibriSpeech dev-clean + ESC-50 noise (synthetic SNR sweep)"},
               "models": {}}

    for name in args.models:
        print(f"\n== {name} ==")
        vad = load_any(name)
        per_cond = {}
        for cond, (audio, labels) in conditions.items():
            t0 = time.time()
            probs = vad.probabilities(audio, sample_rate=D.SR)
            grid = M.probs_to_grid(probs, vad.frame_duration, len(labels))
            auc = M.roc_auc(labels, grid)
            segs = vad.get_speech_segments(audio, sample_rate=D.SR)
            pred = M.segments_to_pred(segs, len(labels))
            op = M.operating_point(labels, pred)
            fpr, tpr = M.roc_curve(labels, grid)
            # store a downsampled ROC curve
            idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
            rtf = (time.time() - t0) / dur
            per_cond[cond] = {"auc": auc, **{k: op[k] for k in
                              ("f1", "precision", "recall", "false_alarm", "miss", "der")},
                              "rtf": rtf,
                              "roc": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}}
            print(f"  {cond:>6}: AUC {auc:.4f}  F1 {op['f1']:.3f}  DER {op['der']:.3f}  "
                  f"FA {op['false_alarm']:.3f}  Miss {op['miss']:.3f}  RTF {rtf:.3f}")
        results["models"][name] = per_cond

    out = os.path.join(HERE, "results", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
