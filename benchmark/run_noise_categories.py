#!/usr/bin/env python3
"""Per-noise-category robustness: score each backend at a fixed SNR against each of the
five ESC-50 macro noise categories (animals, nature/water, human non-speech,
interior/domestic, exterior/urban). Reveals which noise types break which model.

    python benchmark/run_noise_categories.py [--snr 5] [--n-utts 60]
"""
from __future__ import annotations

import argparse
import json
import os

from baselines import load_any
import datasets as D
import metrics as M

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
DEFAULT_MODELS = ["silero", "marblenet", "fsmn", "speechbrain", "pyannote", "webrtc", "energy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--n-utts", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    utts = D.load_librispeech_utts(os.path.join(DATA, "LibriSpeech", "dev-clean"),
                                   args.n_utts, seed=args.seed)
    by_macro = D.load_esc50_by_macro(os.path.join(DATA, "ESC-50-master"))
    print(f"{len(utts)} utts; noise macros: " +
          ", ".join(f"{k}({len(v)})" for k, v in by_macro.items()))

    conditions = {}
    for macro, noises in by_macro.items():
        audio, intervals = D.build_synthetic(utts, noises, args.snr, seed=args.seed)
        conditions[macro] = (audio, M.intervals_to_labels(intervals, len(audio) / D.SR))

    results = {"meta": {"snr_db": args.snr, "n_utts": len(utts),
                        "categories": list(by_macro)}, "models": {}}
    for name in args.models:
        vad = load_any(name)
        row = {}
        for macro, (audio, labels) in conditions.items():
            grid = M.probs_to_grid(vad.probabilities(audio, sample_rate=D.SR),
                                   vad.frame_duration, len(labels))
            row[macro] = M.roc_auc(labels, grid)
            print(f"  {name:>10} / {macro:<18}: AUC {row[macro]:.4f}")
        results["models"][name] = row

    out = os.path.join(HERE, "results", "results_noise.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
