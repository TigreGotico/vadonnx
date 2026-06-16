#!/usr/bin/env python3
"""Score backends on VoxConverse dev — real multi-speaker conversational audio with
RTTM speech references (in-the-wild VAD). Averages per-file metrics.

    python benchmark/run_voxconverse.py [--max-files 40]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from baselines import load_any
import datasets as D
import metrics as M

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
DEFAULT_MODELS = ["silero", "marblenet", "fsmn", "speechbrain", "pyannote", "webrtc", "energy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    rttm_dir = os.path.join(DATA, "voxconverse-master", "dev")
    audio_dir = os.path.join(DATA)  # wavs extracted under data/audio or data/voxconverse_test_wav
    clips = D.load_voxconverse(rttm_dir, audio_dir, max_files=args.max_files)
    if not clips:
        raise SystemExit("no VoxConverse clips found (check download/extraction)")
    total = sum(len(a) for _, a, _ in clips) / D.SR
    speech_frac = np.mean([sum(b - a for a, b in iv) / (len(au) / D.SR)
                           for _, au, iv in clips])
    print(f"{len(clips)} clips, {total/60:.0f} min audio, {speech_frac*100:.0f}% speech")

    results = {"meta": {"dataset": "VoxConverse dev (real conversational, RTTM refs)",
                        "n_files": len(clips), "minutes": total / 60,
                        "speech_frac": float(speech_frac)}, "models": {}}
    for name in args.models:
        vad = load_any(name)
        aucs, f1s, ders, fas, misses = [], [], [], [], []
        for fid, audio, intervals in clips:
            dur = len(audio) / D.SR
            labels = M.intervals_to_labels(intervals, dur)
            if labels.min() == labels.max():
                continue
            grid = M.probs_to_grid(vad.probabilities(audio, sample_rate=D.SR),
                                   vad.frame_duration, len(labels))
            aucs.append(M.roc_auc(labels, grid))
            pred = M.segments_to_pred(vad.get_speech_segments(audio, sample_rate=D.SR), len(labels))
            op = M.operating_point(labels, pred)
            f1s.append(op["f1"])
            ders.append(op["der"])
            fas.append(op["false_alarm"])
            misses.append(op["miss"])
        results["models"][name] = {
            "auc": float(np.mean(aucs)), "f1": float(np.mean(f1s)),
            "der": float(np.mean(ders)), "false_alarm": float(np.mean(fas)),
            "miss": float(np.mean(misses)), "n": len(aucs),
        }
        print(f"  {name:>10}: AUC {np.mean(aucs):.4f}  F1 {np.mean(f1s):.3f}  "
              f"DER {np.mean(ders):.3f}  FA {np.mean(fas):.3f}  Miss {np.mean(misses):.3f}")

    out = os.path.join(HERE, "results", "results_vox.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
