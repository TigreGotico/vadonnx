#!/usr/bin/env python3
"""Compare speech segments from several backends on the same file.

    python examples/compare_backends.py path/to/audio.wav --models silero ten fsmn

Models other than the bundled `silero` are downloaded from the TigreGotico HF org on
first use. Backends that fail to load (missing model, environment caveats) are reported
and skipped.
"""
import argparse

from vadonnx import load_vad
from vadonnx.audio import read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--models", nargs="+", default=["silero", "fsmn", "marblenet"])
    args = ap.parse_args()

    audio, sr = read_wav(args.wav)
    for name in args.models:
        try:
            vad = load_vad(name)
            probs = vad.probabilities(audio, sample_rate=sr)
            segs = vad.get_speech_segments(audio, sample_rate=sr)
            spans = ", ".join(f"{s.start:.1f}-{s.end:.1f}" for s in segs)
            print(f"{name:10s} max={probs.max():.2f} mean={probs.mean():.2f} "
                  f"segments=[{spans}]")
        except Exception as e:
            print(f"{name:10s} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
