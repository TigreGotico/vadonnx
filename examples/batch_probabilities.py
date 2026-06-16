#!/usr/bin/env python3
"""Print summary statistics of per-frame speech probabilities for a WAV file.

    python examples/batch_probabilities.py path/to/audio.wav
"""
import argparse


from vadonnx import load_vad
from vadonnx.audio import read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="silero")
    args = ap.parse_args()

    audio, sr = read_wav(args.wav)
    vad = load_vad(args.model)
    probs = vad.probabilities(audio, sample_rate=sr)

    print(f"{args.model}: {len(probs)} frames @ {vad.frame_duration*1000:.0f} ms")
    print(f"  min={probs.min():.3f}  mean={probs.mean():.3f}  max={probs.max():.3f}")
    print(f"  frames above 0.5: {(probs >= 0.5).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
