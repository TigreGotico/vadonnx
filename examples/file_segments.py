#!/usr/bin/env python3
"""Print the speech segments of a WAV file.

    python examples/file_segments.py path/to/audio.wav [--model silero]
"""
import argparse

from vadonnx import load_vad
from vadonnx.audio import read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="silero")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    audio, sr = read_wav(args.wav)
    vad = load_vad(args.model, threshold=args.threshold)
    segments = vad.get_speech_segments(audio, sample_rate=sr)

    print(f"{args.model}: {len(segments)} speech segment(s) in {len(audio)/sr:.1f}s")
    for seg in segments:
        print(f"  {seg.start:7.2f}s -> {seg.end:7.2f}s  ({seg.duration:.2f}s)")


if __name__ == "__main__":
    main()
