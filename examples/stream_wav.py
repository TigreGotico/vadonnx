#!/usr/bin/env python3
"""Stream a WAV file through the VAD chunk-by-chunk (simulating live audio).

Feeds arbitrary-sized chunks to `process_chunk` and prints when speech starts/stops,
demonstrating the streaming API and `reset()` between runs.

    python examples/stream_wav.py path/to/audio.wav [--model silero] [--chunk-ms 100]
"""
import argparse

from vadonnx import load_vad
from vadonnx.audio import read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--model", default="silero")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    audio, sr = read_wav(args.wav)
    vad = load_vad(args.model, threshold=args.threshold)
    vad.reset()

    hop = int(sr * args.chunk_ms / 1000)
    speaking = False
    t = 0.0
    for i in range(0, len(audio), hop):
        prob = vad.process_chunk(audio[i:i + hop], sample_rate=sr)
        if prob >= args.threshold and not speaking:
            print(f"  speech start @ {t:6.2f}s (p={prob:.2f})")
            speaking = True
        elif prob < args.threshold and speaking:
            print(f"  speech end   @ {t:6.2f}s (p={prob:.2f})")
            speaking = False
        t += hop / sr
    if speaking:
        print(f"  speech end   @ {t:6.2f}s (eof)")


if __name__ == "__main__":
    main()
