#!/usr/bin/env python3
"""Live microphone VAD: print a speech/silence indicator as you talk.

Requires the `mic` extra:  uv pip install "vadonnx[mic]"

    python examples/streaming_mic.py [--model silero]
"""
import argparse


from vadonnx import load_vad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="silero")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("install the mic extra:  uv pip install 'vadonnx[mic]'")

    vad = load_vad(args.model, threshold=args.threshold)
    vad.reset()
    block = vad.frame_size  # one model frame per callback

    print("listening... (Ctrl-C to stop)")
    with sd.InputStream(channels=1, samplerate=vad.sample_rate, blocksize=block,
                        dtype="float32") as stream:
        try:
            while True:
                data, _ = stream.read(block)
                prob = vad.process_chunk(data[:, 0], sample_rate=vad.sample_rate)
                bar = "#" * int(prob * 40)
                state = "SPEECH" if prob >= args.threshold else "  ..  "
                print(f"\r{state} |{bar:<40}| {prob:.2f}", end="", flush=True)
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
