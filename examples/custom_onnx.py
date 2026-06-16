#!/usr/bin/env python3
"""Load an arbitrary VAD ONNX file by path with an explicit IOSignature.

This example points at the bundled Silero model by path and declares its IO by hand —
the same mechanism you would use for any third-party VAD ONNX.

    python examples/custom_onnx.py path/to/audio.wav
"""
import argparse

from vadonnx import IOSignature, load_vad
from vadonnx.audio import read_wav
from vadonnx.resolver import bundled_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    args = ap.parse_args()

    signature = IOSignature(
        sample_rate=16000,
        frame_size=512,
        context_size=64,
        stateful=True,
        audio_input="input",
        audio_layout="BT",
        state_inputs={"state": (2, 1, 128)},
        extra_inputs={"sr": ("int64_scalar", 16000)},
        state_output_map={"state": "stateN"},
        prob_output="output",
        prob_extract="scalar",
    )

    model_path = bundled_path("silero_vad.onnx")
    vad = load_vad(model_path, signature=signature)

    audio, sr = read_wav(args.wav)
    for seg in vad.get_speech_segments(audio, sample_rate=sr):
        print(f"{seg.start:7.2f}s -> {seg.end:7.2f}s")


if __name__ == "__main__":
    main()
