#!/usr/bin/env python3
"""Register a custom model by name, then load it like a built-in.

Shows the `register_model` / `ModelSpec` / `IOSignature` path — here re-registering the
bundled Silero model under a new name, but the same applies to any ONNX VAD on disk or
a HuggingFace repo.

    python examples/register_model.py path/to/audio.wav
"""
import argparse

from vadonnx import IOSignature, ModelSpec, list_models, load_vad, register_model
from vadonnx.audio import read_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    args = ap.parse_args()

    register_model(ModelSpec(
        name="my-silero",
        signature=IOSignature(
            sample_rate=16000, frame_size=512, context_size=64, stateful=True,
            audio_input="input", audio_layout="BT",
            state_inputs={"state": (2, 1, 128)},
            extra_inputs={"sr": ("int64_scalar", 16000)},
            state_output_map={"state": "stateN"},
            prob_output="output", prob_extract="scalar",
        ),
        # bundled file shipped in the wheel; use hf_repo=... + filename=... for a download
        bundled="silero_vad.onnx",
    ))

    print("registered; now in catalogue:", "my-silero" in list_models())
    audio, sr = read_wav(args.wav)
    vad = load_vad("my-silero")
    print("segments:", [(round(s.start, 1), round(s.end, 1))
                        for s in vad.get_speech_segments(audio, sample_rate=sr)])


if __name__ == "__main__":
    main()
