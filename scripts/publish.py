#!/usr/bin/env python3
"""Publish converted VAD ONNX models to the TigreGotico HuggingFace org.

Usage:
    python scripts/publish.py [silero ten fsmn marblenet ...]

With no arguments, publishes every model that can be converted in the current
environment. `marblenet` requires the `convert-marblenet` extra (nemo_toolkit + torch);
`ten`/`fsmn`/`silero` need only the `convert` extra.

Prints the resulting commit SHAs so they can be pinned into vadonnx/registry.py.
"""
import sys

from vadonnx.convert import fsmn, marblenet, pyannote, silero, speechbrain, ten

CONVERTERS = {"silero": silero, "ten": ten, "fsmn": fsmn,
              "marblenet": marblenet, "speechbrain": speechbrain, "pyannote": pyannote}


def main(argv):
    names = argv or list(CONVERTERS)
    results = {}
    for name in names:
        mod = CONVERTERS[name]
        try:
            results[name] = mod.publish()
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} failed: {type(e).__name__}: {e}")
    print("\n# pin these revisions in vadonnx/registry.py:")
    for name, sha in results.items():
        print(f"#   {name}: {sha}")


if __name__ == "__main__":
    main(sys.argv[1:])
