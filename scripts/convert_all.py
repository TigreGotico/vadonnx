#!/usr/bin/env python3
"""Convert (but do not publish) VAD models locally for inspection/verification.

    python scripts/convert_all.py [silero ten fsmn marblenet ...]
"""
import sys

from vadonnx.convert import fsmn, marblenet, silero, ten

CONVERTERS = {"silero": silero, "ten": ten, "fsmn": fsmn, "marblenet": marblenet}


def main(argv):
    for name in (argv or list(CONVERTERS)):
        print(f"== converting {name} ==")
        try:
            print(CONVERTERS[name].convert())
        except Exception as e:  # noqa: BLE001
            print(f"!! {name} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
