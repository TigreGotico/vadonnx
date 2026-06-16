"""``vadonnx`` command-line interface: list models, probe IO, segment a WAV."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .api import load_vad
from .audio import read_wav
from .registry import get_spec, list_models


def _cmd_list(args) -> int:
    for name in list_models():
        spec = get_spec(name)
        lic = f" [{spec.license}]" if spec and spec.license else ""
        desc = f" — {spec.description}" if spec and spec.description else ""
        print(f"{name}{lic}{desc}")
    return 0


def _cmd_probe(args) -> int:
    import onnxruntime as ort

    from .resolver import resolve_spec_file

    spec = get_spec(args.model)
    if spec is None:
        print(f"unknown model: {args.model}", file=sys.stderr)
        return 2
    path = resolve_spec_file(spec, args.revision, None)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"model: {args.model}  ({path})")
    print("inputs:")
    for i in sess.get_inputs():
        print(f"  {i.name}: {i.shape} {i.type}")
    print("outputs:")
    for o in sess.get_outputs():
        print(f"  {o.name}: {o.shape} {o.type}")
    return 0


def _cmd_segment(args) -> int:
    audio, sr = read_wav(args.wav)
    vad = load_vad(args.model, threshold=args.threshold)
    segments = vad.get_speech_segments(audio, sample_rate=sr)
    if not segments:
        print("no speech detected")
        return 0
    for seg in segments:
        print(f"{seg.start:8.3f} -> {seg.end:8.3f}  ({seg.duration:.3f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vadonnx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list available models").set_defaults(func=_cmd_list)

    pp = sub.add_parser("probe", help="print a model's ONNX input/output signature")
    pp.add_argument("model")
    pp.add_argument("--revision", default=None)
    pp.set_defaults(func=_cmd_probe)

    ps = sub.add_parser("segment", help="print speech segments of a WAV file")
    ps.add_argument("wav")
    ps.add_argument("--model", default="silero")
    ps.add_argument("--threshold", type=float, default=0.5)
    ps.set_defaults(func=_cmd_segment)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
