"""Declarative description of a VAD ONNX model's input/output wiring.

An :class:`IOSignature` is *data* that lets the generic :class:`vadonnx.onnx_backend.OnnxVAD`
engine drive almost any VAD ONNX graph without bespoke code. It captures the sample
rate and frame size the model expects, how the audio (or feature) tensor is named and
shaped, which inputs/outputs carry recurrent state, any constant extra inputs (e.g.
Silero's ``sr`` int64 scalar), and how the speech probability is extracted from the
output tensor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Union


# how an extra (constant) input is encoded: (kind, value)
#   kind one of: "int64_scalar", "int64", "float_scalar", "float"
ExtraInput = tuple


@dataclass
class IOSignature:
    """Declarative IO wiring for a VAD ONNX model.

    Args:
        sample_rate: native sample rate the model expects (Hz).
        frame_size: number of audio samples consumed per inference step (the hop).
        context_size: number of trailing samples of the previous frame to prepend to
            the current one before inference (Silero feeds ``context + frame``). The
            fed audio length is ``context_size + frame_size``. Reset to zeros.
        stateful: whether the model carries recurrent state between frames.
        feature: optional feature stage applied before inference.
            ``None`` means raw PCM float32 is fed directly; ``"logmel"``, ``"fbank"`` or
            ``"fbank_cmvn"`` run :mod:`vadonnx.features` first.
        feature_params: kwargs forwarded to the feature function.
        audio_input: name of the ONNX input that receives audio/features.
        audio_layout: how the audio/feature array is shaped before feeding:
            ``"BT"`` -> ``(1, frame_size)``; ``"T"`` -> ``(frame_size,)``;
            ``"BFT"`` -> ``(1, n_feat, n_frames)``; ``"BTF"`` -> ``(1, n_frames, n_feat)``.
        state_inputs: ``{input_name: shape}`` for recurrent-state tensors. They are
            allocated as float32 zeros on reset.
        extra_inputs: ``{input_name: (kind, value)}`` constant inputs fed every step.
        state_output_map: ``{state_input_name: output_name}`` mapping each state input
            to the model output that produces its next value.
        prob_output: name (str) or index (int) of the output holding the speech score.
        prob_extract: how to reduce the probability output to a scalar:
            ``"scalar"`` (squeeze to one value), ``"last"`` (last frame),
            ``"mean"`` (mean over frames), ``"index:N"`` (class N of a softmax),
            ``"1-minus:N"`` (1 - class N, e.g. when class 0 is *non-speech*).
        multiclass_collapse: for multi-class outputs, the list of class indices that
            represent *speech*; their summed probability is returned.
        license: SPDX-ish license string, surfaced to users.
    """

    sample_rate: int
    frame_size: int
    context_size: int = 0
    stateful: bool = False
    feature: Optional[str] = None
    feature_params: dict = field(default_factory=dict)
    audio_input: str = "input"
    audio_layout: str = "BT"
    state_inputs: dict = field(default_factory=dict)
    extra_inputs: dict = field(default_factory=dict)
    state_output_map: dict = field(default_factory=dict)
    prob_output: Union[str, int] = 0
    prob_extract: str = "scalar"
    multiclass_collapse: Optional[list] = None
    license: str = ""

    def __post_init__(self):
        # normalise shapes/tuples that may arrive as lists (e.g. from JSON)
        self.state_inputs = {k: tuple(v) for k, v in self.state_inputs.items()}
        self.extra_inputs = {k: tuple(v) for k, v in self.extra_inputs.items()}
        if self.audio_layout not in ("BT", "T", "BFT", "BTF"):
            raise ValueError(f"unknown audio_layout: {self.audio_layout!r}")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict) -> "IOSignature":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "IOSignature":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


def coerce_signature(sig: Union[IOSignature, dict, None]) -> Optional[IOSignature]:
    """Accept an :class:`IOSignature`, a plain dict, or ``None``."""
    if sig is None or isinstance(sig, IOSignature):
        return sig
    if isinstance(sig, dict):
        return IOSignature.from_dict(sig)
    raise TypeError(f"signature must be IOSignature | dict | None, got {type(sig)}")
