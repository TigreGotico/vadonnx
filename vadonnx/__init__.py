"""vadonnx — load arbitrary Voice Activity Detection models behind a unified ONNX API.

Quick start::

    from vadonnx import load_vad

    vad = load_vad("silero")                       # bundled, works offline
    prob = vad.process_chunk(pcm_bytes)            # streaming, float in [0, 1]
    segments = vad.get_speech_segments(audio, sample_rate=16000)
"""
from .api import list_models, load_vad, register_model
from .base import VADModel
from .registry import ModelSpec
from .segment import SpeechSegment, probs_to_segments
from .signature import IOSignature
from .version import __version__

__all__ = [
    "load_vad",
    "list_models",
    "register_model",
    "VADModel",
    "IOSignature",
    "ModelSpec",
    "SpeechSegment",
    "probs_to_segments",
    "__version__",
]
