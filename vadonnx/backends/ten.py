"""TEN VAD backend (experimental / reference only).

TEN VAD's ONNX expects a precomputed ``[B, 3, 41]`` mel+pitch feature tensor plus four
recurrent state tensors. Feature extraction lives in TEN's **native C library**, which
``vadonnx`` does not reproduce — so there is no faithful pure-ONNX raw-PCM path. The
ONNX and its discovered signature are published on the TigreGotico HF org for
experimentation, but loading ``ten`` raises a clear error rather than returning
incorrect probabilities. Use ``silero`` or ``marblenet`` instead. See docs/backends.md.
"""
from __future__ import annotations

from ..onnx_backend import OnnxVAD

_MSG = (
    "The 'ten' backend is experimental and not functional as a pure-ONNX model: TEN VAD's "
    "mel+pitch feature extraction lives in its native C library, which vadonnx does not "
    "reproduce. The ONNX + signature are published at TigreGotico/ten-vad-onnx for "
    "experimentation. Use load_vad('silero') or load_vad('marblenet') instead. "
    "See https://github.com/TigreGotico/vadonnx/blob/dev/docs/backends.md"
)


class TenVAD(OnnxVAD):
    """Reference-only TEN VAD backend; raises with guidance (see module docstring)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)
