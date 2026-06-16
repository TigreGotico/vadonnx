"""TEN VAD backend.

The TEN VAD ONNX graph consumes a precomputed ``[B, 3, 41]`` mel+pitch feature tensor
and four recurrent state tensors. Feature extraction is implemented in TEN's native
library and is not reproduced here, so the model is not driven through the vadonnx ONNX
API; constructing it raises :class:`NotImplementedError`. The ONNX file and its signature
are published for use with TEN's own feature pipeline.
"""
from __future__ import annotations

from ..onnx_backend import OnnxVAD

_MSG = (
    "The 'ten' backend cannot run through vadonnx's ONNX path: TEN VAD's mel+pitch "
    "feature extraction is provided by its native library, which vadonnx does not "
    "reproduce. The ONNX and signature are published at TigreGotico/ten-vad-onnx for use "
    "with TEN's feature pipeline. For a self-contained model use load_vad('silero'), "
    "load_vad('marblenet') or load_vad('pyannote')."
)


class TenVAD(OnnxVAD):
    """Backend for TEN VAD (see module docstring)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)
