"""Load the bundled Silero ONNX by raw path + explicit signature (the custom-model path).

Verifies that `load_vad("<path>.onnx", signature=...)` reproduces the named model.
"""
import numpy as np

from vadonnx import IOSignature, load_vad
from vadonnx.resolver import bundled_path


def _silero_signature():
    return IOSignature(
        sample_rate=16000, frame_size=512, context_size=64, stateful=True,
        audio_input="input", audio_layout="BT",
        state_inputs={"state": (2, 1, 128)},
        extra_inputs={"sr": ("int64_scalar", 16000)},
        state_output_map={"state": "stateN"},
        prob_output="output", prob_extract="scalar",
    )


def test_custom_path_matches_named(speech_audio):
    audio, sr = speech_audio
    path = bundled_path("silero_vad.onnx")
    assert path is not None

    custom = load_vad(path, signature=_silero_signature())
    named = load_vad("silero")

    pc = custom.probabilities(audio, sample_rate=sr)
    pn = named.probabilities(audio, sample_rate=sr)
    np.testing.assert_allclose(pc, pn, atol=1e-6)


def test_dict_signature_accepted(speech_audio):
    audio, sr = speech_audio
    path = bundled_path("silero_vad.onnx")
    custom = load_vad(path, signature=_silero_signature().to_dict())
    assert custom.probabilities(audio, sample_rate=sr).max() > 0.8


def test_missing_signature_errors():
    import pytest

    path = bundled_path("silero_vad.onnx")
    with pytest.raises(ValueError):
        load_vad(path)  # raw .onnx with no signature / sidecar


def test_missing_file_errors():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_vad("/no/such/model.onnx", signature=_silero_signature())
