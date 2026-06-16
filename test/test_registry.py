import pytest

from vadonnx import IOSignature, ModelSpec, list_models, register_model
from vadonnx.registry import get_backend, get_spec


def test_builtins_present():
    names = list_models()
    for n in ("silero", "silero-8k", "ten", "fsmn", "marblenet", "speechbrain", "pyannote"):
        assert n in names


def test_get_spec_builtin():
    spec = get_spec("silero")
    assert spec is not None
    assert spec.bundled == "silero_vad.onnx"
    assert spec.signature.sample_rate == 16000


def test_unknown_spec():
    assert get_spec("does-not-exist") is None


def test_register_custom_model():
    spec = ModelSpec(
        name="my-test-vad",
        signature=IOSignature(sample_rate=16000, frame_size=512),
        bundled="silero_vad.onnx",
    )
    register_model(spec)
    assert "my-test-vad" in list_models()
    assert get_spec("my-test-vad") is spec


def test_get_backend_onnx():
    from vadonnx.onnx_backend import OnnxVAD

    assert get_backend("onnx") is OnnxVAD


def test_get_backend_dotted_path():
    cls = get_backend("vadonnx.onnx_backend:OnnxVAD")
    from vadonnx.onnx_backend import OnnxVAD

    assert cls is OnnxVAD


def test_get_backend_unknown():
    with pytest.raises(KeyError):
        get_backend("nope")
