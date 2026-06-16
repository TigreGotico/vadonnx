"""Tests for model-reference resolution and the XDG cache helpers."""
import os


from vadonnx import IOSignature
from vadonnx.resolver import (
    bundled_path,
    data_dir,
    get_cache_dir,
    is_onnx_path,
    is_url,
    sidecar_signature,
    xdg_data_home,
)


def test_path_and_url_detection():
    assert is_onnx_path("/a/b/model.onnx")
    assert is_onnx_path("MODEL.ONNX")
    assert not is_onnx_path("silero")
    assert is_url("https://x/m.onnx")
    assert is_url("http://x/m.onnx")
    assert not is_url("/local/m.onnx")


def test_bundled_path():
    assert bundled_path("silero_vad.onnx") == os.path.join(data_dir(), "silero_vad.onnx")
    assert os.path.isfile(bundled_path("silero_vad.onnx"))
    assert bundled_path("nope.onnx") is None


def test_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert xdg_data_home() == str(tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert xdg_data_home().endswith(os.path.join(".local", "share"))


def test_get_cache_dir_created(tmp_path):
    d = get_cache_dir(str(tmp_path / "vc"))
    assert os.path.isdir(d)


def test_sidecar_signature(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    assert sidecar_signature(str(model)) is None
    IOSignature(sample_rate=16000, frame_size=512).save_json(str(tmp_path / "m.signature.json"))
    sig = sidecar_signature(str(model))
    assert sig is not None and sig.frame_size == 512
