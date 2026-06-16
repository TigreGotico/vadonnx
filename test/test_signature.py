import json

from vadonnx.signature import IOSignature, coerce_signature


def test_roundtrip_dict():
    sig = IOSignature(
        sample_rate=16000, frame_size=512, context_size=64, stateful=True,
        state_inputs={"state": (2, 1, 128)},
        extra_inputs={"sr": ("int64_scalar", 16000)},
        state_output_map={"state": "stateN"},
        prob_output="output", prob_extract="scalar",
    )
    d = sig.to_dict()
    back = IOSignature.from_dict(d)
    assert back == sig
    # tuples survive a JSON-ish round-trip (lists coerced back to tuples)
    relaxed = IOSignature.from_dict(json.loads(json.dumps(d)))
    assert relaxed.state_inputs["state"] == (2, 1, 128)
    assert relaxed.extra_inputs["sr"] == ("int64_scalar", 16000)


def test_save_and_load(tmp_path):
    sig = IOSignature(sample_rate=8000, frame_size=256)
    p = tmp_path / "sig.json"
    sig.save_json(str(p))
    loaded = IOSignature.from_json(str(p))
    assert loaded == sig


def test_coerce():
    sig = IOSignature(sample_rate=16000, frame_size=512)
    assert coerce_signature(sig) is sig
    assert coerce_signature(None) is None
    assert coerce_signature({"sample_rate": 16000, "frame_size": 512}) == sig


def test_bad_layout_rejected():
    import pytest

    with pytest.raises(ValueError):
        IOSignature(sample_rate=16000, frame_size=512, audio_layout="ZZ")


def test_extra_keys_ignored():
    sig = IOSignature.from_dict({"sample_rate": 16000, "frame_size": 512, "junk": 1})
    assert sig.sample_rate == 16000
