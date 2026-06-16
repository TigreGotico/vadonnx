"""Network-dependent tests for HuggingFace / URL model resolution.

These are skipped unless ``VADONNX_NETWORK_TESTS=1`` is set, so the default suite
(and CI) runs fully offline against the bundled model. This is an opt-in network
gate, not a missing-dependency skip.
"""
import os

import pytest

NETWORK = os.environ.get("VADONNX_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(not NETWORK, reason="set VADONNX_NETWORK_TESTS=1 to run")


def test_download_silero_from_hf(tmp_path, speech_audio):
    """Force a real HF download by ignoring the bundled copy."""
    from vadonnx import load_vad
    from vadonnx.registry import BUILTIN, ModelSpec, register_model

    spec = BUILTIN["silero"]
    # a spec without `bundled` forces the HF path
    register_model(ModelSpec(
        name="silero-hf-test", signature=spec.signature,
        hf_repo=spec.hf_repo, filename=spec.filename, license="MIT",
    ))
    audio, sr = speech_audio
    vad = load_vad("silero-hf-test", cache_dir=str(tmp_path))
    assert vad.probabilities(audio, sample_rate=sr).max() > 0.8


@pytest.mark.parametrize("name", ["fsmn", "marblenet", "marblenet-int8",
                                  "speechbrain", "silero-op15", "pyannote", "pyannote-int8"])
def test_remote_backends_detect_speech(name, speech_audio, tmp_path):
    """Download each published model and confirm it detects speech in the clip."""
    from vadonnx import load_vad

    audio, sr = speech_audio
    vad = load_vad(name, cache_dir=str(tmp_path))
    probs = vad.probabilities(audio, sample_rate=sr)
    assert probs.max() > 0.8
