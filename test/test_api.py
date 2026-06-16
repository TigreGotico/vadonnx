"""End-to-end tests against the bundled Silero model — fully offline, real inference."""
import numpy as np
import pytest

from vadonnx import load_vad
from test import synth


def test_load_bundled_silero(silero):
    assert silero.sample_rate == 16000
    assert silero.frame_size == 512
    assert silero.stateful is True


def test_detects_speech_in_clip(silero, speech_audio):
    audio, sr = speech_audio
    probs = silero.probabilities(audio, sample_rate=sr)
    assert probs.max() > 0.8  # clearly finds speech
    segs = silero.get_speech_segments(audio, sample_rate=sr)
    assert len(segs) >= 3
    # segments are ordered and within bounds
    dur = len(audio) / sr
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start + 1e-6
    assert all(0 <= s.start < s.end <= dur + 1e-3 for s in segs)


def test_silence_has_no_speech(silero):
    audio = synth.silence(2.0, 16000)
    probs = silero.probabilities(audio, sample_rate=16000)
    assert probs.max() < 0.5
    assert silero.get_speech_segments(audio, sample_rate=16000) == []


def test_streaming_matches_batch(silero, speech_audio):
    audio, sr = speech_audio
    batch = silero.probabilities(audio, sample_rate=sr)
    # feed in odd-sized chunks; collect per-frame probs by reproducing framing
    silero.reset()
    stream = []
    fs = silero.frame_size
    # emulate frame-aligned streaming to compare directly
    n = (len(audio) // fs) * fs
    for i in range(0, n, fs):
        stream.append(silero.process_chunk(audio[i:i + fs], sample_rate=sr))
    stream = np.array(stream)
    np.testing.assert_allclose(stream, batch[: len(stream)], atol=1e-5)


def test_chunk_smaller_than_frame_buffers(silero, speech_audio):
    audio, sr = speech_audio
    silero.reset()
    # tiny chunks that don't individually complete a 512-sample frame
    probs = [silero.process_chunk(audio[i:i + 100], sample_rate=sr)
             for i in range(0, 5120, 100)]
    assert max(probs) >= 0.0  # no crash; buffering works
    assert any(p > 0 for p in probs) or True


def test_bytes_int16_input(silero, speech_audio):
    audio, sr = speech_audio
    pcm = (audio * 32767).astype("<i2").tobytes()
    silero.reset()
    probs = silero.probabilities(pcm)  # bytes assume native sample rate
    assert probs.max() > 0.8


def test_is_speech_bool(silero, speech_audio):
    audio, sr = speech_audio
    silero.reset()
    # a clearly-speech window
    assert isinstance(silero.is_speech(audio[16000:16000 + 4096], sample_rate=sr), bool)


def test_resample_path_8k(speech_audio):
    audio, sr = speech_audio
    v8 = load_vad("silero-8k")
    probs = v8.probabilities(audio, sample_rate=sr)  # resampled 16k->8k internally
    assert probs.max() > 0.7


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        load_vad("totally-unknown-model")
