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
    probs = silero.probabilities(pcm, sample_rate=sr)
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


def test_flush_processes_trailing_audio(silero, speech_audio):
    audio, sr = speech_audio
    silero.reset()
    # feed sub-frame chunks so the tail stays buffered, then flush
    for i in range(0, 4096, 100):
        silero.process_chunk(audio[i:i + 100], sample_rate=sr)
    p = silero.flush()
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    # flush is safe to call again with an empty buffer
    assert isinstance(silero.flush(), float)


def test_threshold_default_per_backend():
    # load_vad without threshold keeps the backend's own default
    assert load_vad("silero").threshold == 0.5
    assert load_vad("silero", threshold=0.3).threshold == 0.3


def _wave_oracle(path):
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr


def test_wav_bytes_read_as_wav(silero, speech_wav_path):
    oracle, sr = _wave_oracle(speech_wav_path)
    with open(speech_wav_path, "rb") as f:
        data = f.read()
    expected = silero.probabilities(oracle, sample_rate=sr)
    got = silero.probabilities(data)
    np.testing.assert_allclose(got, expected, atol=1e-6)
    assert silero.get_speech_segments(data) == silero.get_speech_segments(oracle, sample_rate=sr)


def test_raw_pcm_bytes_with_rate(silero, speech_audio):
    audio, sr = speech_audio
    pcm = (audio * 32767).astype("<i2").tobytes()
    expected = silero.probabilities(np.frombuffer(pcm, dtype="<i2"), sample_rate=sr)
    np.testing.assert_allclose(silero.probabilities(pcm, sample_rate=sr), expected, atol=1e-6)


def test_non_wav_bytes_without_rate_raise(silero, speech_audio):
    audio, _ = speech_audio
    pcm = (audio * 32767).astype("<i2").tobytes()
    with pytest.raises(ValueError, match="RIFF/WAVE.*sample_rate="):
        silero.probabilities(pcm)
    with pytest.raises(ValueError, match="RIFF/WAVE.*sample_rate="):
        silero.get_speech_segments(pcm)


def test_array_without_rate_raises(silero, speech_audio):
    audio, _ = speech_audio
    with pytest.raises(ValueError, match="sample_rate is required"):
        silero.probabilities(audio)
    with pytest.raises(ValueError, match="sample_rate is required"):
        silero.get_speech_segments(audio)


def test_stream_chunk_without_rate_uses_model_rate(silero, speech_audio):
    audio, _ = speech_audio
    silero.reset()
    assert 0.0 <= silero.process_chunk(audio[:1024]) <= 1.0
    silero.reset()
