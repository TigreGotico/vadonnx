import numpy as np
import pytest

from vadonnx.audio import read_wav, resample, to_float32_mono


def test_int16_bytes_roundtrip():
    src = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    out = to_float32_mono(src.tobytes())
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, src.astype(np.float32) / 32768.0, atol=1e-6)


def test_int16_array_and_float_passthrough():
    i16 = np.array([0, 32767, -32768], dtype=np.int16)
    np.testing.assert_allclose(to_float32_mono(i16), [0, 32767 / 32768, -1.0], atol=1e-6)
    f = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(to_float32_mono(f), f, atol=1e-7)


def test_int32_and_uint8_scaling():
    i32 = np.array([0, 2147483647, -2147483648], dtype=np.int32)
    out = to_float32_mono(i32)
    assert -1.001 <= out.min() and out.max() <= 1.001
    u8 = np.array([0, 128, 255], dtype=np.uint8)
    np.testing.assert_allclose(to_float32_mono(u8), [-1.0, 0.0, 127 / 128], atol=1e-6)


def test_stereo_downmix():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(to_float32_mono(stereo), [0.0, 0.5], atol=1e-6)


def test_bad_dtype_rejected():
    with pytest.raises(TypeError):
        to_float32_mono(np.array([1 + 2j], dtype=np.complex64))


def test_resample_length_and_noop():
    x = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    assert resample(x, 16000, 16000) is x or len(resample(x, 16000, 16000)) == len(x)
    y = resample(x, 16000, 8000)
    assert abs(len(y) - 8000) <= 1
    z = resample(x, 8000, 16000)
    assert abs(len(z) - 32000) <= 1


def test_read_wav(speech_wav_path):
    audio, sr = read_wav(speech_wav_path)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.abs(audio).max() <= 1.0
    assert len(audio) / sr > 5  # ~11s clip
