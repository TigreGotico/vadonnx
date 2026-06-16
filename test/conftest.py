"""Shared pytest fixtures."""
import os

import pytest

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="session")
def speech_wav_path():
    return os.path.join(HERE, "resources", "speech.wav")


@pytest.fixture(scope="session")
def speech_audio(speech_wav_path):
    from vadonnx.audio import read_wav

    audio, sr = read_wav(speech_wav_path)
    return audio, sr


@pytest.fixture(scope="session")
def silero(speech_audio):
    """The bundled Silero model — loads fully offline (no download)."""
    from vadonnx import load_vad

    return load_vad("silero")
