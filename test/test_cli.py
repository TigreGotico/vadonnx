"""CLI tests — exercise `vadonnx list|probe|segment` offline on the bundled model."""
from vadonnx.cli import main


def test_list(capsys):
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    for name in ("silero", "marblenet", "pyannote", "speechbrain"):
        assert name in out


def test_probe_silero(capsys):
    rc = main(["probe", "silero"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "inputs:" in out and "outputs:" in out
    assert "input" in out and "state" in out


def test_segment(capsys, speech_wav_path):
    rc = main(["segment", speech_wav_path])
    out = capsys.readouterr().out
    assert rc == 0
    # the JFK clip has several speech spans
    assert out.count("->") >= 3


def test_segment_custom_threshold(capsys, speech_wav_path):
    rc = main(["segment", speech_wav_path, "--model", "silero", "--threshold", "0.6"])
    assert rc == 0


def test_probe_unknown_model():
    assert main(["probe", "does-not-exist"]) == 2
