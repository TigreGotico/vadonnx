"""Family contract: streaming rate, Hub pinning and the scipy fallback."""
import re
import sys
import types

import numpy as np
import pytest

from vadonnx import audio, resolver
from vadonnx.base import VADModel
from vadonnx.registry import BUILTIN

SHA = re.compile(r"[0-9a-f]{40}")


class _CountingVAD(VADModel):
    def __init__(self):
        self.frames = 0
        super().__init__(sample_rate=16000, frame_size=160)

    def _infer_frame(self, frame_f32):
        self.frames += 1
        return 0.5


def test_process_chunk_without_rate_keeps_model_rate():
    x = np.zeros(1600, dtype=np.float32)
    vad = _CountingVAD()
    vad.process_chunk(x)
    assert vad.frames == 10
    vad = _CountingVAD()
    vad.process_chunk(x, sample_rate=8000)
    assert vad.frames == 20


def test_every_hub_spec_pins_a_commit_sha():
    hub = [s for s in BUILTIN.values() if s.hf_repo]
    assert hub
    for spec in hub:
        assert SHA.fullmatch(spec.revision or ""), f"{spec.name}: {spec.revision!r}"


def test_every_url_spec_pins_a_commit_and_digest():
    for spec in BUILTIN.values():
        if spec.url:
            assert re.search(r"/[0-9a-f]{40}/", spec.url), spec.name
            assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256 or ""), spec.name


def test_resolver_passes_the_pinned_revision(monkeypatch):
    seen = {}
    monkeypatch.setattr(resolver, "hf_download",
                        lambda repo, fn, rev, cache=None: seen.setdefault(repo, rev))
    monkeypatch.setattr(resolver, "bundled_path", lambda name: None)
    hub = [s for s in BUILTIN.values() if s.hf_repo]
    for spec in hub:
        resolver.resolve_spec_file(spec, None, None)
        assert SHA.fullmatch(seen[spec.hf_repo]), spec.name


def test_scipy_runtime_error_propagates(monkeypatch):
    signal = types.ModuleType("scipy.signal")

    def broken(*a, **k):
        raise RuntimeError("scipy broke")

    signal.resample_poly = broken
    scipy = types.ModuleType("scipy")
    scipy.signal = signal
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", signal)
    with pytest.raises(RuntimeError, match="scipy broke"):
        audio.resample(np.zeros(800, dtype=np.float32), 8000, 16000)
