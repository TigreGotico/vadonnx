import numpy as np

from vadonnx.segment import SpeechSegment, probs_to_segments


def test_empty():
    assert probs_to_segments(np.zeros(0), 0.032) == []


def test_all_silence():
    probs = np.zeros(100, dtype=np.float32)
    assert probs_to_segments(probs, 0.032) == []


def test_single_block():
    # 1s frames of 32ms: speech in the middle
    fd = 0.032
    probs = np.zeros(100, dtype=np.float32)
    probs[30:60] = 0.9
    segs = probs_to_segments(probs, fd, min_speech_duration=0.1, speech_pad=0.0)
    assert len(segs) == 1
    assert abs(segs[0].start - 30 * fd) < 1e-6
    assert abs(segs[0].end - 60 * fd) < 1e-6


def test_hysteresis_keeps_segment_through_dip():
    fd = 0.032
    probs = np.full(100, 0.9, dtype=np.float32)
    probs[50] = 0.4  # single-frame dip above neg_threshold-ish but below thresh
    segs = probs_to_segments(probs, fd, threshold=0.5, neg_threshold=0.35,
                             min_speech_duration=0.1, min_silence_duration=0.2,
                             speech_pad=0.0)
    # the brief dip should not split the segment
    assert len(segs) == 1


def test_min_speech_filter():
    fd = 0.032
    probs = np.zeros(100, dtype=np.float32)
    probs[30:32] = 0.9  # only ~64ms
    segs = probs_to_segments(probs, fd, min_speech_duration=0.3, speech_pad=0.0)
    assert segs == []


def test_padding_and_merge():
    fd = 0.032
    probs = np.zeros(200, dtype=np.float32)
    probs[30:40] = 0.9
    probs[44:54] = 0.9  # close gap; padding should merge
    segs = probs_to_segments(probs, fd, min_speech_duration=0.05,
                             min_silence_duration=0.05, speech_pad=0.1)
    assert len(segs) == 1


def test_max_speech_split():
    fd = 0.032
    probs = np.full(300, 0.9, dtype=np.float32)
    segs = probs_to_segments(probs, fd, min_speech_duration=0.1,
                             max_speech_duration=2.0, speech_pad=0.0)
    assert len(segs) >= 2
    assert all(s.duration <= 2.01 for s in segs)


def test_segment_dataclass():
    s = SpeechSegment(1.0, 2.5)
    assert s.duration == 1.5
    assert tuple(s) == (1.0, 2.5)
    assert s.as_tuple() == (1.0, 2.5)
