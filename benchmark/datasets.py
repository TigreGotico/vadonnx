"""Dataset builders for the VAD benchmark.

Primary benchmark: real LibriSpeech speech concatenated with silence gaps and mixed with
real ESC-50 background noise at controlled SNRs — exact reference labels, reproducible,
and a clean noise-robustness sweep. Optional AVA-Speech loader for the paper-comparable
real-world number (requires audio fetched separately).
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

SR = 16000


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _read_mono(path: str, target_sr: int = SR) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return _resample(x, sr, target_sr)


def _energy_trim(x: np.ndarray, sr: int = SR, win: float = 0.02, rel: float = 0.06) -> np.ndarray:
    """Trim leading/trailing low-energy samples (simple energy gate, not a VAD model)."""
    w = int(win * sr)
    if len(x) < 2 * w:
        return x
    n = len(x) // w
    rms = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-12)
    thr = max(rms.max() * rel, 1e-4)
    voiced = np.where(rms > thr)[0]
    if voiced.size == 0:
        return x
    return x[voiced[0] * w : (voiced[-1] + 1) * w]


def load_librispeech_utts(root: str, n: int, seed: int = 0) -> List[np.ndarray]:
    files = sorted(glob.glob(os.path.join(root, "**", "*.flac"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .flac under {root}")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False)
    utts = []
    for i in picks:
        u = _energy_trim(_read_mono(files[i]))
        if len(u) > SR * 0.5:  # keep utterances >0.5s
            utts.append(u)
    return utts


def load_esc50_noises(root: str, n: int = 60, seed: int = 1) -> List[np.ndarray]:
    files = sorted(glob.glob(os.path.join(root, "**", "audio", "*.wav"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [_read_mono(files[i]) for i in picks]


ESC50_MACRO = {
    0: "animals", 1: "nature/water", 2: "human non-speech",
    3: "interior/domestic", 4: "exterior/urban",
}


def load_esc50_by_macro(root: str) -> dict:
    """Group ESC-50 clips into the 5 macro categories (target // 10). Returns
    ``{macro_name: [audio, ...]}`` resampled to 16 kHz mono."""
    import csv

    meta = glob.glob(os.path.join(root, "**", "meta", "esc50.csv"), recursive=True)
    audio_dir = glob.glob(os.path.join(root, "**", "audio"), recursive=True)
    if not meta or not audio_dir:
        raise FileNotFoundError("ESC-50 meta/audio not found")
    out: dict = {v: [] for v in ESC50_MACRO.values()}
    with open(meta[0]) as f:
        for row in csv.DictReader(f):
            macro = ESC50_MACRO[int(row["target"]) // 10]
            path = os.path.join(audio_dir[0], row["filename"])
            if os.path.isfile(path):
                out[macro].append(_read_mono(path))
    return out


def _tile_noise(noises: List[np.ndarray], length: int, rng) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    pos = 0
    while pos < length:
        nz = noises[rng.integers(len(noises))]
        take = min(len(nz), length - pos)
        out[pos : pos + take] = nz[:take]
        pos += take
    return out


def build_synthetic(
    utts: List[np.ndarray],
    noises: List[np.ndarray],
    snr_db: Optional[float],
    *,
    seed: int = 0,
    min_gap: float = 0.3,
    max_gap: float = 1.2,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Concatenate utterances with silence gaps; mix noise at ``snr_db`` (None = clean).

    Returns ``(audio_float32_16k, speech_intervals_seconds)``.
    """
    rng = np.random.default_rng(seed)
    parts: List[np.ndarray] = []
    intervals: List[Tuple[float, float]] = []
    t = 0.0
    # leading silence
    lead = rng.uniform(min_gap, max_gap)
    parts.append(np.zeros(int(lead * SR), dtype=np.float32))
    t += lead
    for u in utts:
        intervals.append((t, t + len(u) / SR))
        parts.append(u)
        t += len(u) / SR
        gap = rng.uniform(min_gap, max_gap)
        parts.append(np.zeros(int(gap * SR), dtype=np.float32))
        t += gap
    speech = np.concatenate(parts)

    if snr_db is None:
        return speech, intervals

    # speech RMS over speech regions only
    mask = np.zeros(len(speech), dtype=bool)
    for a, b in intervals:
        mask[int(a * SR) : int(b * SR)] = True
    speech_rms = np.sqrt((speech[mask] ** 2).mean() + 1e-12)
    noise = _tile_noise(noises, len(speech), rng)
    noise_rms = np.sqrt((noise ** 2).mean() + 1e-12)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    noisy = speech + noise * (target_noise_rms / noise_rms)
    peak = np.abs(noisy).max()
    if peak > 1.0:
        noisy = noisy / peak
    return noisy.astype(np.float32), intervals


def _merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def load_voxconverse(rttm_dir: str, audio_dir: str, max_files: int = 40, seed: int = 0):
    """Yield (file_id, audio, speech_intervals) for VoxConverse dev clips.

    The VAD reference is the union of all speaker turns in the RTTM (speech = any
    speaker active). Real multi-speaker conversational YouTube audio.
    """
    rttms = sorted(glob.glob(os.path.join(rttm_dir, "**", "*.rttm"), recursive=True))
    rng = np.random.default_rng(seed)
    if len(rttms) > max_files:
        rttms = [rttms[i] for i in sorted(rng.choice(len(rttms), max_files, replace=False))]
    out = []
    for rt in rttms:
        fid = os.path.splitext(os.path.basename(rt))[0]
        wav = None
        for cand in glob.glob(os.path.join(audio_dir, "**", f"{fid}.wav"), recursive=True):
            wav = cand
            break
        if not wav:
            continue
        ivs = []
        for line in open(rt):
            p = line.split()
            if len(p) >= 5 and p[0] == "SPEAKER":
                start, dur = float(p[3]), float(p[4])
                ivs.append((start, start + dur))
        out.append((fid, _read_mono(wav), _merge_intervals(ivs)))
    return out


def load_ava_speech(labels_csv: str, audio_dir: str, max_clips: int = 20):
    """Yield (clip_id, audio, intervals) for AVA-Speech clips whose audio is present.

    AVA labels mark 15-min movie segments (900–1800s) frame-by-frame as NO_SPEECH /
    CLEAN_SPEECH / SPEECH_WITH_MUSIC / SPEECH_WITH_NOISE. Audio must be fetched
    separately (YouTube) into ``audio_dir`` as ``<video_id>.wav`` (16k mono).
    """
    import csv
    from collections import defaultdict

    rows = defaultdict(list)
    with open(labels_csv) as f:
        for video_id, start, end, label in csv.reader(f):
            rows[video_id].append((float(start), float(end), label))
    out = []
    for vid, segs in rows.items():
        wav = os.path.join(audio_dir, f"{vid}.wav")
        if not os.path.isfile(wav):
            continue
        audio = _read_mono(wav)
        t0 = min(s for s, _, _ in segs)
        intervals = [(s - t0, e - t0) for s, e, lab in segs if lab != "NO_SPEECH"]
        out.append((vid, audio, intervals))
        if len(out) >= max_clips:
            break
    return out
