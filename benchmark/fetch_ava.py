#!/usr/bin/env python3
"""Fetch AVA-Speech clip audio (the labeled 900-1800s window) for clips still on YouTube.

AVA video sources may be unavailable; the script iterates video IDs and keeps the
first `--n` that download successfully.
Writes 16 kHz mono wav to data/ava_audio/<video_id>.wav.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "ava_audio")


def video_ids():
    seen = []
    with open(os.path.join(DATA, "ava_speech_labels_v1.csv")) as f:
        for line in f:
            vid = line.split(",", 1)[0]
            if vid not in seen:
                seen.append(vid)
    return seen


def fetch(vid: str) -> bool:
    dest = os.path.join(OUT, f"{vid}.wav")
    if os.path.isfile(dest):
        return True
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "-x", "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--download-sections", "*900-1800",
        "-o", os.path.join(OUT, f"{vid}.%(ext)s"),
        f"https://www.youtube.com/watch?v={vid}",
    ]
    try:
        subprocess.run(cmd, timeout=300, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.isfile(dest)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    got = 0
    for i, vid in enumerate(video_ids()[: args.max_attempts]):
        ok = fetch(vid)
        print(f"[{i}] {vid}: {'OK' if ok else 'skip'}", flush=True)
        if ok:
            got += 1
        if got >= args.n:
            break
    print(f"fetched {got} AVA clips into {OUT}")


if __name__ == "__main__":
    sys.exit(main())
