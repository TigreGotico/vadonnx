# vadonnx benchmark

Benchmarks the published ONNX backends against standard VAD data, on a common 10 ms
frame grid, reporting threshold-independent **ROC-AUC** and deployed-operating-point
**F1 / false-alarm / miss / detection-error-rate**.

The headline benchmark is a **real-speech + real-noise SNR sweep**: LibriSpeech
dev-clean utterances concatenated with silence gaps and mixed with ESC-50 background
noise at `clean / 20 / 15 / 10 / 5 / 0` dB. This gives exact reference labels and a
controlled noise-robustness curve, fully reproducibly and without scraping.

Scored backends: `silero`, `marblenet` (+int8), `fsmn` (+quant), `speechbrain`, plus two
classical **baselines** — WebRTC VAD (`webrtc`) and an energy gate (`energy`) — as a
reference floor. Three views: the synthetic SNR sweep, a real **VoxConverse** cross-check,
and a per-noise-category breakdown.

See [`results/REPORT.md`](results/REPORT.md) for the generated report and plots.

## Run it

```bash
uv pip install "vadonnx[fsmn]"
uv pip install -r benchmark/requirements.txt

# fetch + extract data (~1 GB)
cd benchmark/data
curl -L -o dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz && tar xzf dev-clean.tar.gz
curl -L -o esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip && unzip -q esc50.zip
cd ../..

python benchmark/run_benchmark.py --n-utts 80      # -> results/results.json
python benchmark/report.py                          # -> results/REPORT.md + plots/
```

## Layout

| file | role |
|------|------|
| `datasets.py` | LibriSpeech+ESC-50 synthetic SNR builder; AVA-Speech loader |
| `metrics.py` | frame-grid resampling, ROC-AUC, operating-point F1/FA/Miss/DER |
| `run_benchmark.py` | scores each backend on every condition → `results.json` |
| `report.py` | renders `REPORT.md` + plots |

## Other / real-world datasets

The standard fully in-the-wild VAD benchmark is **AVA-Speech** (frame-labeled movie
audio). Its labels are public (`datasets.load_ava_speech`), but the audio must be
fetched from YouTube (`yt-dlp` + `ffmpeg`) into `data/ava_audio/<video_id>.wav`; it is
not bundled here. Diarization corpora with VAD references (AMI, VoxConverse, DIHARD) can
be scored the same way by writing a loader that returns `(audio, speech_intervals)`.
