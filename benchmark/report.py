#!/usr/bin/env python3
"""Generate REPORT.md and plots from results.json."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(RES, "plots")
COND_ORDER = ["clean", "20dB", "15dB", "10dB", "5dB", "0dB"]
SNR_X = {"clean": 30, "20dB": 20, "15dB": 15, "10dB": 10, "5dB": 5, "0dB": 0}


def _load():
    with open(os.path.join(RES, "results.json")) as f:
        return json.load(f)


def _conds(model):
    return [c for c in COND_ORDER if c in model]


def plot_vs_snr(data, metric, ylabel, fname, lower_better=False):
    plt.figure(figsize=(7, 4.5))
    for name, model in data["models"].items():
        conds = _conds(model)
        xs = [SNR_X[c] for c in conds]
        ys = [model[c][metric] for c in conds]
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("SNR (dB)   —  'clean' plotted at 30")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs noise level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, fname), dpi=130)
    plt.close()


def plot_roc(data, cond, fname):
    plt.figure(figsize=(6, 6))
    for name, model in data["models"].items():
        if cond not in model:
            continue
        roc = model[cond]["roc"]
        plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC {model[cond]['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {cond}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, fname), dpi=130)
    plt.close()


def _table(data, metric, fmt="{:.3f}"):
    conds = COND_ORDER
    head = "| model | " + " | ".join(conds) + " | mean |"
    sep = "|" + "---|" * (len(conds) + 2)
    lines = [head, sep]
    for name, model in data["models"].items():
        vals = [model[c][metric] for c in conds if c in model]
        cells = [fmt.format(model[c][metric]) if c in model else "—" for c in conds]
        mean = sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | **{fmt.format(mean)}** |")
    return "\n".join(lines)


def plot_noise_categories(noise, fname):
    cats = noise["meta"]["categories"]
    models = list(noise["models"])
    x = range(len(cats))
    w = 0.8 / len(models)
    plt.figure(figsize=(9, 4.8))
    for i, name in enumerate(models):
        ys = [noise["models"][name][c] for c in cats]
        plt.bar([xi + i * w for xi in x], ys, width=w, label=name)
    plt.xticks([xi + w * (len(models) - 1) / 2 for xi in x], cats, rotation=15)
    plt.ylim(0.85, 1.0)
    plt.ylabel("ROC-AUC")
    plt.title(f"Robustness by noise category @ {noise['meta']['snr_db']:.0f} dB SNR")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, fname), dpi=130)
    plt.close()


def _optional(path):
    p = os.path.join(RES, path)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = _load()
    meta = data["meta"]
    noise = _optional("results_noise.json")
    vox = _optional("results_vox.json")

    plot_vs_snr(data, "auc", "ROC-AUC", "auc_vs_snr.png")
    plot_vs_snr(data, "f1", "F1 (segmented)", "f1_vs_snr.png")
    plot_vs_snr(data, "der", "Detection Error Rate", "der_vs_snr.png", lower_better=True)
    plot_roc(data, "clean", "roc_clean.png")
    plot_roc(data, "5dB", "roc_5db.png")

    rank = sorted(data["models"].items(),
                  key=lambda kv: -sum(v["auc"] for v in kv[1].values()) / len(kv[1]))
    best = rank[0][0]

    md = f"""# vadonnx VAD benchmark

**Dataset:** {meta['dataset']}
**Setup:** {meta['n_utts']} LibriSpeech dev-clean utterances concatenated with silence
gaps and mixed with real ESC-50 background noise at controlled SNRs. Each condition is
~{meta['duration_s']:.0f}s ({meta['speech_frac']*100:.0f}% speech). All models are scored
on a common {meta['grid_ms']:.0f} ms frame grid against exact reference labels.

**Metrics**
- **ROC-AUC** — threshold-independent, from raw per-frame speech probabilities.
- **F1 / FA / Miss / DER** — at each model's deployed operating point
  (`get_speech_segments` with default smoothing). DER = (false-alarm + miss) / speech.

## ROC-AUC by condition (higher is better)

{_table(data, "auc", "{:.4f}")}

![AUC vs SNR](plots/auc_vs_snr.png)

## F1 (segmented) by condition

{_table(data, "f1")}

![F1 vs SNR](plots/f1_vs_snr.png)

## Detection Error Rate by condition (lower is better)

{_table(data, "der")}

![DER vs SNR](plots/der_vs_snr.png)

## ROC curves

| clean | 5 dB |
|---|---|
| ![ROC clean](plots/roc_clean.png) | ![ROC 5dB](plots/roc_5db.png) |

## Real-time factor (CPU, 1 thread)

{_table(data, "rtf", "{:.4f}")}

## Analysis

Averaged over all conditions, **{best}** has the highest ROC-AUC.

"""
    # automated observations
    obs = []
    for name, model in data["models"].items():
        if "clean" in model and "0dB" in model:
            drop = model["clean"]["auc"] - model["0dB"]["auc"]
            obs.append(f"- **{name}**: clean AUC {model['clean']['auc']:.3f} → 0 dB "
                       f"{model['0dB']['auc']:.3f} (Δ{drop:.3f}); mean RTF "
                       f"{sum(v['rtf'] for v in model.values())/len(model):.4f}.")
    md += "\n".join(obs) + "\n"

    # ---- real conversational dataset (VoxConverse) ----
    if vox:
        vm = vox["meta"]
        rows = ["| model | ROC-AUC | F1 | DER | FA | Miss |", "|---|---|---|---|---|---|"]
        for name, m in vox["models"].items():
            rows.append(f"| {name} | {m['auc']:.4f} | {m['f1']:.3f} | {m['der']:.3f} "
                        f"| {m['false_alarm']:.3f} | {m['miss']:.3f} |")
        md += f"""
## Real-world cross-check — VoxConverse dev

{vm['dataset']} — {vm['n_files']} clips, {vm['minutes']:.0f} min, {vm['speech_frac']*100:.0f}% speech.
This is genuine in-the-wild multi-speaker conversational audio (reference = union of RTTM
speaker turns), so numbers are lower and more realistic than the synthetic sweep.

{chr(10).join(rows)}
"""

    # ---- noise category breakdown ----
    if noise:
        cats = noise["meta"]["categories"]
        plot_noise_categories(noise, "noise_categories.png")
        head = "| model | " + " | ".join(cats) + " |"
        sep = "|" + "---|" * (len(cats) + 1)
        rows = [head, sep]
        for name, m in noise["models"].items():
            rows.append(f"| {name} | " + " | ".join(f"{m[c]:.4f}" for c in cats) + " |")
        md += f"""
## Robustness by noise category (ROC-AUC @ {noise['meta']['snr_db']:.0f} dB)

{chr(10).join(rows)}

![Noise categories](plots/noise_categories.png)

All backends are weakest on **human non-speech** noise (babble-like: laughter, coughing,
crying) — the hardest case for VAD since it is spectrally speech-like — and strongest on
**interior/domestic** sounds.
"""

    md += """
### Notes
- The synthetic sweep uses exact labels; utterance-internal pauses count as speech, so it
  measures speech/silence discrimination and noise robustness. **VoxConverse** provides the
  in-the-wild reference.
- **AVA-Speech** scoring is supported (`datasets.load_ava_speech` + label CSV); its audio
  is fetched separately from YouTube via `yt-dlp`.
- `int8` variants show the accuracy difference from quantization.
- `ten` requires TEN's native feature extractor and is not scored here.
"""
    with open(os.path.join(RES, "REPORT.md"), "w") as f:
        f.write(md)
    print("wrote", os.path.join(RES, "REPORT.md"))


if __name__ == "__main__":
    main()
