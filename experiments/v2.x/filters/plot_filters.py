"""What frequencies do EEGNet's temporal filters extract, and which channels do its spatial filters weight?

Reads the FINE-TUNED EEGNet-4,2 weights of every fold (9 targets x 10 repeats) of the v1 run and writes, next to this
file:
  filters_example.png    one fold (s3_r0): 4 temporal kernels, their frequency responses, their spatial weights
  filters_all_folds.png  all 4 x 90 frequency responses, histogram of peak frequencies, mean |spatial weight| per channel
  filters_summary.csv    per fold and filter: peak frequency, -3 dB edges, band energy shares, top-3 channels

Only weights are read: no data, no session-2 trials.

Layers used (models.EEGNet):
  conv1.weight      (4, 1, 1, 32)  temporal conv, 4 filters x 32 taps at 128 Hz (250 ms), 'same' padding, no bias
  bn1               BatchNorm per temporal filter: y = a * x + b with a = gamma / sqrt(var + eps). A constant gain plus
                    an offset, so it changes only the overall level of |H(f)|, not its shape. Responses below are shown
                    relative to each filter's own peak (dB re peak), so bn1 is left out.
  depthwise.weight  (8, 1, 22, 1)  depthwise spatial conv, groups = 4: output maps 2k and 2k+1 read temporal filter k
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import freqz  # noqa: E402

HERE = Path(__file__).resolve().parent
# The checkpoints are not in the repository (size); the v1 run writes them to this folder.
CKPT = HERE.parents[1] / "v1" / "results" / "checkpoints" / "ann"  # EEGNet checkpoints of the v1 run (not in the repository)
FS = 128.0          # sampling rate after preprocessing (Hz)
N_FREQ = 1024       # frequency grid points on 0..64 Hz
EXAMPLE = (3, 0)    # representative fold shown in filters_example.png
# BNCI2014_001 EEG channels in dataset order. The cached data stores no names; this order is the MOABB loader's
# ch_names list (moabb/datasets/bnci/bnci_2014.py), and preprocess_data keeps it when it drops the 3 EOG channels.
CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
            "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
BANDS = {"theta_4_8": (4, 8), "mu_8_13": (8, 13), "beta_13_30": (13, 30), "gamma_30_40": (30, 40)}
INPUT_BAND = (4.0, 40.0)  # band-pass of the preprocessed input: outside it the filters see (almost) no signal
# Okabe-Ito colours (colour-blind safe), one per temporal filter; band shading in pale tones
FILTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MU_SHADE, BETA_SHADE, OUT_SHADE = "#56B4E9", "#F0E442", "#BBBBBB"


def load_fold(target: int, repeat: int):
    """Fine-tuned and pretrained temporal kernels (4, 32) and |depthwise| weights (4 filters, 2 spatial filters, 22)."""
    sd = torch.load(CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True)
    ft, pre = sd["finetuned"], sd["pretrained"]
    kernels = ft["conv1.weight"][:, 0, 0, :].double().numpy()
    kernels_pre = pre["conv1.weight"][:, 0, 0, :].double().numpy()
    spatial = ft["depthwise.weight"][:, 0, :, 0].abs().double().numpy().reshape(4, 2, 22)  # maps 2k, 2k+1 -> filter k
    return kernels, kernels_pre, spatial


def response(kernel):
    """Frequency grid (Hz) and |H(f)| of one FIR kernel. The conv is a cross-correlation (kernel not flipped); flipping
    a real kernel changes only the phase, so the magnitude is the same."""
    f, h = freqz(kernel, worN=N_FREQ, fs=FS)
    return f, np.abs(h)


def to_db(mag):
    return 20 * np.log10(np.maximum(mag / mag.max(), 1e-6))


def describe(f, mag):
    """Peak frequency over 0-64 Hz and within the 4-40 Hz input band, -3 dB edges around the peak (contiguous region
    with gain >= peak / sqrt(2)), and the share of |H|^2 in each band relative to |H|^2 over 4-40 Hz (shares sum to 1)."""
    k = int(np.argmax(mag))
    above = mag >= mag[k] / np.sqrt(2)
    lo = k
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = k
    while hi < len(mag) - 1 and above[hi + 1]:
        hi += 1
    inb = (f >= INPUT_BAND[0]) & (f < INPUT_BAND[1])
    p = mag ** 2
    total = p[inb].sum()
    shares = {name: p[(f >= a) & (f < b)].sum() / total for name, (a, b) in BANDS.items()}
    return {"peak_hz": f[k], "peak_in_4_40_hz": f[inb][np.argmax(mag[inb])],
            "edge_lo_3db_hz": f[lo], "edge_hi_3db_hz": f[hi], **shares}


def shade_bands(ax):
    ax.axvspan(8, 13, color=MU_SHADE, alpha=0.25, lw=0, label="mu 8-13 Hz")
    ax.axvspan(13, 30, color=BETA_SHADE, alpha=0.35, lw=0, label="beta 13-30 Hz")
    ax.axvspan(0, INPUT_BAND[0], color=OUT_SHADE, alpha=0.3, lw=0, label="outside 4-40 Hz input")
    ax.axvspan(INPUT_BAND[1], FS / 2, color=OUT_SHADE, alpha=0.3, lw=0)


def plot_example(kernels, spatial, path):
    t_ms = np.arange(kernels.shape[1]) / FS * 1000
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), gridspec_kw={"height_ratios": [1, 1, 1.25]})
    for k in range(4):
        c = FILTER_COLORS[k]
        ax = axes[0, k]  # row 1: kernel in time
        ax.axhline(0, color="#888888", lw=0.6)
        ax.plot(t_ms, kernels[k], "-o", color=c, ms=3)
        ax.set(title=f"Temporal filter {k + 1}", xlabel="Time (ms)", ylabel="Weight (a.u.)")

        ax = axes[1, k]  # row 2: magnitude response
        f, mag = response(kernels[k])
        shade_bands(ax)
        ax.plot(f, to_db(mag), color=c, lw=2)
        d = describe(f, mag)
        ax.axvline(d["peak_hz"], color=c, ls="--", lw=1)
        ax.set(xlim=(0, 64), ylim=(-40, 3), xlabel="Frequency (Hz)", ylabel="Gain (dB re peak)",
               title=f"Peak {d['peak_hz']:.1f} Hz, -3 dB {d['edge_lo_3db_hz']:.1f}-{d['edge_hi_3db_hz']:.1f} Hz")
        if k == 0:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

        ax = axes[2, k]  # row 3: |spatial weights| of the 2 depthwise filters fed by this temporal filter
        x = np.arange(22)
        ax.bar(x - 0.2, spatial[k, 0], 0.4, color=c, label="spatial filter a")
        ax.bar(x + 0.2, spatial[k, 1], 0.4, color=c, alpha=0.45, label="spatial filter b")
        for j in np.argsort(spatial[k].mean(0))[::-1][:3]:  # 3 largest channels (mean of the 2 spatial filters)
            ax.annotate(CHANNELS[j], (j, spatial[k, :, j].max()), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x, CHANNELS, rotation=90, fontsize=8)
        ax.set_ylim(0, spatial[k].max() * 1.4)  # headroom so the legend does not hide the top bars and labels
        ax.set(xlabel="Channel", ylabel="|Depthwise weight|", title=f"Spatial weights of filter {k + 1}")
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.3, lw=0.5)
    fig.suptitle(f"EEGNet fine-tuned filters, fold s{EXAMPLE[0]}_r{EXAMPLE[1]} (target subject {EXAMPLE[0]}, repeat "
                 f"{EXAMPLE[1]}); bn1 only rescales each filter, so responses are shown relative to their peak",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_all(f, mags_db, peaks, peaks_inband, chan_mean, path):
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, 0])
    shade_bands(ax)
    for m in mags_db:
        ax.plot(f, m, color="#0072B2", lw=0.4, alpha=0.15)
    ax.plot(f, np.median(mags_db, 0), color="#000000", lw=2.2, label=f"median of {len(mags_db)} filters")
    q25, q75 = np.percentile(mags_db, [25, 75], axis=0)
    ax.plot(f, q25, color="#D55E00", lw=1.2, ls="--", label="25th / 75th percentile")
    ax.plot(f, q75, color="#D55E00", lw=1.2, ls="--")
    ax.set(xlim=(0, 64), ylim=(-40, 3), xlabel="Frequency (Hz)", ylabel="Gain (dB re each filter's peak)",
           title="Frequency responses of all 4 x 90 fine-tuned temporal filters")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax = fig.add_subplot(gs[0, 1])
    bins = np.arange(0, 66, 2)
    shade_bands(ax)
    ax.hist(peaks, bins=bins, color="#0072B2", alpha=0.8, label="peak over 0-64 Hz")
    ax.hist(peaks_inband, bins=bins, histtype="step", color="#D55E00", lw=2, label="peak within 4-40 Hz")
    ax.set(xlim=(0, 64), xlabel="Peak frequency (Hz)", ylabel=f"Number of filters (of {len(peaks)})",
           title="Peak frequency of each temporal filter")
    ax.legend(fontsize=9, framealpha=0.9)

    ax = fig.add_subplot(gs[1, :])
    order = np.argsort(chan_mean)[::-1]
    ax.bar(np.arange(22), chan_mean[order], color="#009E73")
    ax.set_xticks(np.arange(22), [CHANNELS[i] for i in order])
    ax.set(xlabel="Channel (sorted)", ylabel="Mean |depthwise weight|",
           title="Mean absolute spatial weight per channel (8 spatial filters x 90 folds)")
    for a in fig.axes:
        a.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    rows, mags_db, chan_sum = [], [], np.zeros(22)
    files = sorted(CKPT.glob("s*_r*.pt"), key=lambda p: tuple(int(s[1:]) for s in p.stem.split("_")))
    for p in files:
        t, r = (int(s[1:]) for s in p.stem.split("_"))
        kernels, kernels_pre, spatial = load_fold(t, r)
        chan_sum += spatial.mean((0, 1))
        if (t, r) == EXAMPLE:
            plot_example(kernels, spatial, HERE / "filters_example.png")
        for k in range(4):
            f, mag = response(kernels[k])
            mags_db.append(to_db(mag))
            top = np.argsort(spatial[k].mean(0))[::-1][:3]
            # how much fine-tuning changed the kernel: cosine similarity to the pretrained kernel of the same filter
            cos = kernels[k] @ kernels_pre[k] / (np.linalg.norm(kernels[k]) * np.linalg.norm(kernels_pre[k]))
            rows.append({"target": t, "repeat": r, "filter": k + 1, **describe(f, mag),
                         "cos_to_pretrained": float(cos), "top3_channels": " ".join(CHANNELS[j] for j in top)})
    with open(HERE / "filters_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for d in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, (float, np.floating)) else v) for k, v in d.items()})
    peaks = np.array([d["peak_hz"] for d in rows])
    peaks_in = np.array([d["peak_in_4_40_hz"] for d in rows])
    plot_all(f, np.array(mags_db), peaks, peaks_in, chan_sum / len(files), HERE / "filters_all_folds.png")

    # console summary (the numbers quoted in the README)
    print(f"{len(files)} folds, {len(rows)} filters")
    print(f"peak 0-64 Hz: median {np.median(peaks):.1f} Hz, IQR {np.percentile(peaks, 25):.1f}-"
          f"{np.percentile(peaks, 75):.1f}; below 4 Hz: {np.mean(peaks < 4):.0%}; above 40 Hz: {np.mean(peaks > 40):.0%}")
    print(f"peak within 4-40 Hz: median {np.median(peaks_in):.1f} Hz; 4-8: {np.mean(peaks_in < 8):.0%}, mu: "
          f"{np.mean((peaks_in >= 8) & (peaks_in < 13)):.0%}, beta: {np.mean((peaks_in >= 13) & (peaks_in < 30)):.0%}"
          f", 30-40: {np.mean(peaks_in >= 30):.0%}")
    for b, (lo, hi) in BANDS.items():
        v = np.array([d[b] for d in rows])
        print(f"energy share {b}: median {np.median(v):.2f} (flat filter would give {(hi - lo) / 36:.2f})")
    cos = np.array([d["cos_to_pretrained"] for d in rows])
    print(f"cosine fine-tuned vs pretrained kernel: median {np.median(cos):.3f}, min {cos.min():.3f}")
    order = np.argsort(chan_sum)[::-1]
    print("channels by mean |w|:", ", ".join(f"{CHANNELS[i]} {chan_sum[i] / len(files):.3f}" for i in order))
    for d in (d for d in rows if (d["target"], d["repeat"]) == EXAMPLE):
        print(f"example filter {d['filter']}: peak {d['peak_hz']:.1f} Hz, top3 {d['top3_channels']}")


if __name__ == "__main__":
    main()
