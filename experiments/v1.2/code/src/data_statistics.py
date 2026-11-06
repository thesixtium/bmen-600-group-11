import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure
from mne.io import BaseRaw
from numpy.typing import NDArray
from scipy.signal import welch

from download_data import RawData
from preprocess_data import SFREQ, ProcessedData

OUT_DIR: Path = Path(__file__).resolve().parent.parent / "results" / "data_statistics"

CLASSES: tuple[str, ...] = ("left hand", "right hand", "feet", "tongue")
COLORS: tuple[str, ...] = ("tab:blue", "tab:red", "tab:green", "tab:orange")
BANDS: dict[str, tuple[float, float]] = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
KEY_CHANNELS: tuple[str, ...] = ("C3", "Cz", "C4")
MAX_LAG_S: float = 0.5


@dataclass(frozen=True)
class Trials:
    """Every preprocessed trial of every subject and session, stacked."""

    X: NDArray[np.float32]  # (n_trials, 22, 256), in microvolts
    y: NDArray[np.int64]  # (n_trials,) 0-3
    subject: NDArray[np.int64]  # (n_trials,)
    session: NDArray[np.int64]  # (n_trials,) 0 = session 1 (train), 1 = session 2 (test)
    channels: list[str]


def _eeg_channels(raw: BaseRaw) -> list[str]:
    return [c for c in raw.ch_names if not c.startswith("EOG") and c != "STI"]


# file name -> (how to read it, what to take away from it)
NOTES: dict[str, tuple[str, str]] = {
    "psd_overview.png": (
        "x = frequency, y = power in dB (higher = more power). Thin lines are the 22 channels, black is the channel mean. "
        "Red dotted lines mark 4 and 40 Hz, the gray dashed line marks 50 Hz. Left is the raw recording, right is after "
        "preprocessing, with a shared y-axis so heights are comparable.",
        "After preprocessing the spectrum should fall steeply below 4 Hz and above 40 Hz (band-pass worked) and be "
        "unchanged in between. A bump near 10 Hz is the alpha/mu rhythm. In the raw plot, no spike at 50 Hz means the "
        "notch filter did its job; a surviving spike is residual line noise, which the 40 Hz low-pass then removes. "
        "The overall downward slope is normal for EEG.",
    ),
    "psd_notch_zoom.png": (
        "Zoom on 35-64 Hz of the channel-mean PSD. Blue is the raw continuous data, orange is the preprocessed epochs.",
        "Blue shows what is really at 50 Hz (a dip means the notch worked, a peak means residual line noise). Orange "
        "should drop tens of dB past 40 Hz, so nothing near 50 Hz reaches the network. Orange stops at 64 Hz because "
        "of the 128 Hz sampling. Small level differences between the lines partly come from the different segments used.",
    ),
    "psd_per_subject.png": (
        "Same as the PSD overview, but each line is one subject (channel mean). Left raw, right preprocessed.",
        "Similar shapes across subjects mean the preprocessing treats everyone consistently. Vertical offsets are real "
        "differences in subject amplitude (one reason band power is z-scored per subject elsewhere). A subject with a "
        "strong 50 Hz spike, a missing 10 Hz bump, or a flat spectrum has a noisier or atypical recording.",
    ),
    "class_distribution.png": (
        "Left: each row is one subject and session, each column a class, and the number is the trial count. Right: "
        "total trials per class.",
        "Every cell should be 72 and every total 1296. That means the classes are balanced, so chance accuracy is "
        "25% and no class weighting is needed. Any other number would change the chance level. The script asserts this.",
    ),
    "amplitude_hist_pooled.png": (
        "x = signal amplitude in microvolts after band-pass, y = density on a log scale so the tails are visible. One "
        "curve per class, all subjects and both sessions pooled.",
        "Curves centered on 0 with a bell shape are expected (the band-pass removed the DC offset). Tails heavier than a "
        "Gaussian point to artifacts and outliers. Near-complete overlap between classes means raw amplitude alone does "
        "not separate the classes: the information is in band power, spatial pattern and timing, which the network has "
        "to learn. The width also shows the input scale the network will see (microvolts, not volts).",
    ),
    "amplitude_hist_per_subject.png": (
        "Same as the pooled amplitude histogram, but one panel per subject at channel C3.",
        "Different widths across subjects mean different amplitude scales, which the pooled plot mixes together. This "
        "is part of why cross-subject models struggle. Very heavy tails in one subject flag an artifact-heavy "
        "recording. Class curves should still overlap within each subject.",
    ),
    "band_power_hist.png": (
        "x = log10 band power, z-scored within each subject and session (0 = that subject and session's average, left "
        "= weaker). Rows are mu (8-13 Hz) and beta (13-30 Hz), columns are C3 and C4. One curve per class.",
        "A class curve shifted sideways means that class changes power in that band at that channel. Motor imagery is "
        "expected to lower power over the opposite hemisphere: right-hand imagery shifts left at C3, left-hand imagery "
        "shifts left at C4. Shifts are usually small compared with the spread, so any single feature only weakly "
        "separates the classes, which is why combining channels and time points is needed.",
    ),
    "channel_correlation.png": (
        "Each cell is the Pearson correlation between two channels over time within a trial, averaged over all trials. "
        "Red = move together, blue = move oppositely, the diagonal is 1.",
        "Strong red between neighboring electrodes is volume conduction: each electrode picks up much of the same "
        "activity, so channels are highly redundant. That is the reason to use spatial filtering (CSP, or EEGNet's "
        "depthwise spatial convolution) instead of treating channels as independent.",
    ),
    "class_band_power_correlation.png": (
        "Left: rows are channels, columns are class x band. Each cell is the correlation between 'this trial belongs "
        "to this class (vs the rest)' and log band power at that channel. Red = this class has more power there, blue "
        "= less. Right: eta^2, the fraction of band-power variance explained by class across all four classes (0 to 1). "
        "Band power is z-scored within subject and session.",
        "Look for blue over sensorimotor channels (around C3/C4) in the hand columns, on the side opposite the "
        "imagined hand. Values are expected to be small. Larger eta^2 marks the channels and bands where class matters "
        "most. If everything is near zero, the class signal is not visible in single-channel band power and has to come "
        "from channel combinations or temporal structure.",
    ),
    "class_topomaps.png": (
        "Scalp seen from above, nose at the top. Color = mean log10 band power for that class minus the all-class mean, "
        "averaged over subjects. Blue = less power than average, red = more. The color scale is shared within each row "
        "only.",
        "Hand imagery should show blue (desynchronization) over the opposite sensorimotor cortex, feet around the "
        "midline, and tongue differing from the others. A clear pattern like this means the labels, the cue offset and "
        "the 0.5-2.5 s window are aligned correctly. A blurry or absent pattern suggests misaligned epochs.",
    ),
    "autocorrelation.png": (
        "x = lag in ms, y = how similar a trial's signal is to itself shifted by that lag (1 = identical, 0 = "
        "unrelated), averaged over trials. Thin lines are channels, black is the mean.",
        "Oscillation is the dominant rhythm (about 100 ms per cycle for 10 Hz). How slowly it dies out is how long "
        "temporal context persists in the signal. Slow decay over hundreds of ms means there is structure over time "
        "that a recurrent network could exploit. Fast decay means the signal behaves more like noise. The band-pass "
        "also forces some negative lobes.",
    ),
    "session_shift.png": (
        "Top: one vertical segment per subject, from session 1 (blue) to session 2 (orange), mean log10 band power "
        "averaged over channels (not z-scored). Bottom: subject x channel heatmap of session 2 minus session 1 "
        "(red = more power in session 2).",
        "Long segments or strong colors mean the two sessions differ, so a model trained on session 1 sees shifted "
        "data in session 2. That is the non-stationarity behind your cross-session protocol. A difference of 0.1 is "
        "about a 26% change in power. If the direction and size vary by subject and channel, no single global "
        "correction fixes it. Small differences mean the sessions are comparable.",
    ),
    "trial_rms.png": (
        "Each box is the spread of per-trial RMS amplitude (microvolts, log y-axis) for one subject and session: the "
        "box covers the middle 50% of trials, the line is the median, dots are outlier trials. Blue = session 1, "
        "orange = session 2.",
        "Boxes at different heights mean different amplitude scales across subjects. A gap between a subject's blue "
        "and orange boxes is a session shift. Isolated high dots are artifact trials, which are kept in your plan and "
        "could destabilize training, so check whether normalization or outlier handling is needed.",
    ),
}


def _save(fig: Figure, name: str) -> None:
    read, takeaway = NOTES[name]
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {read}", width) + "\n\n" + textwrap.fill(f"Takeaways: {takeaway}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _db(power: NDArray[np.float64]) -> NDArray[np.float64]:
    return 10 * np.log10(power)


def _stack(data: RawData, processed: ProcessedData) -> Trials:
    sessions = [s for subject in sorted(processed) for s in processed[subject].values()]
    return Trials(
        X=(np.concatenate([s.X for s in sessions]) * 1e6).astype(np.float32),
        y=np.concatenate([s.y for s in sessions]),
        subject=np.concatenate([np.full(len(s.y), s.subject) for s in sessions]),
        session=np.concatenate([np.full(len(s.y), int(s.session[0])) for s in sessions]),
        channels=_eeg_channels(data[min(data)]["0train"]["0"]),
    )


def _log_band_power(X: NDArray[np.float32]) -> NDArray[np.float64]:
    """log10 band power (uV^2) per trial: (n_trials, 22, n_bands)."""
    f, p = welch(X, fs=SFREQ, nperseg=X.shape[-1], axis=-1)
    df = f[1] - f[0]
    bands = [np.log10(p[..., (f >= lo) & (f <= hi)].sum(-1) * df) for lo, hi in BANDS.values()]
    return np.stack(bands, axis=-1).astype(np.float64)


def _zscore(bp: NDArray[np.float64], t: Trials) -> NDArray[np.float64]:
    """Z-score band power within each subject and session (removes amplitude scale and session shift)."""
    z = np.empty_like(bp)
    for s in np.unique(t.subject):
        for sess in (0, 1):
            m = (t.subject == s) & (t.session == sess)
            z[m] = (bp[m] - bp[m].mean(0)) / bp[m].std(0)
    return z


# ---------------------------------------------------------------- 1. PSD


def _psd_plots(data: RawData, t: Trials) -> None:
    # Raw: continuous recordings at 250 Hz, averaged over runs per subject
    raw_psd: dict[int, NDArray[np.float64]] = {}
    for subject, sessions in data.items():
        psds = []
        for runs in sessions.values():
            for raw in runs.values():
                x = raw.get_data(picks=t.channels) * 1e6
                f_raw, p = welch(x, fs=raw.info["sfreq"], nperseg=500, axis=-1)
                psds.append(p)
        raw_psd[subject] = np.mean(psds, axis=0)  # (22, freqs), uV^2/Hz

    # Preprocessed: 128 Hz epochs, averaged over trials per subject
    f_pro, p = welch(t.X, fs=SFREQ, nperseg=t.X.shape[-1], axis=-1)
    pro_psd = {s: p[t.subject == s].mean(0).astype(np.float64) for s in raw_psd}

    raw_grand = np.mean(list(raw_psd.values()), axis=0)
    pro_grand = np.mean(list(pro_psd.values()), axis=0)

    # Overview: every channel, raw vs preprocessed
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, f, grand, title, xmax in (
        (axes[0], f_raw, raw_grand, "Raw (250 Hz, continuous)", 125),
        (axes[1], f_pro, pro_grand, "Preprocessed (128 Hz, epochs)", 64),
    ):
        ax.plot(f, _db(grand).T, lw=0.6, alpha=0.6)
        ax.plot(f, _db(grand.mean(0)), "k", lw=2, label="channel mean")
        for edge in (4, 40):
            ax.axvline(edge, ls=":", c="r")
        ax.axvline(50, ls="--", c="gray")
        ax.set(xlim=(0, xmax), xlabel="Frequency (Hz)", title=title)
        ax.legend()
    axes[0].set_ylabel("PSD (dB re 1 uV$^2$/Hz)")
    fig.suptitle("Grand-average PSD, all subjects. Red: 4 and 40 Hz. Gray: 50 Hz.")
    _save(fig, "psd_overview.png")

    # Zoom on the line-noise / low-pass region
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_raw, _db(raw_grand.mean(0)), label="raw (continuous)")
    ax.plot(f_pro, _db(pro_grand.mean(0)), label="preprocessed (epochs)")
    ax.axvline(50, ls="--", c="gray", label="50 Hz")
    ax.axvline(40, ls=":", c="r", label="40 Hz low-pass")
    ax.set(xlim=(35, 64), xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 uV$^2$/Hz)")
    ax.set_title("Channel-mean PSD around 50 Hz")
    ax.legend()
    _save(fig, "psd_notch_zoom.png")

    # Per subject (channel mean)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, f, psds, title, xmax in (
        (axes[0], f_raw, raw_psd, "Raw", 125),
        (axes[1], f_pro, pro_psd, "Preprocessed", 64),
    ):
        for s, psd in psds.items():
            ax.plot(f, _db(psd.mean(0)), lw=1, label=f"S{s}")
        ax.axvline(50, ls="--", c="gray")
        ax.set(xlim=(0, xmax), xlabel="Frequency (Hz)", title=title)
    axes[0].set_ylabel("PSD (dB re 1 uV$^2$/Hz)")
    axes[1].legend(ncol=3, fontsize=8)
    _save(fig, "psd_per_subject.png")

    # Attenuation table. Note: raw is continuous data, preprocessed is the 0.5-2.5 s epochs.
    print("PSD change after preprocessing (channel mean, dB):")
    for hz in (2, 4, 10, 20, 40, 45, 50, 55, 60):
        i_raw, i_pro = np.argmin(np.abs(f_raw - hz)), np.argmin(np.abs(f_pro - hz))
        diff = _db(pro_grand[:, i_pro].mean()) - _db(raw_grand[:, i_raw].mean())
        print(f"  {hz:>3} Hz: {diff:+6.1f} dB")


# ---------------------------------------------------------------- 2. Classes


def _class_distribution(t: Trials) -> None:
    rows = [(s, sess) for s in np.unique(t.subject) for sess in (0, 1)]
    counts = np.array(
        [[np.sum((t.subject == s) & (t.session == sess) & (t.y == c)) for c in range(4)] for s, sess in rows]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [2, 1]})
    ax1.imshow(counts, cmap="Blues", vmin=0, vmax=counts.max() * 1.3, aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(4):
            ax1.text(j, i, str(counts[i, j]), ha="center", va="center")
    ax1.set_xticks(range(4), CLASSES)
    ax1.set_yticks(range(len(rows)), [f"S{s} {'train' if sess == 0 else 'test'}" for s, sess in rows])
    ax1.set_title("Trials per class, per subject and session")

    totals = counts.sum(0)
    ax2.bar(CLASSES, totals, color=COLORS)
    for j, v in enumerate(totals):
        ax2.text(j, v, str(v), ha="center", va="bottom")
    ax2.set_title("Total trials per class")
    ax2.tick_params(axis="x", rotation=30)
    _save(fig, "class_distribution.png")

    assert (counts == 72).all(), f"Classes are not balanced:\n{counts}"


# ---------------------------------------------------------------- 3. Histograms


def _amplitude_hist(t: Trials) -> None:
    idx = [t.channels.index(c) for c in KEY_CHANNELS]

    def draw(ax: plt.Axes, x: NDArray[np.float32], y: NDArray[np.int64], title: str) -> None:
        lim = np.percentile(np.abs(x), 99.9)
        bins = np.linspace(-lim, lim, 201)
        for c in range(4):
            ax.hist(x[y == c].ravel(), bins=bins, density=True, histtype="step", color=COLORS[c], label=CLASSES[c])
        ax.set(yscale="log", xlabel="Amplitude (uV)", title=title)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ch, name in zip(axes, idx, KEY_CHANNELS):
        draw(ax, t.X[:, ch], t.y, f"{name}, all subjects pooled")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    _save(fig, "amplitude_hist_pooled.png")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, s in zip(axes.ravel(), np.unique(t.subject)):
        m = t.subject == s
        draw(ax, t.X[m][:, idx[0]], t.y[m], f"Subject {s}, {KEY_CHANNELS[0]}")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "amplitude_hist_per_subject.png")


def _band_power_hist(z: NDArray[np.float64], t: Trials) -> None:
    bins = np.linspace(-4, 4, 81)
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(11, 7), sharex=True)
    for b, band in enumerate(BANDS):
        for ax, name in zip(axes[b], ("C3", "C4")):
            ch = t.channels.index(name)
            for c in range(4):
                ax.hist(z[t.y == c, ch, b], bins=bins, density=True, histtype="step", color=COLORS[c], label=CLASSES[c])
            ax.set_title(f"{band} band, {name}")
    axes[-1, 0].set_xlabel("log band power (z-scored within subject and session)")
    axes[-1, 1].set_xlabel("log band power (z-scored within subject and session)")
    axes[0, 0].legend()
    _save(fig, "band_power_hist.png")


# ---------------------------------------------------------------- 4. Correlation


def _channel_correlation(t: Trials) -> None:
    Xc = t.X - t.X.mean(-1, keepdims=True)
    Xn = Xc / np.linalg.norm(Xc, axis=-1, keepdims=True)
    corr = np.mean(Xn @ Xn.transpose(0, 2, 1), axis=0)  # mean of per-trial Pearson matrices

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(t.channels)), t.channels, rotation=90)
    ax.set_yticks(range(len(t.channels)), t.channels)
    ax.set_title("Channel x channel Pearson correlation\n(per trial, averaged over all trials)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "channel_correlation.png")


def _class_band_correlation(z: NDArray[np.float64], t: Trials) -> None:
    n_ch = len(t.channels)
    r = np.empty((n_ch, 4 * len(BANDS)))
    labels: list[str] = []
    for c in range(4):
        ind = (t.y == c).astype(np.float64)
        # one-vs-rest Pearson r (z has mean 0 and std 1)
        r[:, c * len(BANDS) : (c + 1) * len(BANDS)] = np.einsum("ncb,n->cb", z, ind - ind.mean()) / (len(ind) * ind.std())
        labels += [f"{CLASSES[c]}\n{band}" for band in BANDS]

    # eta^2: fraction of band-power variance explained by class (one-way ANOVA)
    eta = sum((t.y == c).sum() * z[t.y == c].mean(0) ** 2 for c in range(4)) / len(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [4, 1]})
    v = np.abs(r).max()
    im1 = ax1.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax1.set_xticks(range(r.shape[1]), labels, fontsize=8)
    ax1.set_yticks(range(n_ch), t.channels)
    ax1.set_title("One-vs-rest correlation: class indicator vs log band power")
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Pearson r")

    im2 = ax2.imshow(eta, cmap="viridis", vmin=0, aspect="auto")
    ax2.set_xticks(range(len(BANDS)), list(BANDS))
    ax2.set_yticks(range(n_ch), t.channels)
    ax2.set_title("Variance explained\nby class (eta$^2$)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    fig.suptitle("Band power z-scored within subject and session")
    _save(fig, "class_band_power_correlation.png")


# ---------------------------------------------------------------- 5. Additions


def _topomaps(data: RawData, t: Trials, bp: NDArray[np.float64]) -> None:
    info = data[min(data)]["0train"]["0"].copy().pick(t.channels).info
    subjects = np.unique(t.subject)

    # (class mean - all-class mean) of log band power, averaged over subjects
    diff = np.zeros((4, len(t.channels), len(BANDS)))
    for s in subjects:
        m = t.subject == s
        overall = bp[m].mean(0)
        for c in range(4):
            diff[c] += bp[m & (t.y == c)].mean(0) - overall
    diff /= len(subjects)

    fig, axes = plt.subplots(len(BANDS), 4, figsize=(14, 7))
    for b, band in enumerate(BANDS):
        vmax = np.abs(diff[..., b]).max()
        for c in range(4):
            im, _ = mne.viz.plot_topomap(
                diff[c, :, b], info, axes=axes[b, c], show=False, vlim=(-vmax, vmax), cmap="RdBu_r"
            )
            axes[b, c].set_title(f"{CLASSES[c]}, {band}")
        fig.colorbar(im, ax=axes[b].tolist(), shrink=0.8, label="log10 power vs all-class mean")
    _save(fig, "class_topomaps.png")


def _autocorrelation(t: Trials) -> None:
    max_lag = int(MAX_LAG_S * SFREQ)
    n = t.X.shape[-1]
    total = np.zeros((len(t.channels), max_lag + 1))
    for i in range(0, len(t.X), 500):
        x = t.X[i : i + 500].astype(np.float64)
        x -= x.mean(-1, keepdims=True)
        spec = np.fft.rfft(x, n=2 * n, axis=-1)
        ac = np.fft.irfft(spec * np.conj(spec), n=2 * n, axis=-1)[..., : max_lag + 1]
        total += (ac / ac[..., :1]).sum(0)
    mean_ac = total / len(t.X)

    lags_ms = np.arange(max_lag + 1) / SFREQ * 1000
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lags_ms, mean_ac.T, lw=0.6, alpha=0.6)
    ax.plot(lags_ms, mean_ac.mean(0), "k", lw=2, label="channel mean")
    ax.axhline(0, c="gray", lw=0.5)
    ax.set(xlabel="Lag (ms)", ylabel="Autocorrelation", title="Temporal autocorrelation (per trial, averaged)")
    ax.legend()
    _save(fig, "autocorrelation.png")


def _session_shift(bp: NDArray[np.float64], t: Trials) -> None:
    subjects = np.unique(t.subject)
    # mean[subject, session, channel, band]
    mean = np.array(
        [[bp[(t.subject == s) & (t.session == sess)].mean(0) for sess in (0, 1)] for s in subjects]
    )

    fig, axes = plt.subplots(2, len(BANDS), figsize=(13, 9))
    for b, band in enumerate(BANDS):
        s1, s2 = mean[:, 0, :, b].mean(-1), mean[:, 1, :, b].mean(-1)
        ax = axes[0, b]
        ax.vlines(subjects, s1, s2, color="gray")
        ax.scatter(subjects, s1, c="tab:blue", label="session 1 (train)")
        ax.scatter(subjects, s2, c="tab:orange", label="session 2 (test)")
        ax.set(xlabel="Subject", ylabel="Mean log10 power (uV$^2$)", title=f"{band} band, channel mean")
        ax.legend()

        diff = mean[:, 1, :, b] - mean[:, 0, :, b]
        v = np.abs(diff).max()
        ax = axes[1, b]
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(t.channels)), t.channels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(subjects)), [f"S{s}" for s in subjects])
        ax.set_title(f"{band}: session 2 minus session 1 (log10)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, "session_shift.png")


def _trial_rms(t: Trials) -> None:
    rms = np.sqrt(np.mean(t.X**2, axis=(1, 2)))
    subjects = np.unique(t.subject)

    fig, ax = plt.subplots(figsize=(12, 5))
    handles = []
    for sess, color, offset, name in ((0, "tab:blue", -0.2, "session 1 (train)"), (1, "tab:orange", 0.2, "session 2 (test)")):
        groups = [rms[(t.subject == s) & (t.session == sess)] for s in subjects]
        boxes = ax.boxplot(groups, positions=subjects + offset, widths=0.35, patch_artist=True, flierprops={"markersize": 2})
        for box in boxes["boxes"]:
            box.set_facecolor(color)
        handles.append(boxes["boxes"][0])
    ax.legend(handles, ["session 1 (train)", "session 2 (test)"])
    ax.set(yscale="log", xlabel="Subject", ylabel="Trial RMS (uV)", title="Per-trial RMS (artifact trials kept)")
    ax.set_xticks(subjects)
    _save(fig, "trial_rms.png")


# ---------------------------------------------------------------- entry point


def data_statistics(data: RawData, processed: ProcessedData) -> None:
    """Save descriptive graphs of the raw and preprocessed data to results/data_statistics/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t = _stack(data, processed)
    bp = _log_band_power(t.X)
    z = _zscore(bp, t)

    _psd_plots(data, t)
    _class_distribution(t)
    _amplitude_hist(t)
    _band_power_hist(z, t)
    _channel_correlation(t)
    _class_band_correlation(z, t)
    _topomaps(data, t, bp)
    _autocorrelation(t)
    _session_shift(bp, t)
    _trial_rms(t)
    print(f"Saved graphs to {OUT_DIR}")