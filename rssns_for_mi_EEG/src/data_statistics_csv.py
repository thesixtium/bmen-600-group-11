"""CSV files with the numbers behind every figure of data_statistics.py (same file stem, same folder as the PNG:
results/data_statistics/<stem>.csv). The computations repeat the ones of data_statistics.py exactly, using its own helper
functions, so the CSVs match the figures. data_statistics.py itself is untouched.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch

from csv_export import box_stats, write_csv
from data_statistics import (
    BANDS,
    CLASSES,
    KEY_CHANNELS,
    MAX_LAG_S,
    OUT_DIR,
    Trials,
    _db,
    _log_band_power,
    _stack,
    _zscore,
)
from download_data import RawData
from preprocess_data import SFREQ, ProcessedData


def _psd_long(f: np.ndarray, arr: np.ndarray, labels: list[str], **tag) -> pd.DataFrame:
    """arr: (n_labels, n_freq) in dB."""
    df = pd.DataFrame({"frequency_hz": np.tile(f, len(labels)), "channel": np.repeat(labels, len(f)), "psd_db": arr.ravel()})
    for k, v in reversed(list(tag.items())):
        df.insert(0, k, v)
    return df


def _psd(data: RawData, t: Trials) -> None:
    raw_psd: dict[int, np.ndarray] = {}
    for subject, sessions in data.items():
        psds = []
        for runs in sessions.values():
            for raw in runs.values():
                x = raw.get_data(picks=t.channels) * 1e6
                f_raw, p = welch(x, fs=raw.info["sfreq"], nperseg=500, axis=-1)
                psds.append(p)
        raw_psd[subject] = np.mean(psds, axis=0)
    f_pro, p = welch(t.X, fs=SFREQ, nperseg=t.X.shape[-1], axis=-1)
    pro_psd = {s: p[t.subject == s].mean(0).astype(np.float64) for s in raw_psd}
    raw_grand, pro_grand = np.mean(list(raw_psd.values()), axis=0), np.mean(list(pro_psd.values()), axis=0)
    labels = list(t.channels) + ["channel_mean"]

    def with_mean(g: np.ndarray) -> np.ndarray:
        return np.vstack([_db(g), _db(g.mean(0))[None]])

    # psd_overview.png
    write_csv(pd.concat([_psd_long(f_raw, with_mean(raw_grand), labels, source="raw"),
                         _psd_long(f_pro, with_mean(pro_grand), labels, source="preprocessed")]), OUT_DIR / "psd_overview.csv")

    # psd_notch_zoom.png (35-64 Hz, channel mean)
    zoom = []
    for src, f, g in (("raw", f_raw, raw_grand), ("preprocessed", f_pro, pro_grand)):
        m = (f >= 35) & (f <= 64)
        zoom.append(pd.DataFrame({"source": src, "frequency_hz": f[m], "psd_db_channel_mean": _db(g.mean(0))[m]}))
    write_csv(pd.concat(zoom), OUT_DIR / "psd_notch_zoom.csv")

    # psd_per_subject.png (channel mean)
    rows = []
    for src, f, psds in (("raw", f_raw, raw_psd), ("preprocessed", f_pro, pro_psd)):
        for s, psd in psds.items():
            rows.append(pd.DataFrame({"source": src, "subject": s, "frequency_hz": f, "psd_db_channel_mean": _db(psd.mean(0))}))
    write_csv(pd.concat(rows), OUT_DIR / "psd_per_subject.csv")

    # attenuation table printed by data_statistics()
    att = []
    for hz in (2, 4, 10, 20, 40, 45, 50, 55, 60):
        i_raw, i_pro = np.argmin(np.abs(f_raw - hz)), np.argmin(np.abs(f_pro - hz))
        raw_db, pro_db = _db(raw_grand[:, i_raw].mean()), _db(pro_grand[:, i_pro].mean())
        att.append({"frequency_hz": hz, "raw_db": raw_db, "preprocessed_db": pro_db, "change_db": pro_db - raw_db})
    write_csv(pd.DataFrame(att), OUT_DIR / "psd_attenuation.csv")


def _classes(t: Trials) -> None:
    rows = [{"subject": int(s), "session": "train" if sess == 0 else "test", "class": CLASSES[c],
             "n_trials": int(np.sum((t.subject == s) & (t.session == sess) & (t.y == c)))}
            for s in np.unique(t.subject) for sess in (0, 1) for c in range(4)]
    write_csv(pd.DataFrame(rows), OUT_DIR / "class_distribution.csv")


def _hist(x: np.ndarray, y: np.ndarray, bins: np.ndarray, **tag) -> pd.DataFrame:
    frames = []
    for c in range(4):
        h, _ = np.histogram(x[y == c].ravel(), bins=bins, density=True)
        frames.append(pd.DataFrame({**tag, "class": CLASSES[c], "bin_left": bins[:-1], "bin_right": bins[1:], "density": h}))
    return pd.concat(frames)


def _amplitude(t: Trials) -> None:
    idx = [t.channels.index(c) for c in KEY_CHANNELS]

    def bins_for(x: np.ndarray) -> np.ndarray:
        lim = np.percentile(np.abs(x), 99.9)
        return np.linspace(-lim, lim, 201)

    pooled = [_hist(t.X[:, ch], t.y, bins_for(t.X[:, ch]), channel=name) for ch, name in zip(idx, KEY_CHANNELS)]
    write_csv(pd.concat(pooled), OUT_DIR / "amplitude_hist_pooled.csv")
    per = []
    for s in np.unique(t.subject):
        m = t.subject == s
        x = t.X[m][:, idx[0]]
        per.append(_hist(x, t.y[m], bins_for(x), subject=int(s), channel=KEY_CHANNELS[0]))
    write_csv(pd.concat(per), OUT_DIR / "amplitude_hist_per_subject.csv")


def _band_power_hist(z: np.ndarray, t: Trials) -> None:
    bins = np.linspace(-4, 4, 81)
    frames = []
    for b, band in enumerate(BANDS):
        for name in ("C3", "C4"):
            ch = t.channels.index(name)
            frames.append(_hist(z[:, ch, b], t.y, bins, band=band, channel=name))
    write_csv(pd.concat(frames), OUT_DIR / "band_power_hist.csv")


def _channel_correlation(t: Trials) -> None:
    Xc = t.X - t.X.mean(-1, keepdims=True)
    Xn = Xc / np.linalg.norm(Xc, axis=-1, keepdims=True)
    corr = np.mean(Xn @ Xn.transpose(0, 2, 1), axis=0)
    n = len(t.channels)
    write_csv(pd.DataFrame({"channel_a": np.repeat(t.channels, n), "channel_b": np.tile(t.channels, n),
                            "mean_pearson_correlation": corr.ravel()}), OUT_DIR / "channel_correlation.csv")


def _class_band_correlation(z: np.ndarray, t: Trials) -> None:
    rows = []
    for c in range(4):
        ind = (t.y == c).astype(np.float64)
        r = np.einsum("ncb,n->cb", z, ind - ind.mean()) / (len(ind) * ind.std())
        for ch, name in enumerate(t.channels):
            for b, band in enumerate(BANDS):
                rows.append({"measure": "one_vs_rest_pearson_r", "channel": name, "band": band, "class": CLASSES[c], "value": r[ch, b]})
    eta = sum((t.y == c).sum() * z[t.y == c].mean(0) ** 2 for c in range(4)) / len(z)
    for ch, name in enumerate(t.channels):
        for b, band in enumerate(BANDS):
            rows.append({"measure": "eta_squared", "channel": name, "band": band, "class": "all", "value": eta[ch, b]})
    write_csv(pd.DataFrame(rows), OUT_DIR / "class_band_power_correlation.csv")


def _topomaps(t: Trials, bp: np.ndarray) -> None:
    subjects = np.unique(t.subject)
    diff = np.zeros((4, len(t.channels), len(BANDS)))
    for s in subjects:
        m = t.subject == s
        overall = bp[m].mean(0)
        for c in range(4):
            diff[c] += bp[m & (t.y == c)].mean(0) - overall
    diff /= len(subjects)
    rows = [{"band": band, "class": CLASSES[c], "channel": name, "delta_log10_power_vs_all_class_mean": diff[c, ch, b]}
            for b, band in enumerate(BANDS) for c in range(4) for ch, name in enumerate(t.channels)]
    write_csv(pd.DataFrame(rows), OUT_DIR / "class_topomaps.csv")


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
    arr = np.vstack([mean_ac, mean_ac.mean(0)[None]])
    labels = list(t.channels) + ["channel_mean"]
    write_csv(pd.DataFrame({"lag_ms": np.tile(lags_ms, len(labels)), "channel": np.repeat(labels, len(lags_ms)),
                            "autocorrelation": arr.ravel()}), OUT_DIR / "autocorrelation.csv")


def _session_shift(bp: np.ndarray, t: Trials) -> None:
    rows = []
    for s in np.unique(t.subject):
        m1 = bp[(t.subject == s) & (t.session == 0)].mean(0)
        m2 = bp[(t.subject == s) & (t.session == 1)].mean(0)
        for b, band in enumerate(BANDS):
            for ch, name in enumerate(t.channels):
                rows.append({"subject": int(s), "band": band, "channel": name, "mean_log10_power_session1": m1[ch, b],
                             "mean_log10_power_session2": m2[ch, b], "session2_minus_session1": m2[ch, b] - m1[ch, b]})
    write_csv(pd.DataFrame(rows), OUT_DIR / "session_shift.csv")


def _trial_rms(t: Trials) -> None:
    rms = np.sqrt(np.mean(t.X.astype(np.float64) ** 2, axis=(1, 2)))
    rows, trials = [], []
    for s in np.unique(t.subject):
        for sess in (0, 1):
            m = (t.subject == s) & (t.session == sess)
            rows.append({"subject": int(s), "session": sess + 1, **box_stats(rms[m])})
            trials.append(pd.DataFrame({"subject": int(s), "session": sess + 1, "trial": np.arange(m.sum()), "rms_uV": rms[m]}))
    write_csv(pd.DataFrame(rows), OUT_DIR / "trial_rms.csv")
    write_csv(pd.concat(trials), OUT_DIR / "trial_rms_trials.csv")


def export_data_statistics_csvs(data: RawData, processed: ProcessedData) -> None:
    """Write one CSV per data_statistics figure to results/data_statistics/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = _stack(data, processed)
    bp = _log_band_power(t.X)
    z = _zscore(bp, t)
    _psd(data, t)
    _classes(t)
    _amplitude(t)
    _band_power_hist(z, t)
    _channel_correlation(t)
    _class_band_correlation(z, t)
    _topomaps(t, bp)
    _autocorrelation(t)
    _session_shift(bp, t)
    _trial_rms(t)
    print(f"Saved data-statistics CSVs to {OUT_DIR}")
