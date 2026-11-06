from dataclasses import dataclass

import mne
import numpy as np
from mne.io import BaseRaw
from numpy.typing import NDArray

from benchmark_config import FRONT_END_CFG
from download_data import RawData

# EEGNet preprocessing parameters
LOW_HZ: float = 4.0
HIGH_HZ: float = 40.0
SFREQ: int = 128
EPOCH_START_S: float = 0.5  # seconds after cue onset
EPOCH_END_S: float = 2.5
CUE_OFFSET_S: float = 0.0   # Seconds from the annotation onset to the cue.
EVENT_ID: dict[str, int] = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}


@dataclass(frozen=True)
class SessionData:
    """All trials of one subject's session, in chronological order.

    Trial i of the session is X[i], y[i], run[i].
    """

    subject: int            # Subject number
    session: str            # "0train" or "1test"
    X: NDArray[np.float32]  # (n_trials, 22, 256), 4-40 Hz: input of the ANN and the converted SNN
    y: NDArray[np.int64]    # (n_trials,) 0=left hand, 1=right hand, 2=feet, 3=tongue
    run: NDArray[np.int64]  # (n_trials,) run index (0-5) each trial came from
    # (n_trials, 44, 256): the same trials split into the fixed mu and beta bands (FrontEndConfig); channels 0-21 are
    # mu, 22-43 are beta, same channel order as X. Input of the RSNN and the RNN control.
    X_bands: NDArray[np.float32]


ProcessedData = dict[int, dict[str, SessionData]]  # subject -> session -> data


def _process_run(raw: BaseRaw) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]:
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)

    # 22 EEG channels only (drops the 3 EOG channels and the STI channel)
    eeg_channels = [c for c in raw.ch_names if not c.startswith("EOG") and c != "STI"]
    assert len(eeg_channels) == 22, eeg_channels
    raw = raw.copy().pick(eeg_channels)
    raw.filter(LOW_HZ, HIGH_HZ, verbose=False)
    raw, events = raw.resample(SFREQ, events=events, verbose=False)

    X, y = _epoch(raw, events)

    # Frequency front end (v1.3): band-pass copies of the SAME continuous 4-40 Hz, 128 Hz recording, epoched with the
    # same events. Zero-phase IIR (Butterworth, forward-backward), fixed, nothing fitted to data.
    band_X = []
    for low, high in FRONT_END_CFG.bands:
        band_raw = raw.copy().filter(
            low, high, method="iir", iir_params={"order": FRONT_END_CFG.filter_order, "ftype": "butter"},
            phase="zero", verbose=False,
        )
        X_band, y_band = _epoch(band_raw, events)
        assert np.array_equal(y_band, y), "band epochs must be the same trials as the 4-40 Hz epochs"
        band_X.append(X_band)
    X_bands = np.concatenate(band_X, axis=1)  # (n_trials, 22 * n_bands, 256): all mu channels, then all beta channels
    return X, y, X_bands


def _epoch(raw: BaseRaw, events: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Cut the 0.5-2.5 s after-cue window (256 samples) of every trial out of a continuous 128 Hz recording."""
    # -1/SFREQ so the window is exactly 256 samples (MNE includes both endpoints)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=CUE_OFFSET_S + EPOCH_START_S,
        tmax=CUE_OFFSET_S + EPOCH_END_S - 1 / SFREQ,
        baseline=None,  # no baseline correction
        reject=None,  # keep artifact-marked trials
        preload=True,
        verbose=False,
    )
    assert len(epochs) == len(events), "Some trials were dropped"

    X: NDArray[np.float32] = epochs.get_data().astype(np.float32)
    y: NDArray[np.int64] = (epochs.events[:, 2] - 1).astype(np.int64)
    return X, y


def preprocess_data(data: RawData) -> ProcessedData:
    """4-40 Hz band-pass, resample to 128 Hz, epoch 0.5-2.5 s after cue onset; plus the mu/beta band copies."""
    out: ProcessedData = {}

    for subject, sessions in data.items():
        out[subject] = {}
        for session, runs in sessions.items():
            Xs: list[NDArray[np.float32]] = []
            ys: list[NDArray[np.int64]] = []
            Xbs: list[NDArray[np.float32]] = []
            run_ids: list[NDArray[np.int64]] = []

            for run_name in sorted(runs, key=int):
                X, y, X_bands = _process_run(runs[run_name])
                Xs.append(X)
                Xbs.append(X_bands)
                ys.append(y)
                run_ids.append(np.full(len(y), int(run_name), dtype=np.int64))

            out[subject][session] = SessionData(
                subject=subject,
                session=session,
                X=np.concatenate(Xs),
                y=np.concatenate(ys),
                run=np.concatenate(run_ids),
                X_bands=np.concatenate(Xbs),
            )

    return out