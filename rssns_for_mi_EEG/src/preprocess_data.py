from dataclasses import dataclass

import mne
import numpy as np
from mne.io import BaseRaw
from numpy.typing import NDArray

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
    X: NDArray[np.float32]  # (n_trials, 22, 256)
    y: NDArray[np.int64]    # (n_trials,) 0=left hand, 1=right hand, 2=feet, 3=tongue
    run: NDArray[np.int64]  # (n_trials,) run index (0-5) each trial came from


ProcessedData = dict[int, dict[str, SessionData]]  # subject -> session -> data


def _process_run(raw: BaseRaw) -> tuple[NDArray[np.float32], NDArray[np.int64]]:

    """
    Args:

    raw (BaseRaw) -> MNE_Raw_Data_Object

    Returns

    X (NDArray) -> (trials, channels , time points)
    y (NDArray) -> (trials,)

    """
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)

    # 22 EEG channels only (drops the 3 EOG channels and the STI channel)
    # raw.ch_names: ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'EOG1', 'EOG2', 'EOG3', 'STI']
    eeg_channels = [c for c in raw.ch_names if not c.startswith("EOG") and c != "STI"]
    
    assert len(eeg_channels) == 22, eeg_channels
    
    raw = raw.copy().pick(eeg_channels)

    raw = raw.filter(LOW_HZ, HIGH_HZ, verbose=False) # filter raw eeg signals before epoching

    raw, events = raw.resample(SFREQ, events=events, verbose=False) # resampling 

    # -1/SFREQ so the window is exactly SFREQ*DURATION samples (MNE includes both endpoints)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=CUE_OFFSET_S + EPOCH_START_S,
        tmax=CUE_OFFSET_S + EPOCH_END_S - 1 / SFREQ,
        baseline=None,  # no baseline correction
        reject=None,  # keep artifact-marked trials (artifacts are annotated)
        preload=True,
        verbose=False,
    )
    assert len(epochs) == len(events), "Some trials were dropped"

    X: NDArray[np.float32] = epochs.get_data().astype(np.float32)
    # epochs.event : [[sample 0 class]]
    y: NDArray[np.int64] = (epochs.events[:, 2] - 1).astype(np.int64) # class labels changed from [1, 2, 3, 4 ] to [0, 1, 2, 3]

    return X, y


def preprocess_data(data: RawData) -> ProcessedData:
    """
    Resample to 128 Hz, epoch 0.5-2.5 s after cue onset 4-40 Hz band-pass

    Args:
    Rawdata (dict) -> {subject: {session: {run: MNE_Raw_Data_Object}}}.

    where;
    subject : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sessions : ['0train', '1test']
    run : ['0', '1', '2', '3', '4', '5'].
    
    Return:
    
    out dict () -> {subject: {session: SessionData(subject, session, X, y, run_ids)}}"""

    out: ProcessedData = {}

    for subject, sessions in data.items(): # (subject,  {session: {run: MNE_Raw_Data_Object}})
        out[subject] = {}
        for session, runs in sessions.items(): # (session: {run: MNE_Raw_Data_Object})

            # Initialize 
            Xs: list[NDArray[np.float32]] = []
            ys: list[NDArray[np.int64]] = []
            run_ids: list[NDArray[np.int64]] = []

            for run_name in sorted(runs, key=int): 
                X, y = _process_run(runs[run_name]) 
                Xs.append(X)
                ys.append(y)
                run_ids.append(np.full(len(y), int(run_name), dtype=np.int64))


            out[subject][session] = SessionData(
                subject=subject,
                session=session,
                X=np.concatenate(Xs), # (# trials per session, channels, time)
                y=np.concatenate(ys), # (# trials per session)
                run=np.concatenate(run_ids),
            )

    return out