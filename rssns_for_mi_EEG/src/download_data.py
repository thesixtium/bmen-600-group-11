import os
from pathlib import Path

from mne.io import BaseRaw
from moabb.datasets import BNCI2014_001

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data" # DATA_DIR: <project>/data , current dir. is <project>/src


RawData = dict[int, dict[str, dict[str, BaseRaw]]] # subject -> session name ("0train" / "1test") -> run name -> raw EEG recording


def download_data(redo: bool) -> RawData:
    """
    Download BNCI2014_001 into <project>/data and return it raw.

    Args:
    redo=True  -> re-download every file, even if already cached.
    redo=False -> only download files that are not already cached.

    Returs:
    unprocessed MNE Raw objects (dict) -> {subject: {session: {run: MNE_Raw_Data_Object}}}.
    
    where;
    subject : [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sessions : ['0train', '1test']
    run : ['0', '1', '2', '3', '4', '5']

    """

    DATA_DIR.mkdir(parents=True, exist_ok=True) # make the directory if doesn't exist

    # Make MOABB/MNE read from and write to our data folder (this process only)
    os.environ["MNE_DATA"] = str(DATA_DIR)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(DATA_DIR)

    dataset: BNCI2014_001 = BNCI2014_001()

    if redo:
        for subject in dataset.subject_list:
            dataset.data_path(subject, path=str(DATA_DIR), force_update=True, update_path=False)

    return dataset.get_data()