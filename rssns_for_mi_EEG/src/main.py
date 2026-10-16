from download_data import download_data
from preprocess_data import preprocess_data

import warnings
warnings.filterwarnings("ignore", message="Montage name 'standard_1005'")

from moabb.datasets import BNCI2014_001
print(BNCI2014_001().interval)

# Cache MOABB dataset so it doesn't need to re-download everytime
data = download_data( redo=False )

s = next(iter(data[1])); r = next(iter(data[1][s]))
raw = data[1][s][r]
print(list(data[1]), s, r)
print(raw.ch_names)
print(set(raw.annotations.description))
print(raw.annotations.onset[:6], raw.annotations.duration[:6])
print(raw.info["sfreq"], raw.times[-1])

# 4 - 40 Hz bandpass
# Resample to 128 Hz
processed = preprocess_data( data )

s = processed[1]["0train"]   # subject 1, session 1
print(s.X.shape)

import numpy as np
import preprocess_data as pp

c3, c4 = 7, 11  # indices from your ch_names list
for offset in (0.0, 2.0):
    pp.CUE_OFFSET_S = offset
    diffs = []
    for subject, sessions in pp.preprocess_data(data).items():
        s = sessions["0train"]
        power = np.log((s.X ** 2).mean(axis=-1))   # (trials, channels)
        lat = power[:, c3] - power[:, c4]
        diffs.append(lat[s.y == 0].mean() - lat[s.y == 1].mean())  # left - right
    print(offset, np.mean(diffs), np.mean(np.array(diffs) > 0))

# Demonstrate it's prefiltered
#data_statistics()

# Make and train EEGNet ANN and benchmark

# Convert EEGNet ANN to SNN and benchmark

# Make and train RSNN model and benchmark

# Comparative metrics between models
