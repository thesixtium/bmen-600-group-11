# v2.0: what the temporal and spatial filters of the trained EEGNet extract

> EEGNet's first layer has 4 temporal filters (32 taps at 128 Hz, 250 ms). Each one feeds 2 spatial filters over the 22 channels. This folder shows the frequency response of the temporal filters and the channel weights of the spatial filters, from the fine-tuned weights of all 90 folds of the v1 run (`experiments/v1`, run `20261029_141407`). No training and no data are used.

## 0. Status

| Item | State | Your open action |
|---|---|---|
| `filters_example.png` | One fold (target subject 3, repeat 0): kernels, frequency responses, channel weights | None |
| `filters_all_folds.png` | All 360 temporal filters, their peak frequencies, mean channel weight | None |
| `filters_summary.csv` | Per fold and filter: peak frequency, -3 dB band, band energy shares, top 3 channels | None |

## 1. Findings

| # | Finding | Numbers (`python plot_filters.py`) |
|---|---|---|
| 1 | The filters select the motor rhythms | Peak inside the 4 to 40 Hz input band: 29 % of the 360 filters in mu (8 to 13 Hz), 47 % in beta (13 to 30 Hz), 20 % in 30 to 40 Hz, 3 % in 4 to 8 Hz. Median peak 18.7 Hz. The largest group peaks at 10 to 14 Hz |
| 2 | They suppress the 4 to 8 Hz range | Median energy share 0.01, against 0.11 for a flat filter. The median response is about 20 dB below the peak near 5 Hz |
| 3 | The largest spatial weights are over the sensorimotor area | Mean absolute weight: CP3 0.154, C3 0.147, CP4 0.147, Cz 0.130, C4 0.128. Lowest: Fz 0.097. The spread is small (factor 1.6) |

## 2. Limits

| Limit | Detail |
|---|---|
| Peak outside the input band | 18 % of the filters have their largest gain below 4 Hz. The preprocessing removes that range, so it has no effect. The table uses the peak inside 4 to 40 Hz |
| Weights are not importance | A large spatial weight shows where a filter looks, not how much the class decision uses it. Channel amplitudes differ, and the 2 spatial filters of one temporal filter can cancel |
| Batch normalization | It only scales each filter output, so the shape of the response is unchanged. Responses are shown relative to each filter's peak |
| Source run | These are the EEGNet weights of the v1 run, with the same model settings as the main pipeline. Its checkpoints are not in the repository (size) |
| Channel order | Taken from the dataset definition in MOABB (22 EEG channels, 3 EOG channels dropped). The cached data stores no channel names |
