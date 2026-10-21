# BMEN 600 Group 11: Code Documentation

Documentation of the EEG motor-imagery benchmark (EEGNet ANN vs. converted SNN vs. RSNN) on BCI Competition IV dataset 2a, written from the uploaded source files. Everything below describes what the code does, not what the plan says it should do. Deviations from the research plan are collected in Section 8. Anything that looks wrong is collected in Section 9.

---

## 0. Scope and method

- **Static analysis only.** I read every file and checked the arithmetic (parameter counts, tensor shapes, normalization algebra). I did not execute the code: PyTorch and snnTorch are not installed in my sandbox, and the dataset is not available there. Statements about runtime behaviour (e.g. "neurons saturate") are reasoning from the code, and are marked as such.
- **Files covered:** `main.py`, `download_data.py`, `preprocess_data.py`, `data_statistics.py`, `folds.py`, `benchmark_config.py`, `models.py`, `ann_to_snn.py`, `training.py`, `benchmark.py`, `benchmark_plots.py`.
- **Plan source:** `BMEN_600_Midterm_Research_Plan_Group_11` (PDF/DOCX, identical content).
- **Image placeholders** are marked `[IMAGE PLACEHOLDER]` and each carries a prompt for an image generator. No images were generated.
- Notation: `B` = batch size, `H` = RSNN hidden size, `T` = number of simulation steps, µV = microvolts.

---

## 1. Repository map and execution flow

### 1.1 What each file does

| File | Role | Libraries it relies on |
|---|---|---|
| `main.py` | Top-level script; runs the whole pipeline in order | stdlib, project modules |
| `download_data.py` | Downloads/caches BNCI2014_001 and returns raw MNE objects | `moabb`, `mne` |
| `preprocess_data.py` | Filter, resample, epoch; produces `ProcessedData` | `mne`, `numpy` |
| `data_statistics.py` | 13 descriptive figures of raw and preprocessed data | `matplotlib`, `mne`, `scipy.signal.welch`, `numpy` |
| `folds.py` | Builds the 90 shared folds and assembles per-fold arrays | `numpy`, `scikit-learn` |
| `benchmark_config.py` | All hyperparameters/constants as frozen dataclasses | stdlib |
| `models.py` | `EEGNet`, `RecurrentLIFNet`, analytic parameter counts | `torch`, `snntorch.surrogate` (surrogate gradient only) |
| `ann_to_snn.py` | ANN → SNN conversion and spiking simulation | `torch`, `numpy` |
| `training.py` | One shared training loop for ANN and RSNN | `torch` |
| `benchmark.py` | Runs the three models per fold, computes and saves metrics | `torch`, `scikit-learn`, `pandas` |
| `benchmark_plots.py` | Descriptive result figures | `matplotlib`, `pandas` |

### 1.2 Pipeline (order of calls in `main.py`)

```
download_data(redo=False)                      raw MNE data   data[subject][session][run]
        │
        ▼
preprocess_data(data)                          ProcessedData  [subject][session] -> X (n,22,256) volts, y, run
        │
        ├──► data_statistics(data, processed)  figures -> results/data_statistics/   (runs every time)
        │
        ▼
generate_folds(processed)                      90 Fold objects (9 targets x 10 repeats)
        │
        ▼
prepare_run_dir(all_folds)                     results/benchmark/<run id>/  + config.json + folds.json
        │
        ▼
filter_folds(env BENCH_TARGETS / BENCH_REPEATS)   subset of the 90 folds for this process
        │
        ▼
run_ann_benchmark   ──► checkpoints/ann/*.pt (pretrained + finetuned), metrics, predictions
        │
        ▼
run_snn_benchmark   ──► converts each fine-tuned ANN checkpoint (no training), metrics, predictions
        │
        ▼
run_rsnn_benchmark  ──► trains RSNN (pretrain + finetune), metrics, predictions
        │
        ▼
collect_results(run_dir)   ──► results.csv, predictions.csv, confusion_matrices.csv
        │
        ▼
plot_benchmarks(run_dir)   ──► figures/*.png
        │
        ▼
# "Comparative metrics between models"   <- comment only, nothing implemented
```

### 1.3 Environment variables

| Variable | Read in | Effect |
|---|---|---|
| `BENCH_RUN_ID` | `benchmark.prepare_run_dir` | Run directory name. Unset → timestamp `YYYYmmdd_HHMMSS`. Setting it resumes/shares a run directory |
| `BENCH_TARGETS` | `main.py` | Comma-separated target subjects to run, e.g. `1,2,3` |
| `BENCH_REPEATS` | `main.py` | Comma-separated repeat indices to run |
| `BENCH_QUICK` | `benchmark_config.py` | `"1"` turns on quick mode (also forced on by `QUICK_MODE = True`, see flag F2) |

---

## 2. Configuration (`benchmark_config.py`)

### 2.1 Constants

| Name | Value | Meaning |
|---|---|---|
| `RESULTS_DIR` | `<project>/results/benchmark` | `Path(__file__).parent.parent / "results" / "benchmark"` |
| `N_CHANNELS` | 22 | EEG channels |
| `N_SAMPLES` | 256 | Samples per epoch (2 s at 128 Hz) |
| `N_CLASSES` | 4 | left hand, right hand, feet, tongue → labels 0–3 |
| `TRAIN_SESSION` | `"0train"` | Session 1 (pretraining subjects + target fine-tuning) |
| `TEST_SESSION` | `"1test"` | Session 2 (held-out test) |
| `INPUT_SCALE` | `1e6` | Volts → microvolts, applied in `folds._stack` |

### 2.2 Dataclasses

| Dataclass | Field | Value |
|---|---|---|
| `FoldConfig` | `n_repeats` | 10 |
| | `n_pretrain_subjects` | 5 (of 8 non-target; other 3 = pretraining validation) |
| | `finetune_val_fraction` | 0.2 |
| | `base_seed` | 2026 (fold seed = `2026 + 100*target + repeat`) |
| `EEGNetConfig` | `F1, D, F2` | 4, 2, 8 (`F2` is typed in, not computed) |
| | `kernel_length` / `separable_kernel` | 32 / 16 |
| | `pool1` / `pool2` | 4 / 8 |
| | `depthwise_max_norm` / `dense_max_norm` | 1.0 / 0.25 |
| | `bn_eps` / `bn_momentum` | 1e-3 / 0.01 |
| `TrainConfig` | `pretrain_epochs` / `finetune_epochs` | 500 / 100 |
| | `pretrain_dropout` / `finetune_dropout` | 0.25 / 0.5 |
| | `learning_rate` | 1e-3 (Adam) |
| | `batch_size` | 16 |
| `SNNConfig` | `n_steps` | 256 |
| | `threshold` | 1.0 |
| | `calibration_percentile` | 99.9 |
| | `signed_spikes` | True |
| | `negative_floor` | 1.0 (membrane floor of signed neurons, in thresholds) |
| | `tie_break_membrane` | True |
| `RSNNConfig` | `n_hidden` | `None` → derived to match SNN parameter count (gives 17) |
| | `n_steps` | 256 |
| | `beta` | 0.9 (fixed, not learned) |
| | `threshold` | 1.0 |
| | `surrogate_slope` | 25.0 |
| | `recurrent_init_gain` | 1.0 (`W_rec ~ N(0, (1/√H)²)`) |
| | `batch_size` | 64 |

Module-level instances: `ANN_TRAIN = TrainConfig()`, `RSNN_TRAIN = TrainConfig(batch_size=64)`, plus `FOLD_CFG`, `EEGNET_CFG`, `SNN_CFG`, `RSNN_CFG`.

### 2.3 Quick mode

```python
QUICK_MODE: bool = True            # as uploaded
QUICK = QUICK_MODE or os.environ.get("BENCH_QUICK") == "1"
if QUICK:
    ANN_TRAIN  -> pretrain_epochs=2, finetune_epochs=2
    RSNN_TRAIN -> pretrain_epochs=2, finetune_epochs=2
    SNN_CFG    -> n_steps=32
    RSNN_CFG   -> n_steps=64
```

As uploaded, quick mode is **on**. See flags F2, F3 and F4: in `benchmark.py` only the `ANN_TRAIN`/`RSNN_TRAIN` overrides take effect.

---

## 3. Data pipeline

### 3.1 `download_data.py`

- `download_data(redo)` creates `<project>/data`, sets `MNE_DATA` and `MNE_DATASETS_BNCI_PATH` to it (this process only), instantiates `moabb.datasets.BNCI2014_001`, and returns `dataset.get_data()`.
- `redo=True` forces a re-download per subject via `dataset.data_path(..., force_update=True, update_path=False)`. `main.py` calls it with `redo=False`.
- Return type `RawData = dict[int, dict[str, dict[str, BaseRaw]]]`: `subject -> "0train"/"1test" -> run name ("0".."5") -> Raw` (250 Hz, unprocessed).
- **Implementation:** library (MOABB + MNE). No custom loading code.

### 3.2 `preprocess_data.py`

Constants: `LOW_HZ=4`, `HIGH_HZ=40`, `SFREQ=128`, `EPOCH_START_S=0.5`, `EPOCH_END_S=2.5`, `CUE_OFFSET_S=0.0`, `EVENT_ID={"left_hand":1,"right_hand":2,"feet":3,"tongue":4}`.

Per run, `_process_run(raw)`:

```
raw (250 Hz, 22 EEG + 3 EOG + STI)
  │ events_from_annotations(raw, event_id=EVENT_ID)       events at ANNOTATION ONSETS (250 Hz sample indices)
  │ keep channels not starting with "EOG" and != "STI"    assert exactly 22
  │ raw.filter(4, 40)                                      MNE default FIR, zero-phase, on the continuous run
  │ raw.resample(128, events=events)                       events re-indexed to 128 Hz
  │ mne.Epochs(tmin = 0.0 + 0.5,
  │            tmax = 0.0 + 2.5 - 1/128,
  │            baseline=None, reject=None, preload=True)
  │ assert len(epochs) == len(events)                      no trial may be dropped
  ▼
X (n_trials, 22, 256) float32, VOLTS        y = epochs.events[:,2] - 1  -> 0..3
```

- `tmax = 2.5 - 1/128` because MNE includes both endpoints, so the window is exactly 256 samples (0.5 s … 2.4921875 s).
- No baseline correction, no artifact rejection, no re-referencing, no channel normalization.
- `preprocess_data(data)` loops subjects → sessions → runs (sorted numerically by `int(run_name)`), concatenates runs in order, and stores a `SessionData(subject, session, X, y, run)`. `run[i]` is the run index of trial `i`.
- Expected sizes: 288 trials per session (6 runs × 48), 72 per class.
- Units are volts here. The ×1e6 happens later in `folds._stack` (and in `data_statistics._stack`).
- **Implementation:** MNE library calls (filter, resample, Epochs). No custom signal processing.

### 3.3 `data_statistics.py`

Descriptive only; nothing here feeds a model. `data_statistics(data, processed)` creates `results/data_statistics/`, stacks all trials of all subjects and both sessions (`Trials`: X in µV, y, subject, session 0/1, channel names), computes log10 band power (`welch`, `nperseg=256`, sum of PSD over bins × Δf) for mu (8–13 Hz) and beta (13–30 Hz), z-scores it per subject and session, and then writes the figures below. Every figure gets a "How to read / Takeaways" caption from the static `NOTES` dict.

| Figure file | Function | What is computed |
|---|---|---|
| `psd_overview.png` | `_psd_plots` | Welch PSD (dB) per channel: raw continuous runs at 250 Hz (`nperseg=500`) vs. preprocessed epochs at 128 Hz. Prints a dB-change table at 2…60 Hz |
| `psd_notch_zoom.png` | `_psd_plots` | 35–64 Hz zoom of the channel-mean PSD, raw vs. preprocessed |
| `psd_per_subject.png` | `_psd_plots` | Channel-mean PSD, one line per subject |
| `class_distribution.png` | `_class_distribution` | Trials per class/subject/session. **Asserts every cell equals 72** after saving |
| `amplitude_hist_pooled.png` | `_amplitude_hist` | Log-density amplitude histograms per class at C3/Cz/C4, pooled |
| `amplitude_hist_per_subject.png` | `_amplitude_hist` | Same at C3, one panel per subject |
| `band_power_hist.png` | `_band_power_hist` | z-scored log band-power histograms per class, C3/C4 × mu/beta |
| `channel_correlation.png` | `_channel_correlation` | Per-trial Pearson channel×channel matrix, averaged over trials |
| `class_band_power_correlation.png` | `_class_band_correlation` | One-vs-rest Pearson r of class indicator vs. z-scored band power; plus η² (class-explained variance) |
| `class_topomaps.png` | `_topomaps` | Per-class minus all-class mean log band power (not z-scored), averaged over subjects, on the 10-20 scalp map |
| `autocorrelation.png` | `_autocorrelation` | FFT-based per-trial autocorrelation up to 0.5 s lag, averaged |
| `session_shift.png` | `_session_shift` | Session 2 minus session 1 mean log band power per subject/channel |
| `trial_rms.png` | `_trial_rms` | Box plots of per-trial RMS (µV, log axis) per subject/session |

---

## 4. Folds (`folds.py`)

### 4.1 Data structures

- `Fold(target, repeat, seed, pretrain_subjects, val_subjects, finetune_train_idx, finetune_val_idx)`: frozen, tuples of ints. Indices point into the **target's session-1 trials**.
- `Split(X, y)`: `X` is `(n,22,256)` float32 in µV.
- `FoldData(pre_train, pre_val, ft_train, ft_val, test)`.

### 4.2 `generate_folds`

For each target subject (1–9) and each repeat (0–9):

1. `seed = 2026 + 100*target + repeat`; `rng = np.random.default_rng(seed)`.
2. `chosen = rng.permutation(other 8 subjects)`; first 5 (sorted) = pretraining set, last 3 (sorted) = pretraining validation set.
3. `train_test_split(arange(288), test_size=0.2, stratify=y_target, random_state=seed)` on the target's session 1 → 230 fine-tune-train, 58 fine-tune-validation trials.

Result: 9 × 10 = 90 folds. The same fold list is used by all three models.

### 4.3 `build_fold_data`

```
For one fold (target = T):

  other 8 subjects ─┬─ 5 subjects, session 1  ──► pre_train   5 x 288 = 1440 trials
                    └─ 3 subjects, session 1  ──► pre_val     3 x 288 =  864 trials

  target T, session 1 (288) ─┬─ 80 % stratified ─► ft_train   230 trials
                             └─ 20 % stratified ─► ft_val      58 trials

  target T, session 2 (288) ───────────────────► test        288 trials (72 per class)
```

`_stack` concatenates sessions, multiplies by `INPUT_SCALE` (1e6) and casts to float32. `filter_folds` selects by target and/or repeat. `save_folds` writes the config and all folds to JSON.

---

## 5. Shared training loop (`training.py`)

Used by both trainable models (EEGNet and RSNN). Both map `(B,22,256)` to 4 logits.

`fit(model, train, val, epochs, batch_size, learning_rate, dropout, seed, device)`:

1. Move train/val tensors fully to the device. `set_dropout(model, dropout)` sets `p` on every `nn.Dropout`.
2. `Adam(lr)`; a CPU `torch.Generator` seeded with `seed` shuffles the training order each epoch; `torch.manual_seed(seed)` seeds dropout.
3. Per epoch: train mode, mini-batches, `cross_entropy`, `backward`, `step`, then `model.apply_max_norm()` **if the model has it** (EEGNet does; RSNN does not).
4. Validation loss in eval mode (mean cross-entropy). The state dict with the strictly lowest validation loss is deep-copied; at the end it is loaded back into `model`.
5. Every history entry gets a `best_epoch` field.

`pretrain_then_finetune`:

```
model ─► fit(pre_train, pre_val, epochs=500, dropout=0.25, seed=S)    ─► best-val weights loaded
      ─► fit(ft_train,  ft_val,  epochs=100, dropout=0.5,  seed=S+1)  ─► best-val weights loaded
```

A fresh Adam optimizer is created in each `fit`. The test set is never passed in. Optimizer steps per epoch: ANN pretrain 90, ANN fine-tune 15, RSNN pretrain 23, RSNN fine-tune 4 (batch sizes 16 and 64).

`predict_logits` runs eval mode in chunks of 64 and returns logits as NumPy.

---

## 6. Architectures

### 6.0 Summary

| | EEGNet ANN | Converted SNN | RSNN |
|---|---|---|---|
| Class / location | `models.EEGNet` | `ann_to_snn.ConvertedEEGNetSNN` | `models.RecurrentLIFNet` |
| Built with | PyTorch `torch.nn` layers, written from scratch following Lawhern et al. 2018 Table 2 | Plain PyTorch, written from scratch following Rueckauer et al. 2017. No conversion library | PyTorch `nn.Linear` + hand-written LIF loop. `snntorch` used **only** for `surrogate.fast_sigmoid` |
| Trained? | Yes (Adam, BP) | No, weights come from the fine-tuned ANN | Yes (Adam, surrogate-gradient BPTT) |
| Trainable params | 796 | 776 (frozen; includes folded biases) | 752 at H = 17 |
| Time axis | None: EEG time is a spatial axis of the conv layers | 256 simulation steps, **independent** of EEG time; trial injected as constant current | 256 steps = 256 EEG samples, one sample per step |
| Neuron model | ELU | IF, no leak, reset by subtraction; signed (±1) spikes in hidden layers | LIF, β = 0.9, reset by subtraction (delayed one step), Heaviside + fast-sigmoid surrogate |
| Recurrence | None | None | Dense H×H, trained |
| Output decision | argmax logit | argmax output spike count, ties by final membrane | argmax logit of readout on mean spike rate |

---

### 6.1 EEGNet ANN (`models.EEGNet`)

**How it is implemented:** hand-written with standard PyTorch layers (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.AvgPool2d`, `nn.Dropout`, `nn.Linear`, `nn.ELU`). No EEG-model library (no Braindecode/TorchEEG). The Keras-specific behaviours are reproduced manually: BN `eps=1e-3`, BN momentum 0.01 (PyTorch equivalent of Keras 0.99), and Keras `MaxNorm` constraints via `apply_max_norm()`.

#### Layer table (EEGNet-4,2, kernel 32, input 22 × 256)

| # | Attribute | Operation | Output shape | Trainable params |
|---|---|---|---|---|
| 0 | | input `(B,22,256)` → `unsqueeze(1)` | `(B,1,22,256)` | |
| 1 | `conv1` | `Conv2d(1→4, k=(1,32), padding="same", bias=False)` | `(B,4,22,256)` | 128 |
| 2 | `bn1` | `BatchNorm2d(4)` | same | 8 |
| 3 | `depthwise` | `Conv2d(4→8, k=(22,1), groups=4, bias=False)`, weight shape (8,1,22,1) | `(B,8,1,256)` | 176 |
| 4 | `bn2` | `BatchNorm2d(8)` | same | 16 |
| 5 | `act` | `ELU` | same | 0 |
| 6 | `pool1` | `AvgPool2d((1,4))` | `(B,8,1,64)` | 0 |
| 7 | `drop1` | `Dropout` (p set by `set_dropout`) | same | 0 |
| 8 | `sep_depth` | `Conv2d(8→8, k=(1,16), padding="same", groups=8, bias=False)` | `(B,8,1,64)` | 128 |
| 9 | `sep_point` | `Conv2d(8→8, k=1, bias=False)` | same | 64 |
| 10 | `bn3` | `BatchNorm2d(8)` | same | 16 |
| 11 | `act` | `ELU` (same module object as #5) | same | 0 |
| 12 | `pool2` | `AvgPool2d((1,8))` | `(B,8,1,8)` | 0 |
| 13 | `drop2` | `Dropout` | same | 0 |
| 14 | | `flatten(1)` (index = map × 8 + position) | `(B,64)` | |
| 15 | `fc` | `Linear(64→4)` | `(B,4)` logits | 260 |
| | | | **Total** | **796** |

BatchNorm running mean/variance are buffers, not counted. `models.eegnet_param_count` reproduces 796 analytically.

#### Data-flow diagram

```
 x  (B, 22, 256)          22 channels x 256 samples, microvolts
   │ unsqueeze(1)
   ▼
 (B, 1, 22, 256)
   │
   ▼  ┌─────────────────────────── BLOCK 1 ────────────────────────────┐
   │  conv1   4 temporal filters, length 32, 'same', no bias    128 p   │
   │          each filter slides along TIME on each channel separately  │
   ▼  └─────────────────────────────────────────────────────────────────┘
 (B, 4, 22, 256)
   │  bn1        BatchNorm2d(4)                                    8 p
   ▼
   │  depthwise  8 spatial filters, each 22 weights (one per channel),   176 p
   │             groups=4: temporal map g feeds output maps 2g and 2g+1
   │             max-norm 1.0 per output filter after every optimizer step
   ▼
 (B, 8, 1, 256)
   │  bn2 (16 p) ─► ELU ─► AvgPool (1,4) ─► Dropout
   ▼
 (B, 8, 1, 64)
   │
   ▼  ┌─────────────────────────── BLOCK 2 ────────────────────────────┐
   │  sep_depth  8 independent temporal filters, length 16, 'same'  128 p│
   │  sep_point  1x1 conv, every one of 8 inputs -> every one of 8 outputs 64 p
   ▼  └─────────────────────────────────────────────────────────────────┘
 (B, 8, 1, 64)
   │  bn3 (16 p) ─► ELU ─► AvgPool (1,8) ─► Dropout
   ▼
 (B, 8, 1, 8)
   │ flatten
   ▼
 (B, 64)
   │  fc   Linear 64 -> 4 (260 p), max-norm 0.25 per output unit
   ▼
 (B, 4)  logits   (softmax lives inside cross_entropy)
```

#### Connectivity details (from the code)

```
 Temporal conv (conv1)              4 filters, SHARED across the 22 channels
   ch1 ──f0──► map(f0,ch1)   ...   ch22 ──f0──► map(f0,ch22)        f0 gives 22 maps
   same for f1, f2, f3                                              total 4 x 22 maps

 Depthwise spatial conv (groups=4)
   maps of f0 (22 of them) ──┬─ w_a (22 weights) ─► out 0
                             └─ w_b (22 weights) ─► out 1
   maps of f1 ──┬─► out 2, out 3
   maps of f2 ──┬─► out 4, out 5
   maps of f3 ──┬─► out 6, out 7
   (no mixing between temporal-filter groups here)

 Separable conv
   sep_depth: out k uses only map k (8 separate length-16 kernels)
   sep_point: 8x8 weights, out j = sum over k of w[j,k] * map k

 Classifier
   each of the 4 logits is connected to all 64 pooled features
```

#### Behavioural details

- **Max-norm** is not part of `forward`. `apply_max_norm()` runs `weight.renorm_(2, 0, max)` on `depthwise.weight` (limit 1.0, per output filter) and `fc.weight` (limit 0.25, per output unit); `training.fit` calls it after each optimizer step.
- **Dropout probabilities** are constructed as 0.25 but overwritten by `set_dropout` to 0.25 (pretrain) and 0.5 (fine-tune).
- `padding="same"` with an even kernel (32 or 16) pads asymmetrically (one more sample on the right side).
- Checkpoints are saved as `{"pretrained": state, "finetuned": state}`.

**[IMAGE PLACEHOLDER: `fig_eegnet_architecture.png`]**

> **Prompt:** "Clean technical block diagram, white background, left-to-right flow, flat vector style with a muted blue/teal/orange palette. Title: 'EEGNet-4,2 as implemented (PyTorch)'. Boxes in order: 'Input 22 x 256 (µV)' → 'Temporal Conv2d 4 filters, kernel (1,32), same, no bias [128 params]' → 'BatchNorm [8]' → 'Depthwise Conv2d 8 filters, kernel (22,1), groups=4, max-norm 1 [176]' → 'BatchNorm [16]' → 'ELU' → 'AvgPool (1,4)' → 'Dropout' → 'Depthwise Conv2d 8 maps, kernel (1,16), same [128]' → 'Pointwise Conv2d 8→8 [64]' → 'BatchNorm [16]' → 'ELU' → 'AvgPool (1,8)' → 'Dropout' → 'Flatten (64)' → 'Dense 64→4, max-norm 0.25 [260]' → 'Logits (4)'. Under each box write the tensor shape: (1,22,256), (4,22,256), (8,1,256), (8,1,64), (8,1,8), (64), (4). Group the first six boxes in a rounded rectangle labelled 'Block 1' and the next six in 'Block 2'. Add a small inset showing the depthwise grouping: four temporal-filter groups, each feeding exactly two spatial filters. Footer: 'Total trainable parameters: 796'."

---

### 6.2 Converted SNN (`ann_to_snn.ConvertedEEGNetSNN`)

**How it is implemented:** written from scratch in plain PyTorch, following Rueckauer et al. 2017 (rate coding, data-based weight normalization, reset by subtraction). No conversion or SNN library is used (no snnTorch, Norse, sinabs, etc.). The code states that nothing is adapted from external sources. The module has no trainable parameters.

#### Construction (`__init__`)

1. `ann = deepcopy(ann).eval()`; the original is untouched.
2. **BatchNorm folding** (`_fold_batchnorm`): for each conv followed by BN,
   `scale = γ / √(running_var + eps)`, `W' = W · scale` (per output channel), `b' = β − running_mean · scale` (+ `conv.bias · scale` if present).
   Folded: `conv1+bn1`, `depthwise+bn2`, `sep_point+bn3`. Unchanged copies: `sep_depth`, `fc`. This is exact in eval mode.
3. **Resulting layers:** `conv1` (with bias), `depthwise` (with bias), `sep_depth` (no bias), `sep_point` (with bias), `fc` (with bias); `pool1`, `pool2` reused. All parameters get `requires_grad=False`.
4. **Calibration** (on the *target's fine-tuning training trials only*, 230 trials, passed in as one batch):

```
a1  = ELU( depthwise( conv1(x) ) )                         activations of "layer 1"
a2  = ELU( sep_point( sep_depth( pool1(a1) ) ) )           "layer 2"
out = fc( flatten( pool2(a2) ) )                           "layer 3"
lam0 = P99.9(|x|), lam1 = P99.9(|a1|), lam2 = P99.9(|a2|), lam3 = P99.9(|out|)
```

5. **Weight/bias rescaling** (in place):

| Parameter | Change |
|---|---|
| `conv1.weight` | × lam0 / lam1 |
| `conv1.bias` | ÷ lam1 |
| `depthwise.weight` | unchanged |
| `depthwise.bias` | ÷ lam1 |
| `sep_depth.weight` | × lam1 / lam2 |
| `sep_point.weight` | unchanged |
| `sep_point.bias` | ÷ lam2 |
| `fc.weight` | × lam2 / lam3 |
| `fc.bias` | ÷ lam3 |

Check of the algebra: layer 1's input current is `depthwise'(conv1'(x/lam0)) = z1 / lam1`, where `z1` is the ANN pre-activation; layer 2 receives rates `≈ a1/lam1` and produces `z2/lam2`; layer 3 produces `logits/lam3`. So every hidden rate is the ANN activation divided by its λ. Note that `lam0` cancels exactly (see flag F10).

6. `lambdas = (lam0, lam1, lam2, lam3)` is stored and written to the metrics JSON.

#### Parameter count

`conv1` 4×32 + 4 = 132; `depthwise` 176 + 8 = 184; `sep_depth` 128; `sep_point` 64 + 8 = 72; `fc` 260. **Total 776** (= 796 − 20 BN params + 20 folded biases; `models.converted_snn_param_count` agrees).

#### Neuron model (`_IFNeurons`)

One population object per layer, created fresh for every `simulate()` call.

```
step(I):
    v  += I
    pos = (v >= th);          v -= pos * th                 # reset by subtraction, max 1 spike / step
    if signed:                                              # hidden layers only
        allowed = (net - 1) >= -neg_rate * (t + 1)          # floor on the NET spike rate
        neg     = (v <= -th) & allowed
        v       = max( v + neg * th , -th * v_floor )       # membrane floor (default -1)
        spike   = pos - neg                                 # ternary: -1, 0, +1
    else:
        spike   = pos                                       # output layer: no floor, no negative spikes
    net += spike;  t += 1
    return spike
```

- `neg_rate = min(1, 1/lam_l)` for layer l: the net rate cannot fall below −1/λ, i.e. ELU's lower bound of −1 in normalized units.
- The negative branch therefore behaves like `max(z, −1)` instead of `exp(z) − 1`. This is a documented approximation (see F11).
- Output population `n3` is a plain IF layer; `n3.net` is the spike count and `n3.v` the final membrane potential.

#### Simulation (`simulate(x)`, x = `(B,22,256)`)

```
current1 = depthwise( conv1( (x / lam0).unsqueeze(1) ) )      # constant in time -> computed once
n1, n2 = signed IF  (neg_rate = min(1, 1/lam1), min(1, 1/lam2));   n3 = unsigned IF
for t in 1..n_steps (256):
    s1 = n1.step(current1)                                     # (B, 8, 1, 256) ternary spikes
    s2 = n2.step( sep_point( sep_depth( pool1(s1) ) ) )        # (B, 8, 1, 64)
    n3.step( fc( flatten( pool2(s2) ) ) )                      # (B, 4)
return n3.net (spike counts), n3.v (final potentials)
```

- All three layers update **in the same loop iteration**: layer 2 sees layer 1's spikes of the same step, so there is no per-layer delay.
- Pooling and flatten operate directly on spikes; biases of `sep_point` and `fc` are added as constant current at every step.
- Neurons per trial: 8×256 = 2048 (n1), 8×64 = 512 (n2), 4 (n3).

#### Prediction (`predict`)

Highest output spike count. If `tie_break_membrane`, only classes tied at the maximum count are kept and the one with the highest final membrane potential wins. `benchmark.run_snn_benchmark` runs the test set in batches of 96 and saves the spike counts as `scores`.

#### Diagram

```
 x (B,22,256)  analog, microvolts
   │  ÷ lam0       (cancelled by conv1's lam0/lam1 scaling)
   ▼
 ┌─ constant current, computed once ──────────────────────────────────────────────┐
 │ conv1'  1→4, (1,32), same, +bias         (weights x lam0/lam1, bias / lam1)    │
 │ depthwise' 4→8, (22,1), groups=4, +bias  (weights unchanged,  bias / lam1)     │
 └───────────────────────────────────────────┬────────────────────────────────────┘
                                             │ current1 (B,8,1,256), same value every step
 ═══════════ repeated for t = 1 … 256 ════════╪══════════════════════════════════════
                                             ▼
                              ┌──────────────────────────┐
                              │ n1: signed IF            │  2048 neurons / trial
                              │ spikes ∈ {-1, 0, +1}     │  net-rate floor -1/lam1
                              └────────────┬─────────────┘
                                           │ s1 (B,8,1,256)
                                    AvgPool (1,4)
                                           │ (B,8,1,64)
                                   sep_depth'  8→8, (1,16), same, no bias (x lam1/lam2)
                                   sep_point'  8→8, 1x1, +bias (bias / lam2)
                                           ▼
                              ┌──────────────────────────┐
                              │ n2: signed IF            │  512 neurons / trial
                              └────────────┬─────────────┘
                                           │ s2 (B,8,1,64)
                                    AvgPool (1,8) → flatten (64)
                                   fc'  64→4, +bias  (weights x lam2/lam3, bias / lam3)
                                           ▼
                              ┌──────────────────────────┐
                              │ n3: IF, no floor         │  4 neurons (one per class)
                              └────────────┬─────────────┘
 ═══════════════════════════════════════════╪══════════════════════════════════════
                                            ▼
              counts = Σ_t spikes (B,4)     v = final membrane (B,4)
              prediction = argmax(counts); ties → argmax(v among tied)
```

Other facts: dropout and BatchNorm do not exist in the converted network; the first layer takes **analog** input (direct encoding), not spikes; the EEG time axis is handled by the convolution kernels exactly as in the ANN, while the 256 simulation steps are a separate axis.

**[IMAGE PLACEHOLDER: `fig_converted_snn_architecture.png`]**

> **Prompt:** "Technical diagram, white background, flat vector style, top-to-bottom flow. Title: 'Converted feedforward SNN (ANN→SNN, rate coding)'. Top: a box 'EEG trial 22 × 256 (analog, constant input current, divided by λ0)'. Below it a shaded bracket 'computed once' containing two boxes: 'Conv (1,32) + bias [folded BN]' and 'Depthwise Conv (22,1) + bias [folded BN]'. Then a large dashed rectangle labelled 'repeated for 256 time steps' containing, in order: a neuron layer drawn as a grid of small circles labelled 'Layer 1: signed integrate-and-fire, 8×256 neurons, spikes −1/0/+1', an 'AvgPool (1,4)' box, 'Depthwise (1,16) + Pointwise 1×1 + bias' box, 'Layer 2: signed IF, 8×64 neurons', 'AvgPool (1,8) + Flatten (64)', 'Dense 64→4 + bias', and 'Output layer: IF, 4 neurons, no negative spikes'. Draw arrows labelled 'spikes' between layers. Below the dashed rectangle: 'Spike counts per class → argmax (ties broken by final membrane potential)'. Side annotation: 'No leak, reset by subtraction, threshold = 1, λ-normalized weights'. Footer: '776 frozen parameters (BatchNorm folded into convolutions)'."

---

### 6.3 RSNN (`models.RecurrentLIFNet`)

**How it is implemented:** a custom `nn.Module`. Weights are ordinary `nn.Linear` layers; the LIF dynamics are a hand-written Python loop over time. The **only** snnTorch component is `snntorch.surrogate.fast_sigmoid(slope=25)`, used as the spike function (Heaviside forward, fast-sigmoid derivative backward). Training is surrogate-gradient backpropagation through time (autograd through the loop).

#### Components

| Attribute | Layer | Shape | Params (H = 17) |
|---|---|---|---|
| `input_proj` | `Linear(22 → H)` with bias | `(H,22)` + `(H)` | 22·17 + 17 = **391** |
| `recurrent` | `Linear(H → H, bias=False)`, dense, diagonal included | `(H,H)` | 17·17 = **289** |
| `readout` | `Linear(H → 4)` with bias | `(4,H)` + `(4)` | 17·4 + 4 = **72** |
| `dropout` | `Dropout` on the readout input (0.25 / 0.5 by schedule) | | 0 |
| `spike_fn` | snnTorch fast-sigmoid surrogate, slope 25 | | 0 |
| | | **Total** | **752** |

`recurrent.weight[i, j]` is the weight from hidden neuron *j* to hidden neuron *i*. It is initialized `N(0, (1/√H)²)`; `input_proj` and `readout` keep PyTorch's default initialization.

**Hidden size selection:** `H² + 27H + 4` is the parameter count, and `choose_hidden_size` picks the integer `H` closest to the converted SNN's 776. H = 17 → 752 (−24, −3.09%); H = 18 → 814 (+38). So **H = 17**. The ANN has 796.

#### Dynamics (`forward`)

Input `(B,22,256)` → permuted to `(T=256, B, 22)`; `n_steps = min(cfg.n_steps, 256)`.

```
input_current = input_proj(x)                      # (T, B, H), all steps at once (linear, no state)
mem = 0 ; spk = 0 ; spike_count = 0                # re-created on every forward call: no state leaks
for t in 0 .. n_steps-1:
    current = input_current[t] + recurrent(spk)    # spk = spikes of step t-1
    mem     = beta * mem + current - threshold * spk.detach()
    spk     = Heaviside(mem - threshold)           # backward: fast-sigmoid surrogate
    spike_count += spk
logits = readout( dropout( spike_count / n_steps ) )
```

- Reset is **delayed by one step** (it uses the previous spike), and uses `spk.detach()`, so no gradient flows through the reset. The feedback `recurrent(spk)` is *not* detached, so gradients flow through recurrent connections across time.
- The input is analog (one 22-value sample, µV, per step); there is no spike encoding of the input.
- The readout sees only the **time-averaged firing rate** (resolution 1/256); temporal order is used only inside the recurrent layer.
- No max-norm constraint and no BatchNorm.

#### Diagram: one layer, unrolled over time

```
        x_t (22)                      x_{t+1} (22)
          │                             │
          ▼                             ▼
     W_in (H x 22) + b             W_in (H x 22) + b
          │                             │
          ▼                             ▼
         (+)◄── W_rec · s_{t-1}        (+)◄── W_rec · s_t
          │         ▲                    │         ▲
          ▼         │                    ▼         │
   u_t = 0.9·u_{t-1} + I_t − 1·s_{t-1}   u_{t+1} = 0.9·u_t + I_{t+1} − 1·s_t
          │         │                    │         │
          ▼         │ (1-step delay)     ▼         │
   s_t = H(u_t − 1) ┴──────────────►    s_{t+1} = H(u_{t+1} − 1) ┴───► …
          │                                │
          └──────────── Σ_t s_t / 256 ◄────┘   (accumulated over all 256 steps)
                              │
                       Dropout (0.25 / 0.5)
                              │
                      W_out (4 x H) + b
                              │
                         logits (4)
```

#### Connectivity (H = 17)

```
 Input layer (22 channels)           Hidden LIF layer (17 neurons)            Readout
 ┌───┐                              ┌────────────────────────────┐
 │ c1│──┐                           │  n1 ◄──► n2 ◄──► n3 ...    │
 │ c2│──┤  full 22 x 17 + 17 bias   │   ▲╲     ╱▲     ╲╱         │  full 17 x 4 + 4 bias
 │...│──┼──────────────────────────►│   │ ╲   ╱ │      ╳         │─────────────────────► 4 logits
 │c22│──┘   (374 w + 17 b)         │   ▼  ╲ ╱  ▼     ╱ ╲         │   (68 w + 4 b) on mean rate
 └───┘                              │  n17  ...  every neuron → every neuron, self included
                                    └────────────────────────────┘
                                     W_rec: 17 x 17 = 289 weights (17 of them self-connections)
```

Every hidden neuron receives all 22 inputs, all 17 hidden spikes of the previous step (including its own), and projects to all 4 readout units.

**[IMAGE PLACEHOLDER: `fig_rsnn_architecture.png`]**

> **Prompt:** "Technical neural-network diagram, white background, flat vector style, left to right. Title: 'Recurrent LIF network (RSNN) as implemented, H = 17'. Left: a column of 22 small input nodes labelled 'EEG channels at sample t (µV)'. Middle: a circle of 17 hidden neurons drawn as a ring with every neuron connected to every other neuron by thin bidirectional arrows, plus a small self-loop on each neuron; caption 'Dense recurrent weights W_rec 17×17 (289, trained, includes self-connections)'. Dense arrows from all 22 inputs to all 17 hidden neurons labelled 'W_in 22→17 + bias (391)'. Right: 4 output nodes labelled 'left hand, right hand, feet, tongue', with dense arrows from the hidden ring labelled 'W_out 17→4 + bias (72) on mean spike rate'. A box between hidden ring and outputs: 'time-average of spikes over 256 steps → dropout'. Below the ring, an equation strip: 'u_t = 0.9·u_{t−1} + I_t − θ·s_{t−1};  s_t = Heaviside(u_t − θ), θ = 1;  I_t = W_in x_t + W_rec s_{t−1}'. Add a small second panel showing the network unrolled over three time steps t−1, t, t+1 with the recurrent arrows crossing between time steps. Footer: '752 trainable parameters; snnTorch fast-sigmoid surrogate gradient (slope 25)'."

---

### 6.4 Input to each spiking model (compared)

```
 Converted SNN                                       RSNN
 ─────────────                                       ────
 whole trial 22x256 injected as ONE static           sample x_t (22 values) injected at step t
 image into conv layers; same current every step    so the 256 steps ARE the 256 EEG samples
 time inside the EEG = conv kernel (32, 16 samples) time inside the EEG = membrane + recurrence
 simulation steps (256) only improve rate estimates  simulation steps carry the signal's dynamics
```

---

## 7. Benchmark driver, metrics and outputs

### 7.1 Per-model runners (`benchmark.py`)

All runners loop over the selected folds, skip a fold if `metrics/<model>/s<S>_r<R>.json` exists, and build the fold data with `build_fold_data`.

| Runner | What it does | Extra fields saved |
|---|---|---|
| `run_ann_benchmark` | `torch.manual_seed(seed)`; `EEGNet(cfg)`; `pretrain_then_finetune(ANN_TRAIN)`; saves `{"pretrained","finetuned"}` checkpoint and history; predicts test logits | `n_params`, `pretrain_best_epoch`, `finetune_best_epoch` |
| `run_snn_benchmark` | Loads the `"finetuned"` ANN state (raises if missing), builds `ConvertedEEGNetSNN(ann, ft_train.X, SNN_CFG)`, predicts test set in batches of 96 | `n_params` (776), `n_steps`, `lambdas` |
| `run_rsnn_benchmark` | `H` from `model_summary`; `RecurrentLIFNet`; `pretrain_then_finetune(RSNN_TRAIN)`; saves checkpoint and history | `n_params`, `n_hidden`, best epochs |

`benchmark.py` creates its own instances `EEGNET_CFG, SNN_CFG, RSNN_CFG, FOLD_CFG = EEGNetConfig(), SNNConfig(), RSNNConfig(), FoldConfig()` (see F3).

### 7.2 Metrics (`compute_metrics`, one function for all models)

`accuracy_score`, `cohen_kappa_score(labels=[0..3])`, `f1_score(average="macro", zero_division=0)`, and `confusion_matrix` (rows = true, columns = predicted), all from scikit-learn.

### 7.3 Run directory layout

```
results/benchmark/<run id>/
├── config.json                 run id, all config dataclasses, parameter counts
├── folds.json                  fold config + all 90 folds
├── checkpoints/ann|rsnn/s<S>_r<R>.pt
├── histories/ann|rsnn/s<S>_r<R>.json          per-epoch train/val loss, best_epoch
├── metrics/ann|snn|rsnn/s<S>_r<R>.json        accuracy, kappa, macro_f1, confusion, extras
├── predictions/ann|snn|rsnn/s<S>_r<R>.npz     y_true, y_pred, scores (logits or spike counts)
├── results.csv                 one row per model x fold (nothing averaged)
├── predictions.csv             one row per model x fold x trial
├── confusion_matrices.csv      model, subject, repeat, true_class, pred_class, count
└── figures/                    from benchmark_plots
```

`_save_result` writes the `.npz` first and the metrics JSON last, so the JSON marks a finished fold.

### 7.4 Plots (`benchmark_plots.py`)

Descriptive only, no statistics. Subject is treated as the independent unit.

| Figure | Content |
|---|---|
| `accuracy_per_subject.png` | Per subject: faint dots = the 10 repeats, diamond = their mean, per model; dashed line at 25% |
| `metric_distributions.png` | Per metric (accuracy, kappa, macro-F1): box of the 9 subject means per model, faint dots = individual folds |
| `confusion_matrices.png` | Row-normalized confusion matrix summed over all subjects/repeats, one panel per model |

Each figure gets the same "How to read / Takeaways" caption mechanism as `data_statistics.py`. Models plotted are only those present in `results.csv`.

---

## 8. Differences between the code and the research plan

Legend: **MATCH** = code does what the plan says; **DIFFERS** = code does something else; **ADDED** = in code, not in plan; **NOT YET** = in plan, not in code.

| # | Plan statement | Code | Status |
|---|---|---|---|
| 1 | Dataset BNCI2014_001 via MOABB, 9 subjects, 2 sessions, 288 trials/session | Same | MATCH |
| 2 | 22 EEG channels, EOG excluded | Same (asserts 22) | MATCH |
| 3 | "Resampled to 128 Hz, band-pass filtered 4–40 Hz" | Band-pass **first** (on 250 Hz data), then resample | DIFFERS (order only) |
| 4 | Epoch 0.5–2.5 s **after cue onset**, 22 × 256 | Epoch 0.5–2.5 s after the **annotation onset** with `CUE_OFFSET_S = 0.0`; whether annotation onset = cue is unverified | DIFFERS / see F1 |
| 5 | Artifact trials retained | `reject=None` | MATCH |
| 6 | Inputs in the networks | ×1e6 volts → µV scaling | ADDED |
| 7 | ANN: EEGNet-4,2, ELU, kernel 32, 796 params | Same, 796 verified | MATCH |
| 8 | Adam default params, categorical CE, 500 epochs, lowest-val-loss weights, dropout 0.25 / 0.5 | Same | MATCH |
| 9 | Batch size not specified | ANN 16, RSNN 64 | ADDED (and differs between models) |
| 10 | Pretrain on 5 other subjects' session 1, validate on 3, 10 seeded draws → 90 folds | Same | MATCH |
| 11 | Fine-tune on target session 1 (80/20 stratified, 100 epochs), test on session 2 (288) | Same | MATCH |
| 12 | All models share folds; SNN converted from the fine-tuned ANN | Same | MATCH |
| 13 | SNN: carry over weights, activations approximated by spike rates | Same, plus BN folding, data-based 99.9th-percentile normalization, signed spikes for ELU | MATCH + ADDED details |
| 14 | SNN prediction = output neuron with highest spike count | Same, plus ties broken by final membrane potential | ADDED |
| 15 | Simulation steps "not yet decided" | 256 | decided |
| 16 | RSNN: input projection, one all-to-all LIF recurrent layer, linear 4-class readout, surrogate-gradient BPTT | Same; recurrent matrix also includes self-connections; readout works on time-averaged spike rate with dropout | MATCH (+ details) |
| 17 | RSNN uses the ANN's optimizer, loss, epochs, checkpoint rule | Same, but batch size 64 (ANN 16) and dropout is applied to the readout input | MATCH, with differences |
| 18 | "Neuron hyperparameters are copied from the converted SNN" | Only `threshold = 1.0` is copied. RSNN has leak β = 0.9, the SNN's IF neurons have no leak and the SNN threshold is relative to λ-normalized weights | DIFFERS / see F7 |
| 19 | Neuron count set so trainable parameters ≈ converted SNN's | H = 17 → 752 vs. 776 (−3.1%) | MATCH |
| 20 | "Both spiking models receive the same input representation" | SNN: whole trial as a static current repeated for 256 steps. RSNN: one EEG sample per step | DIFFERS / see F5 |
| 21 | Accuracy, Cohen's kappa, macro-F1, per-class confusion matrices | Implemented | MATCH |
| 22 | Per-subject mean over the 10 repeats; subject as independent unit | Only in plots (`benchmark_plots`); no tables | partial |
| 23 | Significance threshold from test-trial count + 10,000-permutation label test | Not in code | NOT YET |
| 24 | Exact two-sided Wilcoxon, Holm correction, bootstrap 95% CI of median paired difference | Not in code (`main.py` ends with a comment only) | NOT YET |
| 25 | Seeds and configurations logged | `config.json`, `folds.json`, per-fold seed in metrics | MATCH |
| 26 | Pipeline leakage check, ANN sanity check | Not in code | NOT YET |
| 27 | "PyTorch and snnTorch for the models" | snnTorch used only for the surrogate-gradient function | MATCH (narrowly) |
| 28 | SciPy for metrics and statistics | SciPy used only for Welch PSD in `data_statistics.py` | NOT YET for statistics |
| 29 | ARC cluster computation | Fold sharding via `BENCH_TARGETS`, `BENCH_REPEATS`, `BENCH_RUN_ID` | ADDED |
| 30 | Plan figures (Fig. 2) came from MOABB's default 2–6 s window, "for exploration only" | `data_statistics.py` uses the *preprocessed* 0.5–2.5 s epochs, so its topomaps/PSDs are not the plan's figures | DIFFERS |
| 31 | Section 2 says temporal context in converted SNNs rests on membrane potentials | In the converted EEGNet the EEG time axis is handled by the convolution kernels (32 and 16 samples); the 256 steps are not EEG time | conceptual difference |

---

## 9. Flagged issues (location and what looks wrong)

Severity is my judgement. "Unverified" means I could not confirm it without running the code or inspecting the data.

### High

**F1. `preprocess_data.py`, `CUE_OFFSET_S = 0.0` (epoch alignment).**
The epoch is placed 0.5–2.5 s after the *annotation onset*. The plan says MOABB's default window for this dataset is 2–6 s (1,001 samples at 250 Hz) and that the arrow cue appears at t = 2 s; MOABB's `BNCI2014_001` interval `[2, 6]` implies its event markers sit at trial start (t = 0), not at the cue. If so, the current epoch covers trial time 0.5–2.5 s: mostly the fixation-cross period, with only the last 0.5 s after the cue and none of the imagery period. Unverified: it depends on where MOABB places the annotations in the raw objects. The `data_statistics.py` topomap caption itself says an absent pattern would suggest misaligned epochs, and the plan's topomaps were made from MOABB's default 2–6 s window, not from this pipeline's epochs.

**F2. `benchmark_config.py`, `QUICK_MODE: bool = True`.**
Quick mode is switched on in the uploaded file. Every run with this file uses 2 pretraining epochs and 2 fine-tuning epochs for the ANN and RSNN instead of 500/100.

**F3. `benchmark.py`, line `EEGNET_CFG, SNN_CFG, RSNN_CFG, FOLD_CFG = EEGNetConfig(), SNNConfig(), RSNNConfig(), FoldConfig()`.**
This creates fresh default configs and shadows the module-level instances in `benchmark_config.py`. The quick-mode overrides of `SNN_CFG` (`n_steps=32`) and `RSNN_CFG` (`n_steps=64`) therefore never reach the benchmark. Only `ANN_TRAIN` and `RSNN_TRAIN` (imported by name) carry the quick values. In a quick run the epochs are 2, but SNN and RSNN still use 256 steps, and `config.json` records 256 steps next to 2 epochs. The module-level `FOLD_CFG`, `EEGNET_CFG`, `SNN_CFG`, `RSNN_CFG` in `benchmark_config.py` are otherwise unused.

**F4. `benchmark_config.py` comment vs. `benchmark.py` / `main.py`.**
The comment says quick mode gives the run directory a `quick_` prefix and that `main.py` defaults to one fold (subject 1, repeat 0). Neither exists: `prepare_run_dir` only uses `$BENCH_RUN_ID` or a timestamp, and `main.py` runs all 90 folds unless the env vars are set. A quick run is indistinguishable from a full run by directory name.

**F5. `ann_to_snn.py` vs. `models.py`, input representation.**
The plan requires both spiking models to receive the same input representation. The converted SNN receives the whole 22×256 trial as a static analog current repeated at all 256 steps; the RSNN receives one EEG sample per step. They also use their time axes in different ways (Section 6.4).

### Medium

**F6. `models.RecurrentLIFNet`, `input_proj` receives unnormalized µV values.**
Inputs are in microvolts (`INPUT_SCALE = 1e6`) and go straight into a default-initialized `Linear(22, H)`, against a threshold of 1.0. By reasoning only (not measured): input currents are likely several times the threshold, so hidden neurons would fire almost every step or almost never, depending on sign. The ANN has BatchNorm after the first convolutions; the RSNN has no normalization.

**F7. `benchmark_config.RSNNConfig` vs. plan: "neuron hyperparameters are copied from the converted SNN".**
Only the numeric threshold (1.0) is shared. The RSNN has β = 0.9 leak, a delayed reset, and a Heaviside spike function; the converted SNN has no leak, a same-step reset, and a threshold that is relative to λ-normalized weights. The config comment itself notes β = 1 for the SNN.

**F8. `benchmark.prepare_run_dir` / `_done`: resume and sharding.**
When `$BENCH_RUN_ID` points to an existing directory, `config.json` and `folds.json` are overwritten with the current settings, and `_done` only checks whether a metrics file exists, not whether the existing results used the same configuration. Toggling quick mode (or changing any hyperparameter) between sessions can mix results from different configurations under one `config.json`. Sharded jobs all rewrite the same two files.

**F9. `main.py`, `data_statistics(data, processed)` runs unconditionally.**
It re-runs on every invocation (including each sharded job), processes all subjects including the test sessions (descriptive only), and ends `_class_distribution` with a hard `assert (counts == 72).all()` that would abort the whole benchmark if it ever failed. `main.py` also has no `if __name__ == "__main__":` guard.

### Low / observations

**F10. `ann_to_snn.py`, `lam0` cancels out.**
`conv1.weight` is multiplied by `lam0/lam1` and the input is divided by `lam0`, so `lam0` has no net effect. The module docstring says firing rates stay in [0, 1]; the input is injected as an unbounded analog current and is not a rate.

**F11. `ann_to_snn._IFNeurons`, ELU negative branch.**
Negative activations are represented as `max(z, −1)` rather than `exp(z) − 1` (documented in the module docstring). Additionally, the `allowed` rule limits the *net* spike count, so with `neg_rate = 1/λ < 1` a neuron with net count 0 cannot emit its first negative spike before about `λ` steps. Effect on accuracy not assessed.

**F12. `data_statistics._log_band_power`, band edges overlap.**
`mu = (8, 13)` and `beta = (13, 30)` both use inclusive comparisons (`f >= lo & f <= hi`), so the 13 Hz bin is counted in both bands.

**F13. `folds.generate_folds`, fine-tune split is random per trial.**
The 80/20 split of the target's session 1 is stratified but not blocked by run, so validation trials sit between training trials from the same runs. The validation loss used for checkpoint selection can be optimistic. The test session is separate, so this does not leak into test results.

**F14. Static text in `NOTES` (`data_statistics.py`, `benchmark_plots.py`).**
Captions hard-code numbers (288 trials, 72 per class, 9 subjects, 10 repeats) and pre-written "should look like" interpretations that do not depend on the data actually plotted. They would stay unchanged for a sharded or partial run.

**F15. `models.EEGNetConfig`, `F2 = 8`.**
`F2` is typed in separately instead of `F1 * D`; `EEGNet` also hard-codes `nn.Dropout(0.25)` which is always overwritten by `set_dropout`.

**F16. `models.RecurrentLIFNet`, dropout on a 17-dimensional rate vector.**
Fine-tuning dropout is 0.5 on the readout input, which here has only 17 features. No gradient clipping is applied for BPTT over 256 steps. Both are observations, not confirmed problems.

---

## Appendix A. Parameter arithmetic

```
EEGNet:   128 + 8 + 176 + 16 + 128 + 64 + 16 + 260                         = 796
SNN:      132 + 184 + 128 + 72 + 260                                       = 776
RSNN(H):  (22H + H) + H² + (4H + 4) = H² + 27H + 4
          H = 16 → 692,  H = 17 → 752,  H = 18 → 814   → H = 17 (closest to 776)
```

## Appendix B. Seeds

| Use | Seed |
|---|---|
| Fold (subject draw, fine-tune split, model init `torch.manual_seed`) | `2026 + 100·target + repeat` |
| Pretraining shuffle + dropout (`fit`) | same value |
| Fine-tuning shuffle + dropout (`fit`) | that value + 1 |
