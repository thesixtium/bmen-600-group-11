"""All hyperparameters, seeds and constants of the benchmark in one place.

Every value here is written to <run_dir>/config.json so a run can be reproduced.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

RESULTS_DIR: Path = Path(__file__).resolve().parent.parent / "results" / "benchmark"

N_CHANNELS: int = 22
N_SAMPLES: int = 256
N_CLASSES: int = 4
TRAIN_SESSION: str = "0train"  # session 1: pretraining subjects + target fine-tuning
TEST_SESSION: str = "1test"  # session 2: held out, only used for final prediction

# preprocess_data() returns MNE data in volts (about 1e-5). data_statistics() multiplies by 1e6 before
# using it. This is a fixed unit conversion (volts -> microvolts), NOT a data-dependent normalization,
# so it cannot leak information. Without it the networks would see values ~1e-5 (see README notes).
INPUT_SCALE: float = 1e6


@dataclass(frozen=True)
class FoldConfig:
    n_repeats: int = 10
    n_pretrain_subjects: int = 5  # of the 8 non-target subjects; the other 3 are the pretraining validation set
    finetune_val_fraction: float = 0.2
    base_seed: int = 2026  # seed of a fold = base_seed + 100 * target_subject + repeat


@dataclass(frozen=True)
class EEGNetConfig:
    """EEGNet-4,2 (Lawhern et al. 2018, Table 2)."""

    n_channels: int = N_CHANNELS
    n_samples: int = N_SAMPLES
    n_classes: int = N_CLASSES
    F1: int = 4
    D: int = 2
    F2: int = 8  # = F1 * D
    kernel_length: int = 32
    separable_kernel: int = 16
    pool1: int = 4
    pool2: int = 8
    depthwise_max_norm: float = 1.0
    dense_max_norm: float = 0.25
    bn_eps: float = 1e-3  # Keras defaults
    bn_momentum: float = 0.01  # PyTorch momentum equivalent of Keras momentum=0.99


@dataclass(frozen=True)
class TrainConfig:
    pretrain_epochs: int = 500
    finetune_epochs: int = 100
    pretrain_dropout: float = 0.25
    finetune_dropout: float = 0.5
    learning_rate: float = 1e-3  # Adam default
    batch_size: int = 16  # EEGNet paper value; not specified in the research plan
    log_every_epochs: int = 50  # progress line (elapsed + ETA) every N epochs inside a training phase; 0 = off


@dataclass(frozen=True)
class SNNConfig:
    """ANN -> SNN conversion (Rueckauer et al. 2017 style, see ann_to_snn.py)."""

    n_steps: int = 256  # simulation time steps (unresolved in the plan)
    threshold: float = 1.0  # firing threshold after weight/threshold normalization
    calibration_percentile: float = 99.9  # data-based normalization percentile
    signed_spikes: bool = True  # +/-1 spikes in hidden layers so the negative ELU branch can be represented
    negative_floor: float = 1.0  # membrane potential floor of signed neurons, in thresholds
    tie_break_membrane: bool = True  # break output spike-count ties with the final membrane potential


@dataclass(frozen=True)
class RSNNConfig:
    n_hidden: int | None = None  # None -> chosen to match the converted SNN's parameter count
    n_steps: int = N_SAMPLES  # one simulation step per EEG sample (128 Hz)
    beta: float = 0.9  # membrane decay per step (fixed). The converted SNN's IF neurons have no leak (beta=1)
    threshold: float = 1.0  # copied from the converted SNN
    surrogate_slope: float = 25.0  # snntorch.surrogate.fast_sigmoid default
    recurrent_init_gain: float = 1.0  # W_rec ~ N(0, (gain / sqrt(n_hidden))^2)
    batch_size: int = 64  # larger than the ANN's to keep BPTT over 256 steps tractable
    # Per-channel standardization of the input (mean/std over trials and time of the PRETRAINING subjects' session 1
    # only; fixed afterwards, so no target or test information is used). The ANN has BatchNorm and the converted SNN
    # divides by lambda_0; this gives the RSNN the equivalent scale handling. False = previous behaviour (raw uV).
    normalize_input: bool = True


@dataclass(frozen=True)
class ComparisonConfig:
    """Statistics of the 'Comparative metrics' step (research plan, Section 4: 'Avoiding misleading results')."""

    n_permutations: int = 10_000  # label permutations of the chance test (per subject x model x metric)
    n_bootstrap: int = 10_000  # bootstrap resamples (over subjects) of the median paired difference
    alpha: float = 0.05  # significance level of the chance tests and of the Holm-corrected Wilcoxon tests
    ci_level: float = 0.95
    seed: int = 2026


FOLD_CFG = FoldConfig()
EEGNET_CFG = EEGNetConfig()
SNN_CFG = SNNConfig()
RSNN_CFG = RSNNConfig()
ANN_TRAIN = TrainConfig()
RSNN_TRAIN = TrainConfig(batch_size=RSNN_CFG.batch_size)
COMPARISON_CFG = ComparisonConfig()

# Bug-testing mode: BENCH_QUICK=1 shrinks epochs and simulation steps. Results are NOT comparable to the real
# protocol; the run directory gets a "quick_" prefix and main.py defaults to one fold (subject 1, repeat 0).
QUICK_MODE: bool = False  # <- set to True here for a lightweight bug-testing run
QUICK: bool = QUICK_MODE or os.environ.get("BENCH_QUICK") == "1"  # the environment variable also works
if QUICK:
    ANN_TRAIN = replace(ANN_TRAIN, pretrain_epochs=2, finetune_epochs=2)
    RSNN_TRAIN = replace(RSNN_TRAIN, pretrain_epochs=2, finetune_epochs=2)
    SNN_CFG = replace(SNN_CFG, n_steps=32)
    COMPARISON_CFG = replace(COMPARISON_CFG, n_permutations=200, n_bootstrap=200)
    RSNN_CFG = replace(RSNN_CFG, n_steps=64)  # uses only the first 64 of 256 samples