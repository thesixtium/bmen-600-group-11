"""Shared helpers for the v1.5 to v1.7 screens: data (cached), fold data per variant, models, accuracy.

The pipeline code is the frozen v1.4 copy (../local-train-v1.4/rssns_for_mi_EEG/src), imported unchanged. The only new
code is in this file: the per-subject normalization (v1.6) and the pooled pretraining split (v1.7) are built here, in
build_variant_data(), so the v1.4 code stays exactly as it ran.

Session-2 rule: build_variant_data() returns session-1 data only. The target's session 2 is never loaded into a model.

Variants
  v1.4  the v1.4 data handling (reference): 5 pretraining + 3 validation subjects, no per-subject normalization
  v1.6  as v1.4, but every subject's channels are z-scored with that subject's OWN session-1 mean and SD (over trials and
        time, labels not used). Applies to both input kinds (22-channel 4-40 Hz for EEGNet, 44 band inputs for the
        recurrent models).
  v1.7  as v1.4, but the pretraining pool is session 1 of all 8 non-target subjects, split 80 / 20 stratified by
        subject and class (seed = fold seed), instead of 5 training and 3 validation subjects.
"""
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]  # experiments/: this file sits in experiments/<version>/code/
SRC = EXP / "v1.4" / "code" / "src"  # the screens use the v1.4 pipeline code unchanged
sys.path.insert(0, str(SRC))

from benchmark_config import (  # noqa: E402
    ANN_TRAIN, EEGNET_CFG, INPUT_SCALE, N_CLASSES, N_RECURRENT_INPUTS, RNN_CFG, RNN_TRAIN, RSNN_CFG, RSNN_TRAIN,
    TRAIN_SESSION,
)
from download_data import download_data  # noqa: E402
from folds import Fold, Split, generate_folds  # noqa: E402
from models import EEGNet, RecurrentELUNet, RecurrentLIFNet, model_summary  # noqa: E402
from preprocess_data import preprocess_data  # noqa: E402
from training import predict_logits  # noqa: E402

CACHE = HERE / "cache" / "processed_v14.pkl"
V1_RUN = EXP / "v1" / "results" / "20261029_141407"     # EEGNet checkpoints (v1.4 copied EEGNet from v1)
V14_RUN = EXP / "v1.4" / "results" / "20261103_175004"  # RSNN and RNN checkpoints
SAVED_RUN = {"ann": V1_RUN, "rsnn": V14_RUN, "rnn": V14_RUN}
CHANCE_288 = 0.2917  # binomial threshold, n = 288, p = 0.25, alpha 0.05 (pipeline value)


def results_dir(name: str) -> Path:
    """Result folder of one screen: experiments/<name>/results (name = v1.5, v1.6 or v1.7)."""
    return EXP / name / "results"


DEV_FOLDS = [(t, 0) for t in range(1, 10)]  # development folds: all 9 targets, repeat 0 (as in the earlier screens)

# model name -> (input field, train config); the ANN uses the 22-channel input, the recurrent twins the 44 band inputs
MODELS = {"ann": ("X", ANN_TRAIN), "rsnn": ("X_bands", RSNN_TRAIN), "rnn": ("X_bands", RNN_TRAIN)}


def load_processed():
    """Preprocessed BCI IV 2a exactly as the v1.4 main.py makes it (with X_bands); cached after the first call.
    Build the cache once (python screen_common.py) before starting parallel jobs."""
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    processed = preprocess_data(download_data(redo=False))
    CACHE.parent.mkdir(exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(processed))
    tmp.replace(CACHE)
    return processed


def get_fold(processed, target: int, repeat: int) -> Fold:
    return next(f for f in generate_folds(processed) if f.target == target and f.repeat == repeat)


@dataclass(frozen=True)
class ScreenData:
    pre_train: Split
    pre_val: Split
    ft_train: Split
    ft_val: Split
    target_s1: Split  # all 288 session-1 trials of the target: never used in pretraining, so a clean zero-shot test


def _subject_s1(processed, subject: int, field: str, per_subject_norm: bool) -> Split:
    """One subject's session 1, scaled to microvolts as the pipeline does; optionally z-scored per channel with this
    subject's own session-1 statistics (no labels used)."""
    X = (getattr(processed[subject][TRAIN_SESSION], field) * INPUT_SCALE).astype(np.float32)
    if per_subject_norm:
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
        X = ((X - mean) / std).astype(np.float32)
    return Split(X=X, y=processed[subject][TRAIN_SESSION].y)


def _cat(splits: list[Split]) -> Split:
    return Split(X=np.concatenate([s.X for s in splits]), y=np.concatenate([s.y for s in splits]))


def build_variant_data(processed, fold: Fold, field: str, variant: str) -> ScreenData:
    """Session-1 fold data for one variant. With variant 'v1.4' this equals folds.build_fold_data minus the test split
    (same subjects, same order, same indices), which check_reproduction in screen.py confirms through the weights."""
    if variant not in ("v1.4", "v1.6", "v1.7"):
        raise ValueError(variant)
    norm = variant == "v1.6"
    target = _subject_s1(processed, fold.target, field, norm)
    tr, va = np.array(fold.finetune_train_idx), np.array(fold.finetune_val_idx)
    if variant == "v1.7":
        others = sorted(set(fold.pretrain_subjects) | set(fold.val_subjects))  # all 8 non-target subjects
        pool = [_subject_s1(processed, s, field, False) for s in others]
        X = np.concatenate([p.X for p in pool])
        y = np.concatenate([p.y for p in pool])
        strata = np.concatenate([np.full(len(p.y), i) for i, p in enumerate(pool)]) * N_CLASSES + y
        i_tr, i_va = train_test_split(np.arange(len(y)), test_size=0.2, stratify=strata, random_state=fold.seed)
        i_tr, i_va = np.sort(i_tr), np.sort(i_va)
        pre_train, pre_val = Split(X[i_tr], y[i_tr]), Split(X[i_va], y[i_va])
    else:
        pre_train = _cat([_subject_s1(processed, s, field, norm) for s in fold.pretrain_subjects])
        pre_val = _cat([_subject_s1(processed, s, field, norm) for s in fold.val_subjects])
    return ScreenData(pre_train, pre_val, Split(target.X[tr], target.y[tr]), Split(target.X[va], target.y[va]), target)


def make_model(name: str, seed: int):
    """Same construction as benchmark.py: seed first, then the model (the RSNN and RNN share the hidden size)."""
    torch.manual_seed(seed)
    if name == "ann":
        return EEGNet(EEGNET_CFG)
    n_hidden = model_summary(EEGNET_CFG, RSNN_CFG)["rsnn_hidden"]
    cls, cfg = (RecurrentLIFNet, RSNN_CFG) if name == "rsnn" else (RecurrentELUNet, RNN_CFG)
    return cls(N_RECURRENT_INPUTS, n_hidden, N_CLASSES, cfg)


def uses_input_stats(name: str) -> bool:
    return name != "ann" and (RSNN_CFG if name == "rsnn" else RNN_CFG).normalize_input


def accuracy(model, split: Split, device) -> float:
    return float((predict_logits(model, split.X, device).argmax(1) == split.y).mean())


if __name__ == "__main__":
    p = load_processed()
    s = p[1][TRAIN_SESSION]
    print(f"cache ready: {CACHE} | subjects {sorted(p)} | S1 session 1 X {s.X.shape} X_bands {s.X_bands.shape}")
