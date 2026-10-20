"""Fold generation and per-fold data assembly.

Protocol (research plan, Section 4): for every target subject, session 1 of 5 of the other 8 subjects is the
pretraining set and session 1 of the remaining 3 is the pretraining validation set. The target's session 1 is split
80/20 (stratified) for fine-tuning/validation. The target's session 2 is the test set and is only used for the final
prediction. 10 seeded repeats per target subject give 9 x 10 = 90 folds, shared by all three models.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from benchmark_config import INPUT_SCALE, TEST_SESSION, TRAIN_SESSION, FoldConfig
from preprocess_data import ProcessedData


@dataclass(frozen=True)
class Fold:
    target: int
    repeat: int
    seed: int
    pretrain_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    finetune_train_idx: tuple[int, ...]  # indices into the target's session-1 trials
    finetune_val_idx: tuple[int, ...]


@dataclass(frozen=True)
class Split:
    X: NDArray[np.float32]  # (n, 22, 256), scaled by INPUT_SCALE
    y: NDArray[np.int64]


@dataclass(frozen=True)
class FoldData:
    pre_train: Split
    pre_val: Split
    ft_train: Split
    ft_val: Split
    test: Split  # target session 2: never used for training, validation or checkpoint selection


def generate_folds(processed: ProcessedData, cfg: FoldConfig = FoldConfig()) -> list[Fold]:
    subjects = sorted(processed)
    folds: list[Fold] = []
    for target in subjects:
        others = [s for s in subjects if s != target]
        assert len(others) == 8 and cfg.n_pretrain_subjects < len(others)
        y_target = processed[target][TRAIN_SESSION].y
        for repeat in range(cfg.n_repeats):
            seed = cfg.base_seed + 100 * target + repeat
            rng = np.random.default_rng(seed)
            chosen = [int(s) for s in rng.permutation(others)]
            pre, val = sorted(chosen[: cfg.n_pretrain_subjects]), sorted(chosen[cfg.n_pretrain_subjects :])
            train_idx, val_idx = train_test_split(
                np.arange(len(y_target)), test_size=cfg.finetune_val_fraction, stratify=y_target, random_state=seed
            )
            folds.append(
                Fold(
                    target=target,
                    repeat=repeat,
                    seed=seed,
                    pretrain_subjects=tuple(pre),
                    val_subjects=tuple(val),
                    finetune_train_idx=tuple(int(i) for i in sorted(train_idx)),
                    finetune_val_idx=tuple(int(i) for i in sorted(val_idx)),
                )
            )
    return folds


def filter_folds(folds: list[Fold], targets: list[int] | None = None, repeats: list[int] | None = None) -> list[Fold]:
    """Select a subset (e.g. to shard the 90 folds over several cluster jobs)."""
    return [f for f in folds if (targets is None or f.target in targets) and (repeats is None or f.repeat in repeats)]


def save_folds(folds: list[Fold], cfg: FoldConfig, path: Path) -> None:
    path.write_text(json.dumps({"config": asdict(cfg), "folds": [asdict(f) for f in folds]}, indent=1))


def _stack(processed: ProcessedData, subjects: list[int], session: str) -> Split:
    X = np.concatenate([processed[s][session].X for s in subjects])
    y = np.concatenate([processed[s][session].y for s in subjects])
    return Split(X=(X * INPUT_SCALE).astype(np.float32), y=y)


def build_fold_data(processed: ProcessedData, fold: Fold) -> FoldData:
    target_train = _stack(processed, [fold.target], TRAIN_SESSION)
    tr, va = np.array(fold.finetune_train_idx), np.array(fold.finetune_val_idx)
    return FoldData(
        pre_train=_stack(processed, list(fold.pretrain_subjects), TRAIN_SESSION),
        pre_val=_stack(processed, list(fold.val_subjects), TRAIN_SESSION),
        ft_train=Split(target_train.X[tr], target_train.y[tr]),
        ft_val=Split(target_train.X[va], target_train.y[va]),
        test=_stack(processed, [fold.target], TEST_SESSION),
    )
