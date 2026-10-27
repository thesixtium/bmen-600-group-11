"""Benchmark driver: trains/evaluates the three models on the shared folds and saves per-fold results.

Output layout (<run_dir> = results/benchmark/<run id>, run id = $BENCH_RUN_ID or a timestamp):
  config.json, folds.json            configuration, parameter counts, fold definitions and seeds
  checkpoints/<model>/s<S>_r<R>.pt   pretrained + fine-tuned weights (ann, rsnn)
  histories/<model>/s<S>_r<R>.json   per-epoch train/validation loss
  metrics/<model>/s<S>_r<R>.json     accuracy, kappa, macro-F1, confusion matrix, config metadata (one per fold)
  predictions/<model>/s<S>_r<R>.npz  y_true, y_pred, scores (logits / output spike counts)
  results.csv, predictions.csv, confusion_matrices.csv   aggregated by collect_results()
Existing per-fold results are skipped, so an interrupted run can be resumed by re-using the same $BENCH_RUN_ID.
"""

import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from ann_to_snn import ConvertedEEGNetSNN
from benchmark_config import (
    ANN_TRAIN,
    N_CHANNELS,
    N_CLASSES,
    RESULTS_DIR,
    RSNN_TRAIN,
    EEGNetConfig,
    FoldConfig,
    RSNNConfig,
    SNNConfig,
)
from folds import Fold, build_fold_data, save_folds
from models import EEGNet, RecurrentLIFNet, count_parameters, model_summary
from preprocess_data import ProcessedData
from timing import PipelineTimer
from training import TrainingOutcome, predict_logits, pretrain_then_finetune

EEGNET_CFG, SNN_CFG, RSNN_CFG, FOLD_CFG = EEGNetConfig(), SNNConfig(), RSNNConfig(), FoldConfig()
MODELS: tuple[str, ...] = ("ann", "snn", "rsnn")


# ---------------------------------------------------------------- setup


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_run_dir(folds: list[Fold], run_id: str | None = None) -> Path:
    """Create the run directory (never overwrites an existing run unless its id is given) and save config + folds."""
    run_id = run_id or os.environ.get("BENCH_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    for sub in ("checkpoints", "histories", "metrics", "predictions"):
        for model in MODELS:
            (run_dir / sub / model).mkdir(parents=True, exist_ok=True)
    summary = model_summary(EEGNET_CFG, RSNN_CFG)
    config = {
        "run_id": run_id,
        "fold": asdict(FOLD_CFG),
        "eegnet": asdict(EEGNET_CFG),
        "ann_train": asdict(ANN_TRAIN),
        "snn": asdict(SNN_CFG),
        "rsnn": asdict(RSNN_CFG),
        "rsnn_train": asdict(RSNN_TRAIN),
        "parameter_counts": summary,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))
    save_folds(folds, FOLD_CFG, run_dir / "folds.json")
    print(f"Run directory: {run_dir}")
    print(f"Parameters: {summary}")
    return run_dir


# ---------------------------------------------------------------- shared metrics and saving


def compute_metrics(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> dict:
    """The single metric function used for all three models."""
    labels = list(range(N_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),  # rows = true, columns = predicted
    }


def _stem(fold: Fold) -> str:
    return f"s{fold.target}_r{fold.repeat}"


def _done(run_dir: Path, model: str, fold: Fold) -> bool:
    return (run_dir / "metrics" / model / f"{_stem(fold)}.json").exists()


def pending_folds(run_dir: Path, model: str, folds: list[Fold]) -> list[Fold]:
    """Folds of `model` that still have to be run (no metrics file yet)."""
    return [f for f in folds if not _done(run_dir, model, f)]


def _save_result(
    run_dir: Path,
    model: str,
    fold: Fold,
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    scores: NDArray,
    extra: dict,
) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    record = {
        "model": model,
        "subject": fold.target,
        "repeat": fold.repeat,
        "seed": fold.seed,
        "n_test": int(len(y_true)),
        **metrics,
        **extra,
    }
    np.savez_compressed(run_dir / "predictions" / model / f"{_stem(fold)}.npz", y_true=y_true, y_pred=y_pred, scores=scores)
    (run_dir / "metrics" / model / f"{_stem(fold)}.json").write_text(json.dumps(record, indent=1))
    print(f"[{model}] subject {fold.target} repeat {fold.repeat}: acc={metrics['accuracy']:.3f} kappa={metrics['kappa']:.3f}")
    return record


def _save_history(run_dir: Path, model: str, fold: Fold, outcome: TrainingOutcome) -> None:
    history = {"pretrain": outcome.pretrain_history, "finetune": outcome.finetune_history}
    (run_dir / "histories" / model / f"{_stem(fold)}.json").write_text(json.dumps(history))


def _best_epochs(outcome: TrainingOutcome) -> dict:
    return {
        "pretrain_best_epoch": outcome.pretrain_history[0]["best_epoch"],
        "finetune_best_epoch": outcome.finetune_history[0]["best_epoch"],
    }


# ---------------------------------------------------------------- Model 1: EEGNet ANN


def run_ann_benchmark(processed: ProcessedData, folds: list[Fold], run_dir: Path, device: torch.device, timer: PipelineTimer) -> None:
    tracker = timer.track("ann", pending_folds(run_dir, "ann", folds))
    for fold in tracker.folds:
        with tracker.fold(fold) as t:
            data = build_fold_data(processed, fold)
            torch.manual_seed(fold.seed)
            model = EEGNet(EEGNET_CFG)
            outcome = pretrain_then_finetune(
                model, data, ANN_TRAIN, fold.seed, device, tag=f"[ann {_stem(fold)}]"
            )  # model = fine-tuned best weights
            t["pretrain_s"], t["finetune_s"] = outcome.pretrain_seconds, outcome.finetune_seconds
            torch.save(
                {"pretrained": outcome.pretrained_state, "finetuned": outcome.finetuned_state},
                run_dir / "checkpoints" / "ann" / f"{_stem(fold)}.pt",
            )
            _save_history(run_dir, "ann", fold, outcome)
            logits = predict_logits(model, data.test.X, device)
            _save_result(
                run_dir, "ann", fold, data.test.y, logits.argmax(1), logits,
                {"n_params": count_parameters(model), **_best_epochs(outcome)},
            )


# ---------------------------------------------------------------- Model 2: converted SNN


def run_snn_benchmark(processed: ProcessedData, folds: list[Fold], run_dir: Path, device: torch.device, timer: PipelineTimer) -> None:
    """Converts each fold's fine-tuned ANN checkpoint (run_ann_benchmark must have produced it). No training."""
    tracker = timer.track("snn", pending_folds(run_dir, "snn", folds))
    for fold in tracker.folds:
        with tracker.fold(fold) as t:
            ckpt_path = run_dir / "checkpoints" / "ann" / f"{_stem(fold)}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"No fine-tuned ANN checkpoint for {_stem(fold)}; run the ANN benchmark first: {ckpt_path}")
            data = build_fold_data(processed, fold)
            ann = EEGNet(EEGNET_CFG)
            ann.load_state_dict(torch.load(ckpt_path, map_location="cpu")["finetuned"])
            ann.to(device).eval()
            # calibration for the data-based normalization: the target's fine-tuning TRAINING trials only
            t0 = time.perf_counter()
            snn = ConvertedEEGNetSNN(ann, torch.from_numpy(data.ft_train.X).to(device), SNN_CFG)
            t["convert_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            X = torch.from_numpy(data.test.X).to(device)
            preds, counts = [], []
            for i in range(0, len(X), 96):
                p, c = snn.predict(X[i : i + 96])
                preds.append(p.cpu())  # .cpu() synchronizes the GPU, so the timing is accurate
                counts.append(c.cpu())
            y_pred = torch.cat(preds).numpy()
            spike_counts = torch.cat(counts).numpy()
            t["simulate_s"] = time.perf_counter() - t0
            _save_result(
                run_dir, "snn", fold, data.test.y, y_pred, spike_counts,
                {"n_params": snn.parameter_count(), "n_steps": SNN_CFG.n_steps, "lambdas": list(snn.lambdas)},
            )


# ---------------------------------------------------------------- Model 3: RSNN


def run_rsnn_benchmark(processed: ProcessedData, folds: list[Fold], run_dir: Path, device: torch.device, timer: PipelineTimer) -> None:
    n_hidden = model_summary(EEGNET_CFG, RSNN_CFG)["rsnn_hidden"]
    tracker = timer.track("rsnn", pending_folds(run_dir, "rsnn", folds))
    for fold in tracker.folds:
        with tracker.fold(fold) as t:
            data = build_fold_data(processed, fold)
            torch.manual_seed(fold.seed)
            model = RecurrentLIFNet(N_CHANNELS, n_hidden, N_CLASSES, RSNN_CFG)
            if RSNN_CFG.normalize_input:
                # per-channel standardization from the PRETRAINING subjects' session 1 only (no target / test data)
                model.set_input_stats(data.pre_train.X)
            outcome = pretrain_then_finetune(model, data, RSNN_TRAIN, fold.seed, device, tag=f"[rsnn {_stem(fold)}]")
            t["pretrain_s"], t["finetune_s"] = outcome.pretrain_seconds, outcome.finetune_seconds
            torch.save(
                {"pretrained": outcome.pretrained_state, "finetuned": outcome.finetuned_state},
                run_dir / "checkpoints" / "rsnn" / f"{_stem(fold)}.pt",
            )
            _save_history(run_dir, "rsnn", fold, outcome)
            logits = predict_logits(model, data.test.X, device)
            _save_result(
                run_dir, "rsnn", fold, data.test.y, logits.argmax(1), logits,
                {"n_params": count_parameters(model), "n_hidden": n_hidden, **_best_epochs(outcome)},
            )


# ---------------------------------------------------------------- aggregation


def collect_results(run_dir: Path) -> None:
    """Gather the per-fold files into results.csv (metrics), predictions.csv (per trial) and confusion_matrices.csv.

    All 90 folds per model are kept; nothing is averaged here (the later analysis averages the 10 repeats per subject).
    """
    rows, pred_frames, conf_rows = [], [], []
    for model in MODELS:
        for path in sorted((run_dir / "metrics" / model).glob("*.json")):
            rec = json.loads(path.read_text())
            confusion = np.array(rec.pop("confusion"))
            rows.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()})
            for t in range(N_CLASSES):
                for p in range(N_CLASSES):
                    conf_rows.append(
                        {"model": model, "subject": rec["subject"], "repeat": rec["repeat"], "true_class": t,
                         "pred_class": p, "count": int(confusion[t, p])}
                    )
            pred = np.load(run_dir / "predictions" / model / f"{path.stem}.npz")
            pred_frames.append(
                pd.DataFrame(
                    {"model": model, "subject": rec["subject"], "repeat": rec["repeat"],
                     "trial": np.arange(len(pred["y_true"])), "y_true": pred["y_true"], "y_pred": pred["y_pred"]}
                )
            )
    if not rows:
        print("No results to collect.")
        return
    pd.DataFrame(rows).sort_values(["model", "subject", "repeat"]).to_csv(run_dir / "results.csv", index=False)
    pd.concat(pred_frames).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(run_dir / "confusion_matrices.csv", index=False)
    print(f"Saved tables to {run_dir}")