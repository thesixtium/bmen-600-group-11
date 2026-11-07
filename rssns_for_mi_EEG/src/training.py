"""Shared training loop (EEGNet ANN and RSNN) and prediction helpers.

Both trainable models map a (batch, 22, 256) tensor to 4 logits, so one loop serves both. Pretraining and
fine-tuning use the same loop with different data, epochs and dropout. The checkpoint with the lowest validation
loss is kept (research plan, Section 4). The test session is never passed to this module.

Speed changes compared with the previous version (the protocol is unchanged):
  * validation passes use a large batch (BENCH_EVAL_BATCH, default 288). BatchNorm runs in eval mode and dropout is off,
    so the result does not depend on the batch size (up to float rounding). Far fewer kernel launches per epoch.
  * the running training loss is summed on the GPU, so there is no host synchronization per optimizer step.
Optional, OFF by default (it changes the protocol, see run_benchmark.slurm):
  * BENCH_PATIENCE=N stops PRETRAINING when the validation loss has not improved for N epochs. The kept checkpoint is
    still the epoch with the lowest validation loss. Fine-tuning always runs all of its epochs.
"""

import copy
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from benchmark_config import TrainConfig
from folds import FoldData, Split
from timing import format_duration

EVAL_BATCH_SIZE: int = int(os.environ.get("BENCH_EVAL_BATCH", "288"))
_PATIENCE = os.environ.get("BENCH_PATIENCE")
PRETRAIN_PATIENCE: int | None = int(_PATIENCE) if _PATIENCE else None


@dataclass
class TrainingOutcome:
    pretrained_state: dict
    finetuned_state: dict
    pretrain_history: list[dict]
    finetune_history: list[dict]
    pretrain_seconds: float = 0.0
    finetune_seconds: float = 0.0


def set_dropout(model: nn.Module, p: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def _to_tensors(split: Split, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(split.X).to(device), torch.from_numpy(split.y).to(device)


@torch.no_grad()
def _mean_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = torch.zeros((), device=X.device)
    for i in range(0, len(X), batch_size):
        total += F.cross_entropy(model(X[i : i + batch_size]), y[i : i + batch_size], reduction="sum")
    return total.item() / len(X)  # one synchronization per call


def fit(
    model: nn.Module,
    train: Split,
    val: Split,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    seed: int,
    device: torch.device,
    log_label: str = "",
    log_every: int = 0,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    patience: int | None = None,
) -> tuple[dict, list[dict]]:
    """Adam + cross-entropy; on return `model` holds the weights of the epoch with the lowest validation loss.

    patience=None runs all `epochs`. patience=N stops when the validation loss has not improved for N epochs."""
    X_tr, y_tr = _to_tensors(train, device)
    X_va, y_va = _to_tensors(val, device)
    set_dropout(model, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffler = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # dropout masks

    best_loss, best_state, history = float("inf"), copy.deepcopy(model.state_dict()), []
    best_epoch, since_best = 0, 0
    phase_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_tr), generator=shuffler).to(device)
        running = torch.zeros((), device=device)  # summed on the GPU: no host sync per optimizer step
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            running += loss.detach() * len(idx)
        val_loss = _mean_loss(model, X_va, y_va, eval_batch_size)
        history.append({"epoch": epoch, "train_loss": float(running) / len(X_tr), "val_loss": val_loss})
        if log_every and epoch % log_every == 0 and epoch != epochs:
            elapsed = time.perf_counter() - phase_start
            print(f"[time]     {log_label} epoch {epoch}/{epochs} | val_loss {val_loss:.4f} | "
                  f"elapsed {format_duration(elapsed)}, phase ETA {format_duration(elapsed / epoch * (epochs - epoch))}")
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, since_best = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since_best += 1
            if patience is not None and since_best >= patience:
                print(f"[time]     {log_label} early stop at epoch {epoch}/{epochs} (best epoch {best_epoch}, patience {patience})")
                break

    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    for h in history:
        h["best_epoch"] = best_epoch
    return best_state, history


def pretrain_then_finetune(
    model: nn.Module, data: FoldData, cfg: TrainConfig, seed: int, device: torch.device, tag: str = ""
) -> TrainingOutcome:
    """Cross-subject pretraining, then fine-tuning on the target's session 1 (80/20 split). `tag` labels progress lines."""
    model.to(device)
    common = {"batch_size": cfg.batch_size, "learning_rate": cfg.learning_rate, "device": device,
              "log_every": cfg.log_every_epochs, "eval_batch_size": EVAL_BATCH_SIZE}
    t0 = time.perf_counter()
    pre_state, pre_hist = fit(
        model, data.pre_train, data.pre_val, epochs=cfg.pretrain_epochs, dropout=cfg.pretrain_dropout, seed=seed,
        log_label=f"{tag} pretrain".strip(), patience=PRETRAIN_PATIENCE, **common
    )
    t1 = time.perf_counter()
    ft_state, ft_hist = fit(
        model, data.ft_train, data.ft_val, epochs=cfg.finetune_epochs, dropout=cfg.finetune_dropout, seed=seed + 1,
        log_label=f"{tag} finetune".strip(), **common
    )
    t2 = time.perf_counter()
    return TrainingOutcome(pre_state, ft_state, pre_hist, ft_hist, t1 - t0, t2 - t1)


@torch.no_grad()
def predict_logits(model: nn.Module, X: NDArray[np.float32], device: torch.device, batch_size: int = 64) -> NDArray[np.float32]:
    """Inference adapter for the ANN (continuous scores) and the RSNN (readout of its spiking dynamics)."""
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    return torch.cat([model(X_t[i : i + batch_size]) for i in range(0, len(X_t), batch_size)]).cpu().numpy()
