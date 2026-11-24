"""v2.2c and v2.2d screens: training-schedule changes on the front-end RSNN, each with an EEGNet counterpart.
One arm on one development fold, session-1 data only.

Arms
  sched_rsnn64    v2.2c: conv_rsnn64 with a learning-rate schedule in both phases
  sched_eegnet    v2.2c: EEGNet with the same schedule (its own batch size 16, no gradient clipping, max-norm)
  ft1000_rsnn64   v2.2d: conv_rsnn64 with 1,000 fine-tuning epochs instead of 100 (constant learning rate)
  ft1000_eegnet   v2.2d: EEGNet with 1,000 fine-tuning epochs

Schedule (v2.2c): 5 warm-up epochs (learning rate 0.2, 0.4, ..., 1.0 times 1e-3), then cosine decay to 0 at the last
epoch of the phase. Changed once per epoch.
Cut-offs (v2.2d): the kept model (lowest validation loss so far) is scored after 100, 300 and 1,000 fine-tuning
epochs. The 20 % split both selects the epoch and gives the score, so a longer run can only look better or equal;
the three cut-offs show how much. The first 100 epochs use the same seeds and steps as screen_v2x.py, so the
100-epoch cut-off of ft1000_rsnn64 must equal conv_rsnn64.

fit_v2 below copies the pipeline's training.fit (same seeding, shuffling, clipping, max-norm call, best-validation
rule) and adds the schedule and the cut-offs.

    python screen_v2x_sched.py --arm sched_rsnn64 --fold 1:0
"""
import argparse
import copy
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
from screen_v2x import HERE, accuracy, build_variant_data, get_fold, load_processed, make_model  # noqa: E402
from benchmark_config import ANN_TRAIN, EEGNET_CFG, RSNN_TRAIN  # noqa: E402
from models import EEGNet  # noqa: E402
from training import _mean_loss, _to_tensors, set_dropout  # noqa: E402

ARMS = {  # name: (model, schedule, fine-tuning epochs, cut-offs)
    "sched_rsnn64": ("rsnn", True, 100, (100,)),
    "sched_eegnet": ("eegnet", True, 100, (100,)),
    "ft1000_rsnn64": ("rsnn", False, 1000, (100, 300, 1000)),
    "ft1000_eegnet": ("eegnet", False, 1000, (100, 300, 1000)),
}
WARMUP = 5


def lr_factor(epoch: int, epochs: int) -> float:
    """Epoch is 1-based. Linear warm-up over WARMUP epochs, then cosine decay to 0 at the last epoch."""
    if epoch <= WARMUP:
        return epoch / WARMUP
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - WARMUP) / (epochs - WARMUP)))


def fit_v2(model, train, val, *, epochs, batch_size, learning_rate, dropout, seed, device, grad_clip_norm,
           schedule=False, cutoffs=()):
    """Returns (history, {cut-off: (best state so far, its epoch)}). On return the model holds the overall best."""
    X_tr, y_tr = _to_tensors(train, device)
    X_va, y_va = _to_tensors(val, device)
    set_dropout(model, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffler = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    best_loss, best_state, best_epoch, history, kept = float("inf"), copy.deepcopy(model.state_dict()), 0, [], {}
    for epoch in range(1, epochs + 1):
        if schedule:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_factor(epoch, epochs)
        model.train()
        order = torch.randperm(len(X_tr), generator=shuffler).to(device)
        running = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if hasattr(model, "apply_max_norm"):
                model.apply_max_norm()
            running += loss.item() * len(idx)
        val_loss = _mean_loss(model, X_va, y_va, batch_size)
        history.append({"epoch": epoch, "train_loss": running / len(X_tr), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
        if epoch in cutoffs:
            kept[epoch] = (copy.deepcopy(best_state), best_epoch)
    model.load_state_dict(best_state)
    return history, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", required=True)
    args = ap.parse_args()
    target, repeat = map(int, args.fold.split(":"))
    out = HERE / "results" / f"{args.arm}_s{target}_r{repeat}.json"
    if out.exists():
        print(f"skip {out.name} (done)")
        return
    kind, schedule, ft_epochs, cutoffs = ARMS[args.arm]
    device = torch.device("cuda")
    processed = load_processed()
    fold = get_fold(processed, target, repeat)
    data = build_variant_data(processed, fold, "X", "v1.4")  # session 1 only
    if kind == "eegnet":  # as benchmark.py: seed, then the model; no input standardization (batch normalization)
        cfg = ANN_TRAIN
        torch.manual_seed(fold.seed)
        model = EEGNet(EEGNET_CFG)
    else:
        cfg = RSNN_TRAIN
        model = make_model("conv_rsnn64", fold.seed)
        model.set_input_stats(data.pre_train.X)
    model.to(device)
    common = dict(batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, device=device,
                  grad_clip_norm=cfg.grad_clip_norm, schedule=schedule)
    t0 = time.perf_counter()
    pre_hist, _ = fit_v2(model, data.pre_train, data.pre_val, epochs=cfg.pretrain_epochs, dropout=cfg.pretrain_dropout,
                         seed=fold.seed, **common)
    res = {"arm": args.arm, "model": kind, "schedule": schedule, "finetune_epochs": ft_epochs, "target": target,
           "repeat": repeat, "fold_seed": fold.seed, "batch_size": cfg.batch_size,
           "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "pre_pre_train": accuracy(model, data.pre_train, device), "pre_target_s1": accuracy(model, data.target_s1, device),
           "pre_ft_val": accuracy(model, data.ft_val, device),
           "pretrain_best_epoch": min(pre_hist, key=lambda h: h["val_loss"])["epoch"]}
    ft_hist, kept = fit_v2(model, data.ft_train, data.ft_val, epochs=ft_epochs, dropout=cfg.finetune_dropout,
                           seed=fold.seed + 1, cutoffs=cutoffs, **common)
    for c in cutoffs:  # the kept model within the first c fine-tuning epochs
        state, epoch = kept[c]
        model.load_state_dict(state)
        res[f"ft_val_{c}"] = accuracy(model, data.ft_val, device)
        res[f"ft_train_{c}"] = accuracy(model, data.ft_train, device)
        res[f"best_epoch_{c}"] = epoch
    last = cutoffs[-1]
    losses = [h[k] for h in pre_hist + ft_hist for k in ("train_loss", "val_loss")]
    res.update(ft_ft_val=res[f"ft_val_{last}"], ft_ft_train=res[f"ft_train_{last}"],
               finetune_best_epoch=res[f"best_epoch_{last}"],
               max_loss=max(losses) if all(math.isfinite(v) for v in losses) else float("inf"),
               max_pretrain_train_loss=max(h["train_loss"] for h in pre_hist), seconds=time.perf_counter() - t0)
    hist = out.parent / "histories"  # per-epoch training and validation loss, for the loss curves of the report
    hist.mkdir(exist_ok=True)
    (hist / out.name).write_text(json.dumps({"pretrain": pre_hist, "finetune": ft_hist}))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res.get(k) for k in ("arm", "target", "pre_target_s1", "ft_val_100", "ft_ft_val", "seconds")}))


if __name__ == "__main__":
    main()
