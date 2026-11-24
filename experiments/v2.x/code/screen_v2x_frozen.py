"""v2.2b screen (diagnostic): the feature-extraction stage is COPIED from the trained EEGNet of the same fold and
FROZEN. Only the recurrent layer and the dense layer are trained. One arm on one development fold, session 1 only.

Arms
  frozen_rsnn64   EEGNet block 1 (frozen) -> 64 recurrent LIF neurons, 64 steps
  frozen_rnn64    EEGNet block 1 (frozen) -> 64 recurrent ELU units (non-spiking twin)

Question: if the RSNN gets EEGNet's features, can it use them? (v2.2 asks if it can learn them itself.)
The RSNN borrows a part that EEGNet trained, so this is a diagnostic and not a model for the comparison.

EEGNet weights: the v1 run (experiments/v1, run 20261029_141407; same folds, seeds and split),
checkpoints/ann/s<T>_r0.pt with the keys "pretrained" and "finetuned". Phase-matched, so no phase sees more data than
the group protocol gives it: pretraining uses the block of the PRETRAINED EEGNet (other subjects only); fine-tuning
uses the block of the FINE-TUNED EEGNet (target's 80 % split). The block runs in evaluation mode (fixed batch
normalization statistics, no dropout) and takes the raw input in microvolts, as EEGNet does.
Training calls: the two training.fit calls of the pipeline's pretrain_then_finetune, with RSNN_TRAIN.

    python screen_v2x_frozen.py --arm frozen_rsnn64 --fold 1:0
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.dont_write_bytecode = True
from screen_v2x import EXP, HERE, N_MAPS, accuracy, build_variant_data, get_fold, load_processed  # noqa: E402
from benchmark_config import EEGNET_CFG, RNN_CFG, RSNN_CFG, RSNN_TRAIN  # noqa: E402
from models import EEGNet, RecurrentELUNet, RecurrentLIFNet  # noqa: E402
from training import fit  # noqa: E402

CKPT = EXP / "v1" / "results" / "checkpoints" / "ann"  # not in the repository (size): made by the v1 run
ARMS = {"frozen_rsnn64": ("lif", 64), "frozen_rnn64": ("elu", 64)}


class FrozenFrontNet(nn.Module):
    """Frozen EEGNet block 1 -> recurrent net. The EEGNet lives outside the module tree (not trained, not saved)."""

    def __init__(self, recurrent: nn.Module) -> None:
        super().__init__()
        self.recurrent_net = recurrent
        self._eegnet = [EEGNet(EEGNET_CFG).eval()]  # list: hidden from parameters() and state_dict()

    def load_block(self, state: dict, device) -> None:
        self._eegnet[0].load_state_dict(state)
        self._eegnet[0].to(device).eval()

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        e = self._eegnet[0]
        x = e.pool1(e.act(e.bn2(e.depthwise(e.bn1(e.conv1(x.unsqueeze(1)))))))  # (B, 8, 1, 64), no dropout
        return x.squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recurrent_net(self.features(x))


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
    device, cfg = torch.device("cuda"), RSNN_TRAIN
    processed = load_processed()
    fold = get_fold(processed, target, repeat)
    data = build_variant_data(processed, fold, "X", "v1.4")  # session 1 only
    ckpt = torch.load(CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True)
    kind, hidden = ARMS[args.arm]
    torch.manual_seed(fold.seed)
    rec = (RecurrentELUNet(N_MAPS, hidden, 4, RNN_CFG) if kind == "elu" else RecurrentLIFNet(N_MAPS, hidden, 4, RSNN_CFG))
    model = FrozenFrontNet(rec).to(device)
    common = dict(batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, device=device,
                  log_every=cfg.log_every_epochs, grad_clip_norm=cfg.grad_clip_norm)
    tag = f"[{args.arm} s{target}_r{repeat}]"
    t0 = time.perf_counter()
    model.load_block(ckpt["pretrained"], device)
    _, pre_hist = fit(model, data.pre_train, data.pre_val, epochs=cfg.pretrain_epochs, dropout=cfg.pretrain_dropout,
                      seed=fold.seed, log_label=f"{tag} pretrain", **common)
    res = {"arm": args.arm, "front_end": "frozen EEGNet block 1", "kind": kind, "hidden": hidden, "steps": 64,
           "target": target, "repeat": repeat, "fold_seed": fold.seed,
           "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "pre_pre_train": accuracy(model, data.pre_train, device), "pre_pre_val": accuracy(model, data.pre_val, device),
           "pre_target_s1": accuracy(model, data.target_s1, device), "pre_ft_val": accuracy(model, data.ft_val, device),
           "pretrain_best_epoch": pre_hist[0]["best_epoch"]}
    model.load_block(ckpt["finetuned"], device)
    _, ft_hist = fit(model, data.ft_train, data.ft_val, epochs=cfg.finetune_epochs, dropout=cfg.finetune_dropout,
                     seed=fold.seed + 1, log_label=f"{tag} finetune", **common)
    losses = [h[k] for h in pre_hist + ft_hist for k in ("train_loss", "val_loss")]
    res.update(ft_ft_train=accuracy(model, data.ft_train, device), ft_ft_val=accuracy(model, data.ft_val, device),
               finetune_best_epoch=ft_hist[0]["best_epoch"],
               max_loss=max(losses) if all(math.isfinite(v) for v in losses) else float("inf"),
               seconds=time.perf_counter() - t0)
    hist = out.parent / "histories"  # per-epoch training and validation loss, for the loss curves of the report
    hist.mkdir(exist_ok=True)
    (hist / out.name).write_text(json.dumps({"pretrain": pre_hist, "finetune": ft_hist}))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("arm", "target", "trainable_params", "pre_target_s1", "ft_ft_val", "seconds")}))


if __name__ == "__main__":
    main()
