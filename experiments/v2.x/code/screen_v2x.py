"""v2.1 / v2.2 screen: train ONE arm on ONE development fold (session-1 data only) and write one JSON.

Arms (see ../README.md)
  base17        the main pipeline's RSNN architecture (22 raw EEG inputs -> 17 recurrent LIF neurons, 256 steps),
                trained with this screen's settings (batch 64, gradient clipping, v1.4 initialisation)   (reference)
  wide256       the same with 256 neurons                                                              (v2.1)
  wide1024      the same with 1,024 neurons                                                            (v2.1)
  conv_rsnn17   ConvFrontEnd -> 17 recurrent LIF neurons (the group RSNN's size), 64 steps             (v2.2)
  conv_rnn17    ConvFrontEnd -> 17 recurrent ELU units (its non-spiking twin)                          (v2.2)
  conv_rsnn64   ConvFrontEnd -> 64 recurrent LIF neurons, 64 steps                                     (v2.2)
  conv_rsnn256  ConvFrontEnd -> 256 recurrent LIF neurons, 64 steps                                    (v2.2)
  conv_rnn64    ConvFrontEnd -> 64 recurrent ELU units (non-spiking twin)                              (v2.2)
  conv_fflif64  ConvFrontEnd -> 64 LIF neurons, recurrent weights fixed at zero (no-recurrence twin)   (v2.2)

ConvFrontEnd is EEGNet's block 1 (temporal convolution, depthwise spatial convolution, batch normalization, ELU,
average pooling by 4, dropout), trained end to end with the recurrent layer. It turns (B, 22, 256) into (B, 8, 64):
8 feature maps, 64 steps of 31 ms. EEGNet's max-norm limit on the spatial weights is not used here.

Data: the group pipeline's split (5 pretraining + 3 validation subjects, target's session 1 split 80 / 20), built by
screen_common.build_variant_data(..., "X", "v1.4"). The target's session 2 is never loaded.
Training: the pipeline's pretrain_then_finetune with RSNN_TRAIN (500 + 100 epochs, Adam 1e-3, batch 64, dropout
0.25 / 0.5, gradient clipping 1.0, lowest validation loss kept). Model code: the frozen v1.4 pipeline copy.

    python screen_v2x.py --arm conv_rsnn64 --fold 1:0          add --smoke for a 2 + 2 epoch test (results_smoke/)
"""
import argparse
import json
import math
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]  # experiments/v2.x/: results, audit and logs are written here
EXP = HERE.parent                           # experiments/
sys.path.insert(0, str(EXP / "v1.5" / "code"))  # screen_common.py: data cache, folds, the v1.4 pipeline code
sys.dont_write_bytecode = True

from screen_common import accuracy, build_variant_data, get_fold, load_processed  # noqa: E402
from benchmark_config import EEGNET_CFG, RNN_CFG, RSNN_CFG, RSNN_TRAIN  # noqa: E402
from models import RecurrentELUNet, RecurrentLIFNet  # noqa: E402
from training import pretrain_then_finetune  # noqa: E402

ARMS = {  # name: (front end, recurrent kind, hidden neurons)
    "base17": (False, "lif", 17),
    "wide256": (False, "lif", 256),
    "wide1024": (False, "lif", 1024),
    "conv_rsnn17": (True, "lif", 17),   # the group RSNN's size, with the front end
    "conv_rnn17": (True, "elu", 17),    # its non-spiking twin
    "conv_rsnn64": (True, "lif", 64),
    "conv_rsnn256": (True, "lif", 256),
    "conv_rnn64": (True, "elu", 64),
    "conv_fflif64": (True, "fflif", 64),
}
N_MAPS = EEGNET_CFG.F1 * EEGNET_CFG.D  # 8 feature maps


class ConvFrontEnd(nn.Module):
    """EEGNet block 1. Input (B, 22, 256) in microvolts, output (B, 8, 64)."""

    def __init__(self) -> None:
        super().__init__()
        c = EEGNET_CFG
        self.register_buffer("ch_mean", torch.zeros(1, c.n_channels, 1))  # fixed, from the pretraining trials
        self.register_buffer("ch_std", torch.ones(1, c.n_channels, 1))
        self.temporal = nn.Conv2d(1, c.F1, (1, c.kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(c.F1, eps=c.bn_eps, momentum=c.bn_momentum)
        self.spatial = nn.Conv2d(c.F1, N_MAPS, (c.n_channels, 1), groups=c.F1, bias=False)
        self.bn2 = nn.BatchNorm2d(N_MAPS, eps=c.bn_eps, momentum=c.bn_momentum)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, c.pool1))
        self.dropout = nn.Dropout(0.25)  # probability is set by the training schedule

    @torch.no_grad()
    def set_input_stats(self, X) -> None:
        X_t = torch.as_tensor(X, dtype=torch.float32)
        self.ch_mean.copy_(X_t.mean(dim=(0, 2)).view(1, -1, 1))
        self.ch_std.copy_(X_t.std(dim=(0, 2)).clamp_min(1e-6).view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x - self.ch_mean) / self.ch_std).unsqueeze(1)  # (B, 1, 22, 256)
        x = self.bn1(self.temporal(x))
        x = self.act(self.bn2(self.spatial(x)))  # (B, 8, 1, 256)
        return self.dropout(self.pool(x)).squeeze(2)  # (B, 8, 64)


class FrontEndNet(nn.Module):
    """ConvFrontEnd followed by a recurrent net of the pipeline (its own input standardization stays the identity)."""

    def __init__(self, recurrent: nn.Module) -> None:
        super().__init__()
        self.front = ConvFrontEnd()
        self.recurrent_net = recurrent

    def set_input_stats(self, X) -> None:
        self.front.set_input_stats(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recurrent_net(self.front(x))  # the recurrent nets run min(256, T) = 64 steps


def make_model(arm: str, seed: int) -> nn.Module:
    front, kind, hidden = ARMS[arm]
    torch.manual_seed(seed)
    n_in = N_MAPS if front else EEGNET_CFG.n_channels
    if kind == "elu":
        rec = RecurrentELUNet(n_in, hidden, 4, RNN_CFG)
    else:
        rec = RecurrentLIFNet(n_in, hidden, 4, RSNN_CFG)
        if kind == "fflif":  # no recurrence: weight zero and frozen (Adam and clipping skip it)
            with torch.no_grad():
                rec.recurrent.weight.zero_()
            rec.recurrent.weight.requires_grad_(False)
    return FrontEndNet(rec) if front else rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", required=True, help="target:repeat, e.g. 1:0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    target, repeat = map(int, args.fold.split(":"))
    out = HERE / ("results_smoke" if args.smoke else "results") / f"{args.arm}_s{target}_r{repeat}.json"
    if out.exists():
        print(f"skip {out.name} (done)")
        return
    cfg = replace(RSNN_TRAIN, pretrain_epochs=2, finetune_epochs=2) if args.smoke else RSNN_TRAIN
    device = torch.device("cuda")
    processed = load_processed()
    fold = get_fold(processed, target, repeat)
    data = build_variant_data(processed, fold, "X", "v1.4")  # session 1 only
    model = make_model(args.arm, fold.seed)
    model.set_input_stats(data.pre_train.X)  # pretraining trials only
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = f"[{args.arm} s{target}_r{repeat}]"
    t0 = time.perf_counter()
    outcome = pretrain_then_finetune(model, data, cfg, fold.seed, device, tag=tag)  # model = fine-tuned best
    fine = {k: v.clone() for k, v in model.state_dict().items()}
    losses = [h[k] for h in outcome.pretrain_history + outcome.finetune_history for k in ("train_loss", "val_loss")]
    front, kind, hidden = ARMS[args.arm]
    res = {"arm": args.arm, "front_end": front, "kind": kind, "hidden": hidden, "steps": 64 if front else 256,
           "target": target, "repeat": repeat, "fold_seed": fold.seed, "trainable_params": n_train,
           "batch_size": cfg.batch_size, "pretrain_epochs": cfg.pretrain_epochs, "finetune_epochs": cfg.finetune_epochs}
    model.load_state_dict(outcome.pretrained_state)
    res.update(pre_pre_train=accuracy(model, data.pre_train, device), pre_pre_val=accuracy(model, data.pre_val, device),
               pre_target_s1=accuracy(model, data.target_s1, device), pre_ft_val=accuracy(model, data.ft_val, device),
               pretrain_best_epoch=outcome.pretrain_history[0]["best_epoch"])
    model.load_state_dict(fine)
    res.update(ft_ft_train=accuracy(model, data.ft_train, device), ft_ft_val=accuracy(model, data.ft_val, device),
               finetune_best_epoch=outcome.finetune_history[0]["best_epoch"],
               max_loss=max(losses) if all(math.isfinite(v) for v in losses) else float("inf"),
               pretrain_seconds=outcome.pretrain_seconds, finetune_seconds=outcome.finetune_seconds,
               seconds=time.perf_counter() - t0,
               environment={"python": platform.python_version(), "torch": torch.__version__,
                            "gpu": torch.cuda.get_device_name(0)})
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = out.parent / "histories"  # per-epoch training and validation loss, for the loss curves of the report
    hist.mkdir(exist_ok=True)
    (hist / out.name).write_text(json.dumps({"pretrain": outcome.pretrain_history, "finetune": outcome.finetune_history}))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("arm", "target", "trainable_params", "pre_target_s1", "ft_ft_val", "seconds")}))


if __name__ == "__main__":
    main()
