"""Empirical audit runs for the v2.x screens (session-1 data only). Each sub-command writes one JSON into audit/.

  shuffle  --arm A --fold T:0   Label-shuffle test: the TRAINING labels (pretraining and fine-tuning) are permuted,
                                the validation labels stay true. A sound pipeline must fall to chance on the 20 %
                                split (binomial threshold for 58 trials at 0.25: 0.362). Arms: conv_rsnn64, frozen_rsnn64.
  reset    --fold T:0           State-reset test: the output for one trial must not depend on which trials are in
                                the batch or were run before it (untrained conv_rsnn64 and frozen_rsnn64, eval mode).
  firing   --fold T:0           frozen_rsnn64 trained normally on one fold: share of silent (rate < 0.005) and
                                saturated (rate > 0.5) LIF neurons and the gradient norm per layer, at
                                initialisation and after fine-tuning.
  eegnet   --repeats 0 1 2      Accuracy of the saved EEGNet checkpoints (v1 run) on the 20 % split of every
                                target and the given repeats: the like-for-like reference of the frozen arm, no training.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
import screen_v2x as S  # noqa: E402
import screen_v2x_frozen as Z  # noqa: E402
from screen_common import ScreenData  # noqa: E402
from benchmark_config import EEGNET_CFG, RSNN_CFG, RSNN_TRAIN  # noqa: E402
from folds import Split  # noqa: E402
from models import EEGNet, RecurrentLIFNet  # noqa: E402
from training import fit, pretrain_then_finetune  # noqa: E402

OUT = S.HERE / "audit"
DEV = torch.device("cuda")


def load(target, repeat):
    processed = S.load_processed()
    fold = S.get_fold(processed, target, repeat)
    return fold, S.build_variant_data(processed, fold, "X", "v1.4")


def frozen_model(seed):
    torch.manual_seed(seed)
    return Z.FrozenFrontNet(RecurrentLIFNet(S.N_MAPS, 64, 4, RSNN_CFG)).to(DEV)


def train_frozen(model, data, ckpt, seed):
    c = RSNN_TRAIN
    common = dict(batch_size=c.batch_size, learning_rate=c.learning_rate, device=DEV, log_every=0, grad_clip_norm=c.grad_clip_norm)
    model.load_block(ckpt["pretrained"], DEV)
    fit(model, data.pre_train, data.pre_val, epochs=c.pretrain_epochs, dropout=c.pretrain_dropout, seed=seed, **common)
    model.load_block(ckpt["finetuned"], DEV)
    fit(model, data.ft_train, data.ft_val, epochs=c.finetune_epochs, dropout=c.finetune_dropout, seed=seed + 1, **common)


def cmd_shuffle(args):
    target, repeat = map(int, args.fold.split(":"))
    fold, d = load(target, repeat)
    rng = np.random.default_rng(fold.seed)
    data = ScreenData(Split(d.pre_train.X, rng.permutation(d.pre_train.y)), d.pre_val,
                      Split(d.ft_train.X, rng.permutation(d.ft_train.y)), d.ft_val, d.target_s1)
    if args.arm == "frozen_rsnn64":
        model = frozen_model(fold.seed)
        train_frozen(model, data, torch.load(Z.CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True), fold.seed)
    else:
        model = S.make_model("conv_rsnn64", fold.seed)
        model.set_input_stats(data.pre_train.X)
        pretrain_then_finetune(model, data, RSNN_TRAIN, fold.seed, DEV)
    res = {"test": "label_shuffle", "arm": args.arm, "target": target, "ft_val_true_labels": S.accuracy(model, d.ft_val, DEV),
           "pre_val_true_labels": S.accuracy(model, d.pre_val, DEV), "chance": 0.25, "binomial_threshold_58": 21 / 58}
    (OUT / f"shuffle_{args.arm}_s{target}.json").write_text(json.dumps(res, indent=1))
    print(res)


@torch.no_grad()
def cmd_reset(args):
    target, repeat = map(int, args.fold.split(":"))
    fold, d = load(target, repeat)
    x = torch.from_numpy(d.ft_val.X).to(DEV)
    res = {"test": "state_reset", "target": target}
    conv = S.make_model("conv_rsnn64", fold.seed)
    conv.set_input_stats(d.pre_train.X)
    fro = frozen_model(fold.seed)
    fro.load_block(torch.load(Z.CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True)["finetuned"], DEV)
    for name, m in (("conv_rsnn64", conv.to(DEV)), ("frozen_rsnn64", fro)):
        m.eval()
        alone = m(x[5:6])                 # trial 5 alone
        m(x[:32])                         # other trials run in between
        in_batch = m(x[:32])[5:6]         # trial 5 inside a batch
        after = m(x[5:6])                 # trial 5 alone again
        res[name] = {"max_diff_alone_vs_in_batch": float((alone - in_batch).abs().max()),
                     "max_diff_before_vs_after_other_trials": float((alone - after).abs().max())}
    (OUT / f"reset_s{target}.json").write_text(json.dumps(res, indent=1))
    print(res)


def firing_and_grads(model, split):
    """Mean firing rate per LIF neuron over trials and steps, and gradient norms per trainable tensor, on one split."""
    rec = model.recurrent_net
    x = torch.from_numpy(split.X[:128]).to(DEV)
    y = torch.from_numpy(split.y[:128]).to(DEV)
    model.eval()
    with torch.no_grad():
        f = model.features(x).permute(2, 0, 1)                      # (T, B, 8); the inner net's standardization is identity
        cur = rec.input_proj(f)
        mem = f.new_zeros(f.shape[1], rec.n_hidden)
        spk = f.new_zeros(f.shape[1], rec.n_hidden)
        count = f.new_zeros(rec.n_hidden)
        for t in range(f.shape[0]):
            mem = RSNN_CFG.beta * mem + cur[t] + rec.recurrent(spk) - RSNN_CFG.threshold * spk
            spk = (mem > RSNN_CFG.threshold).float()
            count += spk.sum(0)
        rate = count / (f.shape[0] * f.shape[1])
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    grads = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    return {"median_rate": float(rate.median()), "frac_silent": float((rate < 0.005).float().mean()),
            "frac_saturated": float((rate > 0.5).float().mean()), "grad_norms": grads}


def cmd_firing(args):
    target, repeat = map(int, args.fold.split(":"))
    fold, d = load(target, repeat)
    ckpt = torch.load(Z.CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True)
    model = frozen_model(fold.seed)
    model.load_block(ckpt["pretrained"], DEV)
    res = {"test": "firing_and_gradients", "arm": "frozen_rsnn64", "target": target,
           "initial_loss_expected_ln4": float(np.log(4)), "at_init": firing_and_grads(model, d.pre_train)}
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(d.pre_train.X[:256]).to(DEV)
        res["initial_loss"] = float(F.cross_entropy(model(x), torch.from_numpy(d.pre_train.y[:256]).to(DEV)))
    train_frozen(model, d, ckpt, fold.seed)
    res["after_finetuning"] = firing_and_grads(model, d.ft_train)
    res["ft_ft_val"] = S.accuracy(model, d.ft_val, DEV)
    (OUT / f"firing_s{target}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


@torch.no_grad()
def cmd_eegnet(args):
    rows = []
    for repeat in args.repeats:
        for target in range(1, 10):
            fold, d = load(target, repeat)
            ckpt = torch.load(Z.CKPT / f"s{target}_r{repeat}.pt", map_location="cpu", weights_only=True)
            m = EEGNet(EEGNET_CFG)
            m.load_state_dict(ckpt["finetuned"])
            m.to(DEV)
            rows.append({"target": target, "repeat": repeat, "ft_ft_val": S.accuracy(m, d.ft_val, DEV),
                         "ft_ft_train": S.accuracy(m, d.ft_train, DEV)})
    (OUT / "eegnet_saved_checkpoints.json").write_text(json.dumps(rows, indent=1))
    for r in args.repeats:
        v = [x["ft_ft_val"] for x in rows if x["repeat"] == r]
        print(f"repeat {r}: EEGNet 20 % split mean {np.mean(v):.3f}", [round(a, 3) for a in v])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("shuffle", "reset", "firing", "eegnet"))
    ap.add_argument("--arm", default="frozen_rsnn64")
    ap.add_argument("--fold", default="1:0")
    ap.add_argument("--repeats", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"shuffle": cmd_shuffle, "reset": cmd_reset, "firing": cmd_firing, "eegnet": cmd_eegnet}[a.cmd](a)
