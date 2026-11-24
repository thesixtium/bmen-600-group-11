"""v2.3 screen: fixed LIF reservoirs of 500, 1,000 and 2,000 neurons with a trained dense layer. All arms of ONE
development fold in one process, one JSON per arm in results_reservoir/. Session-1 data only.

Arms: res<H>_r<radius>_<input>, H in {500, 1000, 2000}, recurrent spectral radius in {0 (no recurrence), 1.0},
input in {raw (22 EEG channels), bands (44 mu / beta band inputs)}.
Reservoir code: reservoir.py in this folder (same LIF equations as the pipeline RSNN).
Per-channel input standardization, input gain (median rate 0.1 spikes per step at radius 0) and feature
standardization use the pretraining trials only. The dense layer is trained with the pipeline's
pretrain_then_finetune and RSNN_TRAIN. Split: the group pipeline's (variant "v1.4").

    python screen_reservoir_v2x.py --fold 1:0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]  # experiments/v2.x/: results, audit and logs are written here
EXP = HERE.parent                           # experiments/
sys.path.insert(0, str(EXP / "v1.5" / "code"))  # screen_common.py: data cache, folds, the v1.4 pipeline code
sys.dont_write_bytecode = True

from screen_common import ScreenData, accuracy, build_variant_data, get_fold, load_processed  # noqa: E402
from benchmark_config import RSNN_TRAIN  # noqa: E402
from folds import Split  # noqa: E402
from training import pretrain_then_finetune  # noqa: E402
from reservoir import Readout, draw_weights, fit_input_gain, simulate  # noqa: E402

SIZES, RADII, INPUTS = (500, 1000, 2000), (1.0, 0.0), {"raw": "X", "bands": "X_bands"}
SPLITS = ("pre_train", "pre_val", "ft_train", "ft_val", "target_s1")
SIM_BATCH = {500: 256, 1000: 256, 2000: 128}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    args = ap.parse_args()
    target, repeat = map(int, args.fold.split(":"))
    device = torch.device("cuda")
    out_dir = HERE / "results_reservoir"
    out_dir.mkdir(exist_ok=True)
    processed = load_processed()
    fold = get_fold(processed, target, repeat)
    for inp, field in INPUTS.items():
        data = build_variant_data(processed, fold, field, "v1.4")  # session 1 only
        mean = data.pre_train.X.mean(axis=(0, 2), keepdims=True)
        std = data.pre_train.X.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
        X = {s: torch.from_numpy(((getattr(data, s).X - mean) / std).astype(np.float32)).to(device) for s in SPLITS}
        y = {s: getattr(data, s).y for s in SPLITS}
        for h in SIZES:
            names = {rho: f"res{h}_r{rho:g}_{inp}" for rho in RADII}
            if all((out_dir / f"{n}_s{target}_r{repeat}.json").exists() for n in names.values()):
                continue
            w_in, w_rec = draw_weights(X["pre_train"].shape[1], h, fold.seed)
            gain, rate0 = fit_input_gain(X["pre_train"], w_in.to(device), "lif", seed=fold.seed)
            for rho, name in names.items():
                path = out_dir / f"{name}_s{target}_r{repeat}.json"
                if path.exists():
                    continue
                t0 = time.perf_counter()
                wi, wr = (w_in * gain).to(device), (w_rec * rho).to(device)
                F = {s: simulate(X[s], wi, wr, "lif", 1, batch=SIM_BATCH[h]) for s in SPLITS}
                raw = F["pre_train"]
                f_mean, f_std = raw.mean(0, keepdim=True), raw.std(0, keepdim=True).clamp_min(1e-2)
                fd = ScreenData(*(Split(((F[s] - f_mean) / f_std).cpu().numpy().astype(np.float32), y[s]) for s in SPLITS))
                torch.manual_seed(fold.seed)
                model = Readout(h).to(device)
                outcome = pretrain_then_finetune(model, fd, RSNN_TRAIN, fold.seed, device, tag=f"[{name} s{target}]")
                fine = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(outcome.pretrained_state)
                rate = raw.mean(0)
                res = {"arm": name, "neurons": h, "spectral_radius": rho, "input": inp, "target": target,
                       "repeat": repeat, "fold_seed": fold.seed,
                       "trainable_params": sum(p.numel() for p in model.parameters()),
                       "fixed_weights": int(wi.numel() + (h * h if rho > 0 else 0)), "input_gain": gain,
                       "median_rate": float(rate.median()), "frac_silent": float((rate < 0.005).float().mean()),
                       "pre_pre_train": accuracy(model, fd.pre_train, device),
                       "pre_target_s1": accuracy(model, fd.target_s1, device),
                       "pre_ft_val": accuracy(model, fd.ft_val, device)}
                model.load_state_dict(fine)
                res.update(ft_ft_train=accuracy(model, fd.ft_train, device), ft_ft_val=accuracy(model, fd.ft_val, device),
                           seconds=time.perf_counter() - t0)
                path.write_text(json.dumps(res, indent=1))
                print(json.dumps({k: res[k] for k in ("arm", "target", "pre_pre_train", "ft_ft_val", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
