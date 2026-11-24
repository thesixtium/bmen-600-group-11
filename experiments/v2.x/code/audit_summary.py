"""Prints the outcome of the audit runs (audit/*.json and the frozen-stage repeats). Run: python audit_summary.py"""
import glob
import json
import os
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parents[1]  # experiments/v2.x/
os.chdir(HERE)

print("label-shuffle test (training labels permuted; chance 0.25, binomial threshold for 58 trials 0.362)")
for f in sorted(glob.glob("audit/shuffle_*.json")):
    d = json.load(open(f))
    print(f"  {d['arm']:14s} s{d['target']}  20 % split {d['ft_val_true_labels']:.3f}  pretraining validation {d['pre_val_true_labels']:.3f}")

if os.path.exists("audit/firing_s3.json"):
    d = json.load(open("audit/firing_s3.json"))
    print(f"\nfiring and gradients, frozen_rsnn64 s3: initial loss {d['initial_loss']:.3f} (ln 4 = 1.386), 20 % split {d['ft_ft_val']:.3f}")
    for k in ("at_init", "after_finetuning"):
        x = d[k]
        print(f"  {k:17s} median rate {x['median_rate']:.3f}  silent {x['frac_silent']:.3f}  saturated {x['frac_saturated']:.3f}  "
              f"gradient norms {({n: round(v, 4) for n, v in x['grad_norms'].items()})}")

print("\nfrozen_rsnn64 against the EEGNet checkpoint its stage comes from, per repeat (20 % split)")
eeg = json.load(open("audit/eegnet_saved_checkpoints.json"))
for r in (0, 1, 2):
    paths = [f"results/frozen_rsnn64_s{t}_r{r}.json" for t in range(1, 10)]
    if not all(os.path.exists(p) for p in paths):
        print(f"  repeat {r}: incomplete")
        continue
    fr = [json.load(open(p))["ft_ft_val"] for p in paths]
    e = [x["ft_ft_val"] for x in eeg if x["repeat"] == r]
    dd = [a - b for a, b in zip(fr, e)]
    print(f"  repeat {r}: frozen {mean(fr):.3f}  EEGNet {mean(e):.3f}  gap {mean(dd):+.3f}  better {sum(x > 0 for x in dd)}  "
          f"worse {sum(x < 0 for x in dd)}  per subject in trials {[round(x * 58) for x in dd]}")

print("\nfine-tuning cut-offs (kept model within the first N epochs)")
for a in ("ft1000_rsnn64", "ft1000_eegnet"):
    v = [json.load(open(f)) for f in glob.glob(f"results/{a}_s*_r0.json")]
    if v:
        print(f"  {a:14s} folds {len(v)}  " + "  ".join(f"{c}: {mean(x[f'ft_val_{c}'] for x in v):.3f}" for c in (100, 300, 1000)))
for a in ("sched_rsnn64", "sched_eegnet", "wide256", "wide1024"):
    v = [json.load(open(f)) for f in glob.glob(f"results/{a}_s*_r0.json")]
    if v:
        print(f"  {a:14s} folds {len(v)}  20 % split {mean(x['ft_ft_val'] for x in v):.3f}  largest loss {max(x['max_loss'] for x in v):.2f}")
print(f"\nreservoir results: {len(glob.glob('results_reservoir/*.json'))} of 108")
