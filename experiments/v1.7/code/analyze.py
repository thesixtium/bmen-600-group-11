"""Analyze one screen against the decision rules written before the runs the plan for the screens.

    python analyze.py v1.5     part A (saved checkpoints, 90 folds) + part B (fine-tune-only arm, reproduction check)
    python analyze.py v1.6     per-subject normalization against the v1.4 reference on the 9 development folds
    python analyze.py v1.7     pooled pretraining split against the same reference

Writes results/<name>/summary.json and prints the tables. All scores are session 1. The 20 % split (58 trials) also
selected the fine-tuning checkpoint, so its scores are slightly optimistic for every arm alike; the target's full
session 1 is never seen in pretraining, so the pretrained score on it is clean.
Reference for v1.6 / v1.7: the saved v1.4 RSNN and RNN checkpoints of repeat 0 (part A; part B reproduced them
exactly) and, for EEGNet, its v1.5 retrained v1.4 run (EEGNet is not bit-reproducible on the GPU).
"""
import csv
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

from screen_common import CHANCE_288, HERE, results_dir

MODELS = ("ann", "rsnn", "rnn")


def load_saved() -> list[dict]:
    with (results_dir("v1.5") / "saved_eval.csv").open() as fh:
        return [{k: (v if k == "model" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]


def load_jobs(name: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((results_dir(name)).glob("v1.*.json"))]


def per_subject(rows, model, key, repeat=None) -> np.ndarray:
    """Mean over the repeats of each subject (or one repeat), subjects 1..9."""
    out = []
    for s in range(1, 10):
        vals = [r[key] for r in rows if r["model"] == model and r["target"] == s and (repeat is None or r["repeat"] == repeat)]
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def paired(new: np.ndarray, ref: np.ndarray) -> dict:
    d = new - ref
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {"mean_new": float(np.mean(new)), "mean_ref": float(np.mean(ref)), "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)), "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "wilcoxon_p_unadjusted": p, "per_subject_new": [round(float(x), 4) for x in new],
            "per_subject_ref": [round(float(x), 4) for x in ref]}


def fmt(x) -> str:
    return f"{x:.3f}"


def analyze_v15() -> dict:
    saved, jobs = load_saved(), load_jobs("v1.5")
    out = {"part_a": {}, "reproduction": {}, "ftonly_vs_full": {}}
    print("PART A: saved checkpoints, 90 folds, mean of 10 repeats per subject (session 1)")
    print(f"{'model':5} {'pretrained: target S1':>22} {'subj > chance':>14} {'folds > chance':>15} {'val subj':>9} "
          f"{'pre 20%':>8} {'fine 20%':>9} {'fine-tune gain, better':>23}")
    for m in MODELS:
        s1 = per_subject(saved, m, "pre_target_s1")
        folds_above = sum(r["pre_target_s1"] > CHANCE_288 for r in saved if r["model"] == m)
        gain = paired(per_subject(saved, m, "ft_ft_val"), per_subject(saved, m, "pre_ft_val"))
        out["part_a"][m] = {"pre_target_s1_mean": float(s1.mean()), "pre_target_s1_per_subject": [round(float(x), 4) for x in s1],
                            "subjects_above_chance": int((s1 > CHANCE_288).sum()), "folds_above_chance": int(folds_above),
                            "pre_pre_val_mean": float(per_subject(saved, m, "pre_pre_val").mean()),
                            "pre_ft_val_mean": gain["mean_ref"], "ft_ft_val_mean": gain["mean_new"], "finetune_gain": gain}
        print(f"{m:5} {fmt(s1.mean()):>22} {int((s1 > CHANCE_288).sum()):>12}/9 {folds_above:>12}/90 "
              f"{fmt(out['part_a'][m]['pre_pre_val_mean']):>9} {fmt(gain['mean_ref']):>8} {fmt(gain['mean_new']):>9} "
              f"{gain['mean_diff']:+.3f}, {gain['better']}/9 (p {gain['wilcoxon_p_unadjusted']:.3f})".rjust(10))
    print("\nPART B: reproduction (variant v1.4, full arm, --check): max |weight difference| to the saved checkpoint")
    for m in MODELS:
        diffs = {f"s{j['target']}": j.get("max_abs_diff_to_saved") for j in jobs if j["model"] == m and j["arm"] == "full"}
        out["reproduction"][m] = diffs
        print(f"  {m}: {diffs}")
    print("\nPART B: fine-tuning only (no pretraining) against pretraining + fine-tuning, repeat 0, target 20 % split")
    for m in MODELS:
        ft = {j["target"]: j["ft_ft_val"] for j in jobs if j["model"] == m and j["arm"] == "ftonly"}
        if len(ft) < 9:
            print(f"  {m}: only {len(ft)} of 9 folds"); continue
        new = np.array([ft[s] for s in range(1, 10)])
        ref = per_subject(saved, m, "ft_ft_val", repeat=0)
        r = paired(new, ref)
        out["ftonly_vs_full"][m] = r
        print(f"  {m}: fine-tune only {fmt(r['mean_new'])}, with pretraining {fmt(r['mean_ref'])}, "
              f"diff {r['mean_diff']:+.3f}, only-better {r['better']}/9, pretraining-better {r['worse']}/9, p {r['wilcoxon_p_unadjusted']:.3f}")
    # decision rules (plan section 3)
    a = out["part_a"]
    rule1 = (9 - a["rsnn"]["subjects_above_chance"] >= 6) and (a["ann"]["subjects_above_chance"] >= 6)
    f = out["ftonly_vs_full"].get("rsnn")
    rule2 = None if f is None else (abs(f["mean_diff"]) < 0.02 and f["better"] < 7 and f["worse"] < 7)
    out["rules"] = {"pretraining_is_rsnn_first_problem": rule1, "pretraining_gives_rsnn_nothing": rule2}
    print(f"\nRULES: pretraining is the RSNN's first problem: {rule1}; pretraining gives the RSNN nothing: {rule2}")
    return out


def analyze_variant(name: str) -> dict:
    saved, jobs = load_saved(), load_jobs(name)
    out = {"models": {}}
    print(f"{name}: full arm, 9 development folds (repeat 0), against the v1.4 reference (saved checkpoints, repeat 0)")
    for m in MODELS:
        js = {j["target"]: j for j in jobs if j["model"] == m and j["arm"] == "full" and j["variant"] == name}
        if len(js) < 9:
            print(f"  {m}: only {len(js)} of 9 folds"); continue
        res = {}
        # Reference: the saved checkpoints for the RSNN and RNN (v1.5 reproduced them exactly). EEGNet training is not
        # bit-reproducible on the GPU (cuDNN convolution), so its reference is its own v1.5 retrained v1.4 run, made
        # with the same screen code (accuracy difference to the saved v1 weights: mean +0.004, at most 0.017).
        ann_ref = {j["target"]: j for j in load_jobs("v1.5") if j["model"] == "ann" and j["arm"] == "full"}
        for key in ("pre_target_s1", "ft_ft_val"):
            ref = (np.array([ann_ref[s][key] for s in range(1, 10)]) if m == "ann"
                   else per_subject(saved, m, key, repeat=0))
            res[key] = paired(np.array([js[s][key] for s in range(1, 10)]), ref)
        res["pretrain_best_epoch_median"] = float(np.median([js[s]["pretrain_best_epoch"] for s in range(1, 10)]))
        res["pretrain_best_val_loss_median"] = float(np.median([js[s]["pretrain_best_val_loss"] for s in range(1, 10)]))
        res["pre_target_s1_subjects_above_chance"] = int(sum(js[s]["pre_target_s1"] > CHANCE_288 for s in range(1, 10)))
        res["max_loss"] = float(max(js[s]["max_loss"] for s in range(1, 10)))
        res["minutes_per_fold_median"] = float(np.median([js[s]["seconds"] for s in range(1, 10)]) / 60)
        out["models"][m] = res
        for key, label in (("pre_target_s1", "pretrained, target S1"), ("ft_ft_val", "fine-tuned, 20 % split")):
            r = res[key]
            print(f"  {m:4} {label:22}: {fmt(r['mean_ref'])} -> {fmt(r['mean_new'])} ({r['mean_diff']:+.3f}), "
                  f"better {r['better']}/9, worse {r['worse']}/9, p {r['wilcoxon_p_unadjusted']:.3f}")
        print(f"       best pretraining epoch median {res['pretrain_best_epoch_median']:.0f}, best val loss median "
              f"{res['pretrain_best_val_loss_median']:.3f}, max loss {res['max_loss']:.3g}, "
              f"{res['minutes_per_fold_median']:.1f} min per fold")
    mm = out["models"]
    if all(k in mm for k in ("rsnn", "ann")):
        helps = (mm["rsnn"]["ft_ft_val"]["better"] >= 7 and mm["rsnn"]["pre_target_s1"]["better"] >= 7
                 and mm["ann"]["ft_ft_val"]["worse"] <= 2)
        out["rules"] = {"screen_helps": helps}
        print(f"\nRULE: the screen helps (RSNN better on both measures in >= 7/9, EEGNet worse in <= 2/9): {helps}")
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    summary = analyze_v15() if name == "v1.5" else analyze_variant(name)
    (results_dir(name) / "summary.json").write_text(json.dumps(summary, indent=1))
