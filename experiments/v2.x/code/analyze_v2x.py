"""Summary of the v2.x screens (target's 20 % split of session 1, 9 development folds). Works on partial results.

Reads results/<arm>_s<T>_r0.json (screen_v2x.py) and results_reservoir/<arm>_s<T>_r0.json (screen_reservoir_v2x.py).
Reference EEGNet: the session-1 screen of 4 November on the same folds and split
(experiments/v1.5/results/v1.4_full_ann_s<T>_r0.json).
Rules (../README.md): an arm beats base17 if it is better for at least 7 of 9 subjects and the
mean gain is at least 0.02; the other comparisons need 7 of 9.
Writes summary.csv, comparisons.csv and per_subject_ft_val.csv into results/.
"""
import csv
import json
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parents[1]  # experiments/v2.x/
REF = HERE.parent / "v1.5" / "results"
ARMS = ["base17", "wide256", "wide1024", "conv_rsnn17", "conv_rnn17", "conv_rsnn64", "conv_rsnn256", "conv_rnn64", "conv_fflif64", "frozen_rsnn64", "frozen_rnn64",
        "sched_rsnn64", "sched_eegnet", "ft1000_rsnn64", "ft1000_eegnet"]
RES = [f"res{h}_r{r}_{i}" for i in ("raw", "bands") for h in (500, 1000, 2000) for r in ("0", "1")]
KEYS = ("pre_pre_train", "pre_target_s1", "ft_ft_train", "ft_ft_val")

# Training on the GPU is not exactly repeatable (front-end convolutions), so a fold that was run twice (the first
# pass is kept in results/first_pass/ when a job is rerun to save its loss curve) counts with the MEAN of its runs.
# repeatability.csv lists the two runs side by side: that difference is the run-to-run noise of one fold.
data, repeat_rows = {}, []
for arm in ARMS + RES:
    folder = HERE / ("results_reservoir" if arm.startswith("res") else "results")
    d = {}
    for t in range(1, 10):
        runs = [json.loads(q.read_text()) for q in (HERE / "results" / "first_pass" / f"{arm}_s{t}_r0.json",
                                                    folder / f"{arm}_s{t}_r0.json") if q.exists()]
        if not runs:
            continue
        d[t] = dict(runs[-1])
        for k in KEYS:
            if all(k in r for r in runs):
                d[t][k] = mean(r[k] for r in runs)
        d[t]["n_runs"] = len(runs)
        if len(runs) == 2:
            repeat_rows.append({"arm": arm, "target": t, "first_run": round(runs[0]["ft_ft_val"], 3),
                                "second_run": round(runs[1]["ft_ft_val"], 3),
                                "difference_trials": round((runs[1]["ft_ft_val"] - runs[0]["ft_ft_val"]) * 58)})
    if d:
        data[arm] = d
saved = HERE / "audit" / "eegnet_saved_checkpoints.json"  # the EEGNet whose block 1 the frozen arms use
if saved.exists():
    data["EEGNet (saved checkpoint)"] = {r["target"]: r for r in json.loads(saved.read_text()) if r["repeat"] == 0}
ref = {t: json.loads(p.read_text()) for t in range(1, 10) if (p := REF / f"v1.4_full_ann_s{t}_r0.json").exists()}
if ref:
    data["EEGNet (reference)"] = ref

rows = []
for arm, d in data.items():
    row = {"arm": arm, "folds": len(d), "runs_per_fold": max(v.get("n_runs", 1) for v in d.values()), "trainable_params": next(iter(d.values())).get("trainable_params", "")}
    for k in KEYS:
        vals = [v[k] for v in d.values() if k in v]
        row[k] = round(mean(vals), 3) if vals else ""
    secs = [v["seconds"] for v in d.values() if "seconds" in v]
    row["median_seconds"] = round(median(secs)) if secs else ""
    row["non_finite_loss_folds"] = sum(1 for v in d.values() if v.get("max_loss") == float("inf"))
    rows.append(row)


def compare(a, b, question, need_gain=False):
    if a not in data or b not in data:
        return None
    common = sorted(set(data[a]) & set(data[b]))
    diff = [data[a][t]["ft_ft_val"] - data[b][t]["ft_ft_val"] for t in common]
    better, worse = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
    passes = better >= 7 and (mean(diff) >= 0.02 or not need_gain)
    return {"question": question, "arm": a, "against": b, "folds": len(common), "mean_difference": round(mean(diff), 3),
            "better": better, "worse": worse, "passes_rule": passes if len(common) == 9 else "partial"}


comps = [compare(a, "base17", "beats the group's RSNN", True) for a in ARMS[1:] + RES]
comps += [compare("conv_rsnn64", "conv_fflif64", "recurrence helps with a front end"),
          compare("conv_rnn64", "conv_rsnn64", "non-spiking twin against spiking"),
          compare("conv_rsnn64", "EEGNet (reference)", "against EEGNet"),
          compare("conv_rsnn256", "EEGNet (reference)", "against EEGNet"),
          compare("conv_rsnn256", "conv_rsnn64", "more neurons with a front end"),
          compare("conv_rnn17", "conv_rsnn17", "non-spiking twin against spiking"),
          compare("conv_rsnn64", "conv_rsnn17", "more neurons with a front end"),
          compare("sched_rsnn64", "conv_rsnn64", "learning-rate schedule helps the RSNN"),
          compare("sched_eegnet", "EEGNet (reference)", "learning-rate schedule helps EEGNet"),
          compare("sched_rsnn64", "sched_eegnet", "RSNN against EEGNet, both with the schedule"),
          compare("ft1000_rsnn64", "conv_rsnn64", "1,000 fine-tuning epochs help the RSNN"),
          compare("ft1000_eegnet", "EEGNet (reference)", "1,000 fine-tuning epochs help EEGNet"),
          compare("ft1000_rsnn64", "ft1000_eegnet", "RSNN against EEGNet, both with 1,000 epochs"),
          compare("frozen_rsnn64", "conv_rsnn64", "frozen EEGNet stage against trained stage"),
          compare("frozen_rsnn64", "EEGNet (saved checkpoint)", "against the EEGNet its stage comes from"),
          compare("frozen_rnn64", "EEGNet (saved checkpoint)", "against the EEGNet its stage comes from"),
          compare("frozen_rnn64", "frozen_rsnn64", "non-spiking twin against spiking")]
comps += [compare(f"res{h}_r1_{i}", f"res{h}_r0_{i}", "reservoir recurrence helps") for i in ("raw", "bands") for h in (500, 1000, 2000)]
comps = [c for c in comps if c]
cut = [{"arm": a, **{f"ft_val_{c}": round(mean(v[f"ft_val_{c}"] for v in data[a].values()), 3) for c in (100, 300, 1000)},
        **{f"median_best_epoch_{c}": median(v[f"best_epoch_{c}"] for v in data[a].values()) for c in (100, 300, 1000)},
        "folds": len(data[a])} for a in ("ft1000_rsnn64", "ft1000_eegnet") if a in data]
per_subject = [{"arm": a, **{f"s{t}": round(d[t]["ft_ft_val"], 3) if t in d else "" for t in range(1, 10)}} for a, d in data.items()]

(HERE / "results").mkdir(exist_ok=True)
for name, table in (("summary.csv", rows), ("comparisons.csv", comps), ("per_subject_ft_val.csv", per_subject),
                    ("finetune_cutoffs.csv", cut), ("repeatability.csv", repeat_rows)):
    if not table:
        continue
    with open(HERE / "results" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    print(f"\n{name}")
    for r in table:
        print("  ", r)
