"""Session-2 accuracy after pretraining only and after fine-tuning, from SAVED checkpoints (no training).

Every checkpoint file of the benchmark holds two weight sets:
  "pretrained" : the best-validation state of cross-subject pretraining (5 + 3 other subjects, or the pooled 80/20
                 split from v1.7 on). The target subject has not been seen yet.
  "finetuned"  : the same model after fine-tuning on the target's session 1. These are the weights the benchmark
                 scored on session 2.

For one run, this script imports that run's own pipeline code, rebuilds all 90 folds exactly as the run built them,
loads both weight sets and scores each on the target's session 2 (288 trials). The fine-tuned predictions must equal the
saved predictions/<model>/s<T>_r<R>.npz of the run; this checks that the folds, inputs and models were rebuilt correctly.

The converted SNN is not scored after pretraining: the SNN only exists as a conversion of the fine-tuned EEGNet, and
its calibration uses the target's own trials, so a "pretrained-only SNN" would be a new model, not a saved one.

Usage (from the repository root; needs the dataset, which MOABB downloads on first use; a GPU makes it faster):
    python experiments/summary/score_pretrain_s2.py v1.7     -> experiments/summary/pretrain_s2_scores/v1.7.json
    python experiments/summary/score_pretrain_s2.py v1.7 --data <folder>   (dataset folder; default
                                                    rssns_for_mi_EEG/data, which the repository's .gitignore excludes)
One process per run, because every run has its own models.py / benchmark_config.py under the same module names.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent      # experiments/summary/
EXP = HERE.parent                           # experiments/
REPO = EXP.parent

# run name -> (folder in experiments/, run id, models with their own checkpoints in that run)
RUNS = {
    "v1": ("v1", "20261029_141407", ["ann", "rsnn", "rnn"]),
    "v1.1": ("v1.1", "20261031_164232", ["rsnn"]),
    "v1.2": ("v1.2", "20261101_191212", ["rsnn"]),
    "v1.3": ("v1.3", "20261102_163908", ["rsnn", "rnn"]),
    "v1.4": ("v1.4", "20261103_175004", ["rsnn", "rnn"]),
    "v1.6": ("v1.6-full", "20261107_085030", ["ann", "rnn"]),
    "v1.7": ("v1.7-full", "20261106_144835", ["ann", "rsnn", "rnn"]),
    "v1.1-rnn": ("v1.1-rnn", "20261106_144857", ["rnn"]),
    "v1.2-rnn": ("v1.2-rnn", "20261107_093450", ["rnn"]),
    "v1.0-fflif": ("v1.0-fflif", "20261107_160943", ["fflif"]),
    "v1.7-fflif": ("v1.7-fflif", "20261107_160945", ["fflif"]),
}
STATES = ("pretrained", "finetuned")


def main(name: str) -> None:
    folder, run_id, models = RUNS[name]
    root = EXP / folder / "code"
    data_dir = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else REPO / "rssns_for_mi_EEG" / "data"
    sys.dont_write_bytecode = True  # no __pycache__ inside the experiment folders
    sys.path.insert(0, str(root / "src"))
    # imported here, after sys.path points at this run's code
    import benchmark as bm
    import benchmark_config as bc
    import download_data as dd
    from download_data import download_data
    from folds import build_fold_data, generate_folds
    from models import EEGNet, RecurrentELUNet, RecurrentLIFNet, model_summary
    from preprocess_data import preprocess_data
    from training import predict_logits

    run_dir = EXP / folder / "results" / run_id
    dd.DATA_DIR = data_dir  # keep the dataset out of experiments/
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name}: code {root / 'src'}\n       run  {run_dir}", flush=True)

    processed = preprocess_data(download_data(redo=False))
    folds = {(f.target, f.repeat): f for f in generate_folds(processed)}
    saved = {(f["target"], f["repeat"]): f["seed"] for f in json.loads((run_dir / "folds.json").read_text())["folds"]}
    assert len(saved) == 90 and all(saved[k] == folds[k].seed for k in saved), "fold seeds differ from folds.json"

    band_inputs = hasattr(bc, "N_RECURRENT_INPUTS")  # v1.3 and later: 44 mu + beta band inputs for the recurrent models
    n_in_rec = bc.N_RECURRENT_INPUTS if band_inputs else bc.N_CHANNELS

    def build(model_name):
        if model_name == "ann":
            return EEGNet(bm.EEGNET_CFG)
        if model_name == "fflif":
            from models import FeedforwardLIFNet  # only in the -fflif run folders
            return FeedforwardLIFNet(n_in_rec, bm.fflif_size()["fflif_hidden"], bc.N_CLASSES, bm.RSNN_CFG)
        n_hidden = model_summary(bm.EEGNET_CFG, bm.RSNN_CFG)["rsnn_hidden"]  # as run_rsnn/run_rnn_benchmark
        if model_name == "rsnn":
            return RecurrentLIFNet(n_in_rec, n_hidden, bc.N_CLASSES, bm.RSNN_CFG)
        return RecurrentELUNet(n_in_rec, n_hidden, bc.N_CLASSES, bm.RNN_CFG)

    out = {"run": name, "run_dir": "experiments/" + str(run_dir.relative_to(EXP)), "split": "test (target session 2, 288 trials)",
           "folds": 90, "models": {}}
    t0 = time.perf_counter()
    data_cache = {}
    for model_name in models:
        field = "X_bands" if (band_inputs and model_name in ("rsnn", "rnn", "fflif")) else "X"
        per_fold = {s: {} for s in STATES}
        mismatched = []
        for (target, repeat) in sorted(saved):
            key = (target, repeat, field)
            if key not in data_cache:
                data_cache.clear()  # keep one fold's arrays in memory
                fold = folds[(target, repeat)]
                data_cache[key] = (build_fold_data(processed, fold, field=field) if field != "X"
                                   else build_fold_data(processed, fold))
            data = data_cache[key]
            ckpt = torch.load(run_dir / "checkpoints" / model_name / f"s{target}_r{repeat}.pt",
                              map_location="cpu", weights_only=True)
            ref = np.load(run_dir / "predictions" / model_name / f"s{target}_r{repeat}.npz")
            assert np.array_equal(ref["y_true"], data.test.y), f"test labels differ, s{target}_r{repeat}"
            for state in STATES:
                model = build(model_name)
                model.load_state_dict(ckpt[state])  # includes the input-statistics buffers of the recurrent models
                model.to(device)
                pred = predict_logits(model, data.test.X, device).argmax(1)
                per_fold[state][f"s{target}_r{repeat}"] = round(float((pred == data.test.y).mean()), 6)
                if state == "finetuned":
                    n_diff = int((pred != ref["y_pred"]).sum())
                    if n_diff:
                        mismatched.append((f"s{target}_r{repeat}", n_diff))
        res = {}
        for state in STATES:
            subj = {t: float(np.mean([v for k, v in per_fold[state].items() if k.startswith(f"s{t}_")])) for t in range(1, 10)}
            res[state] = {"per_fold": per_fold[state], "per_subject": subj, "mean": float(np.mean(list(subj.values())))}
        res["finetune_check"] = {"folds_with_any_difference": len(mismatched), "differences": mismatched}
        out["models"][model_name] = res
        print(f"  {model_name:5s} pretrained {res['pretrained']['mean']:.3f}  finetuned {res['finetuned']['mean']:.3f}"
              f"  fine-tune check: {len(mismatched)} folds differ {mismatched[:5]}  ({time.perf_counter() - t0:.0f} s)",
              flush=True)
    (HERE / "pretrain_s2_scores").mkdir(exist_ok=True)
    (HERE / "pretrain_s2_scores" / f"{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
