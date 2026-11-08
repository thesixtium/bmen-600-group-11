"""Session-1 scores for Table 31 from SAVED checkpoints (no training; session 2 never reaches a model).

For one run, this script imports that run's own pipeline code (<run>/code/src), rebuilds the 9 development folds
(targets 1 to 9, repeat 0) exactly as that run built them, loads the saved fine-tuned weights and scores them on the
target's 20 % session-1 split (ft_val, 57 or 58 trials). This is the right block of Table 31 ("Session 1, after
fine-tuning, 20 % split"). Its v1.4 RSNN value (0.494) equals the v1.4 reference of the v1.6 screen.

The converted SNN is not trained: it is made from the saved fine-tuned EEGNet exactly as run_snn_benchmark does
(calibration on the target's fine-tuning training trials), then scored on the same ft_val trials.

Usage (from the repository root; needs a GPU or patience, and the dataset, which MOABB downloads on first use):
    python experiments/summary/score_saved.py v1.3     -> experiments/summary/session1_scores/v1.3.json
    python experiments/summary/score_saved.py v1.3 --data <folder>   (dataset folder; default rssns_for_mi_EEG/data,
                                                                       which the repository's .gitignore excludes)

One process per run, because every run has its own models.py / benchmark_config.py under the same module names.
The test split (session 2) is built by build_fold_data but is never passed to a model here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent      # experiments/summary/
EXP = HERE.parent                           # experiments/
REPO = EXP.parent

# run name -> (folder in experiments/, run id, models with checkpoints there)
RUNS = {
    "v1": ("v1", "20261029_141407", ["ann", "snn", "rsnn", "rnn"]),
    "v1.1": ("v1.1", "20261031_164232", ["rsnn"]),
    "v1.2": ("v1.2", "20261101_191212", ["rsnn"]),
    "v1.3": ("v1.3", "20261102_163908", ["rsnn", "rnn"]),
    "v1.4": ("v1.4", "20261103_175004", ["rsnn", "rnn"]),
    "v1.6": ("v1.6-full", "20261107_085030", ["ann", "snn", "rnn"]),
    "v1.7": ("v1.7-full", "20261106_144835", ["ann", "snn", "rsnn", "rnn"]),
    "v1.1-rnn": ("v1.1-rnn", "20261106_144857", ["rnn"]),
    "v1.2-rnn": ("v1.2-rnn", "20261107_093450", ["rnn"]),
    "v1.0-fflif": ("v1.0-fflif", "20261107_160943", ["fflif"]),  # recurrence-free control (RSNN without W_rec)
    "v1.7-fflif": ("v1.7-fflif", "20261107_160945", ["fflif"]),
}
DEV_FOLDS = [(t, 0) for t in range(1, 10)]


def main(name: str) -> None:
    folder, run_id, models = RUNS[name]
    root = EXP / folder / "code"
    data_dir = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else REPO / "rssns_for_mi_EEG" / "data"
    sys.dont_write_bytecode = True  # no __pycache__ inside the experiment folders
    sys.path.insert(0, str(root / "src"))
    # imported here, after sys.path points at this run's code
    import benchmark as bm  # the run's own configs (EEGNET_CFG, RSNN_CFG, SNN_CFG, ...) as the benchmark uses them
    import benchmark_config as bc
    from ann_to_snn import ConvertedEEGNetSNN
    import download_data as dd
    from download_data import download_data
    from folds import build_fold_data, generate_folds
    from models import EEGNet, RecurrentELUNet, RecurrentLIFNet, model_summary
    from preprocess_data import preprocess_data
    from training import predict_logits

    run_dir = EXP / folder / "results" / run_id
    dd.DATA_DIR = data_dir  # keep the dataset out of experiments/
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name}: code {root / 'src'}\n       run  {run_dir}")

    processed = preprocess_data(download_data(redo=False))
    folds = {(f.target, f.repeat): f for f in generate_folds(processed)}
    saved = {(f["target"], f["repeat"]): f["seed"] for f in json.loads((run_dir / "folds.json").read_text())["folds"]}
    assert all(saved[k] == folds[k].seed for k in DEV_FOLDS), "fold seeds differ from the run's folds.json"

    band_inputs = hasattr(bc, "N_RECURRENT_INPUTS")  # v1.3 and later: 44 mu + beta band inputs for the recurrent models
    n_hidden = model_summary(bm.EEGNET_CFG, bm.RSNN_CFG)["rsnn_hidden"]  # the RNN uses the RSNN's size in every scored run
    out = {"run": name, "run_dir": "experiments/" + str(run_dir.relative_to(EXP)), "split": "ft_val (target session 1, 20 %)",
           "folds": "targets 1-9, repeat 0"}
    for model_name in models:
        per_subject = {}
        for target, repeat in DEV_FOLDS:
            fold = folds[(target, repeat)]
            ckpt_name = "ann" if model_name == "snn" else model_name
            ckpt_path = run_dir / "checkpoints" / ckpt_name / f"s{target}_r{repeat}.pt"
            if not ckpt_path.exists():
                continue  # run still training
            field = "X_bands" if (band_inputs and model_name in ("rsnn", "rnn", "fflif")) else "X"
            data = build_fold_data(processed, fold, field=field) if field != "X" else build_fold_data(processed, fold)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["finetuned"]
            X = torch.from_numpy(data.ft_val.X).to(device)
            if model_name in ("ann", "snn"):
                ann = EEGNet(bm.EEGNET_CFG)
                ann.load_state_dict(state)
                ann.to(device).eval()
                if model_name == "ann":
                    pred = predict_logits(ann, data.ft_val.X, device).argmax(1)
                else:
                    # as run_snn_benchmark: calibration on the target's fine-tuning TRAINING trials only
                    snn = ConvertedEEGNetSNN(ann, torch.from_numpy(data.ft_train.X).to(device), bm.SNN_CFG)
                    pred = snn.predict(X)[0].cpu().numpy()
            elif model_name == "fflif":
                from models import FeedforwardLIFNet  # only in the -fflif run folders
                model = FeedforwardLIFNet(bc.N_RECURRENT_INPUTS if band_inputs else bc.N_CHANNELS,
                                          bm.fflif_size()["fflif_hidden"], bc.N_CLASSES, bm.RSNN_CFG)
                model.load_state_dict(state)
                model.to(device)
                pred = predict_logits(model, data.ft_val.X, device).argmax(1)
            else:
                n_in = bc.N_RECURRENT_INPUTS if band_inputs else bc.N_CHANNELS
                cls, cfg = (RecurrentLIFNet, bm.RSNN_CFG) if model_name == "rsnn" else (RecurrentELUNet, bm.RNN_CFG)
                model = cls(n_in, n_hidden, bc.N_CLASSES, cfg)
                model.load_state_dict(state)  # includes the input-statistics buffers of the recurrent models
                model.to(device)
                pred = predict_logits(model, data.ft_val.X, device).argmax(1)
            per_subject[target] = round(float((np.asarray(pred) == data.ft_val.y).mean()), 4)
        values = list(per_subject.values())
        out[model_name] = {"per_subject": per_subject, "n_subjects": len(values),
                           "mean": float(np.mean(values)) if values else None}
        print(f"  {model_name:5s} {len(values)}/9 subjects  mean {np.mean(values) if values else float('nan'):.3f}  {per_subject}")
    (HERE / "session1_scores").mkdir(exist_ok=True)
    (HERE / "session1_scores" / f"{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
