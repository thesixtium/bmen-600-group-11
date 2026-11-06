"""v1.5 part A: score the SAVED checkpoints of all 90 folds on session-1 data (no training, session 2 never loaded).

For every fold and model (EEGNet from v1, RSNN and RNN from v1.4) both saved states are scored:
  pretrained  (after pretraining on other subjects, best pretraining-validation epoch, before any target data)
      target_s1 : all 288 session-1 trials of the target. The target is not in pretraining, so this is a clean
                  zero-shot score of what pretraining alone gives.
      pre_val   : the validation subjects (optimistic: they chose the pretraining checkpoint)
      ft_val    : the target's 20 % split (58 trials), to pair with the fine-tuned score on the same trials
  finetuned   (after fine-tuning on the target's 80 % split)
      ft_val    : the target's 20 % split (optimistic: it chose the fine-tuning checkpoint)

    python eval_saved.py        -> results/v1.5/saved_eval.csv
"""
import csv

import torch

from screen_common import HERE, MODELS, SAVED_RUN, accuracy, build_variant_data, get_fold, load_processed, make_model, results_dir


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = load_processed()
    out = results_dir("v1.5") / "saved_eval.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (field, _) in MODELS.items():
        for target in range(1, 10):
            for repeat in range(10):
                fold = get_fold(processed, target, repeat)
                data = build_variant_data(processed, fold, field, "v1.4")
                ckpt = torch.load(SAVED_RUN[name] / "checkpoints" / name / f"s{target}_r{repeat}.pt",
                                  map_location="cpu", weights_only=True)
                model = make_model(name, fold.seed).to(device)
                model.load_state_dict(ckpt["pretrained"])
                row = {"model": name, "target": target, "repeat": repeat,
                       "pre_target_s1": accuracy(model, data.target_s1, device),
                       "pre_pre_val": accuracy(model, data.pre_val, device),
                       "pre_ft_val": accuracy(model, data.ft_val, device)}
                model.load_state_dict(ckpt["finetuned"])
                row["ft_ft_val"] = accuracy(model, data.ft_val, device)
                rows.append(row)
        print(f"{name}: 90 folds scored")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
