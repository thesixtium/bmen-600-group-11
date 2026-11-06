"""Train ONE model on ONE development fold for one screen (session-1 data only) and write a JSON result.

Arms
  full     pretraining then fine-tuning, the pipeline's own pretrain_then_finetune and configs (v1.4 code, unchanged)
  ftonly   fine-tuning only, from the same random start (same seed, same input statistics), no pretraining (v1.5)

Variants (screen_common.build_variant_data): v1.4 (reference data handling), v1.6 (per-subject normalization),
v1.7 (pooled 80/20 pretraining split of all 8 non-target subjects).

Scores (all session 1): the pretrained model on the target's 288 session-1 trials (zero-shot), on the pretraining
validation data and on the target's 20 % split; the fine-tuned model on the target's 80 % and 20 % splits.
--check compares the final weights with the saved v1 (EEGNet) or v1.4 (RSNN, RNN) checkpoint (variant v1.4, arm full).

    python screen.py --variant v1.4 --arm full   --model rsnn --fold 1:0 --check
    python screen.py --variant v1.6 --arm full   --model ann  --fold 3:0
"""
import argparse
import json
import platform
import time

import torch

from screen_common import (  # noqa: E402
    results_dir,
    HERE, MODELS, SAVED_RUN, accuracy, build_variant_data, get_fold, load_processed, make_model, uses_input_stats,
)
from training import fit, pretrain_then_finetune


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("v1.4", "v1.6", "v1.7"), required=True)
    ap.add_argument("--arm", choices=("full", "ftonly"), required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--fold", required=True, help="target:repeat, e.g. 1:0")
    ap.add_argument("--out", required=True, help="results subfolder, e.g. v1.5")
    ap.add_argument("--check", action="store_true", help="compare the final weights with the saved checkpoint")
    args = ap.parse_args()

    target, repeat = map(int, args.fold.split(":"))
    field, cfg = MODELS[args.model]
    processed = load_processed()
    fold = get_fold(processed, target, repeat)
    data = build_variant_data(processed, fold, field, args.variant)
    device = torch.device("cuda")
    model = make_model(args.model, fold.seed)
    if uses_input_stats(args.model):
        model.set_input_stats(data.pre_train.X)  # pretraining data of this variant only, as benchmark.py does
    model.to(device)
    tag = f"[{args.out} {args.variant} {args.arm} {args.model} s{target}_r{repeat}]"
    t0 = time.perf_counter()
    result = {"variant": args.variant, "arm": args.arm, "model": args.model, "target": target, "repeat": repeat,
              "fold_seed": fold.seed, "n_pre_train": len(data.pre_train.y), "n_pre_val": len(data.pre_val.y)}

    if args.arm == "full":
        outcome = pretrain_then_finetune(model, data, cfg, fold.seed, device, tag=tag)  # model = fine-tuned best
        fine = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(outcome.pretrained_state)
        result.update(pre_target_s1=accuracy(model, data.target_s1, device),
                      pre_pre_val=accuracy(model, data.pre_val, device),
                      pre_ft_val=accuracy(model, data.ft_val, device),
                      pretrain_best_epoch=outcome.pretrain_history[0]["best_epoch"],
                      pretrain_best_val_loss=min(h["val_loss"] for h in outcome.pretrain_history),
                      pretrain_final_train_loss=outcome.pretrain_history[-1]["train_loss"])
        model.load_state_dict(fine)
        ft_hist = outcome.finetune_history
    else:
        # same fine-tuning call as pretrain_then_finetune's second phase, from the untrained model
        _, ft_hist = fit(model, data.ft_train, data.ft_val, epochs=cfg.finetune_epochs, batch_size=cfg.batch_size,
                         learning_rate=cfg.learning_rate, dropout=cfg.finetune_dropout, seed=fold.seed + 1,
                         device=device, log_label=f"{tag} finetune", log_every=cfg.log_every_epochs,
                         grad_clip_norm=cfg.grad_clip_norm)
    result.update(ft_ft_val=accuracy(model, data.ft_val, device), ft_ft_train=accuracy(model, data.ft_train, device),
                  finetune_best_epoch=ft_hist[0]["best_epoch"],
                  finetune_best_val_loss=min(h["val_loss"] for h in ft_hist),
                  max_loss=max(max(h["train_loss"], h["val_loss"]) for h in ft_hist),
                  seconds=time.perf_counter() - t0,
                  environment={"python": platform.python_version(), "torch": torch.__version__,
                               "gpu": torch.cuda.get_device_name(0)})
    if args.arm == "full":
        result["max_loss"] = max(result["max_loss"], max(max(h["train_loss"], h["val_loss"])
                                                         for h in outcome.pretrain_history))
    if args.check:
        saved = torch.load(SAVED_RUN[args.model] / "checkpoints" / args.model / f"s{target}_r{repeat}.pt",
                           map_location="cpu", weights_only=True)["finetuned"]
        mine = model.state_dict()
        result["max_abs_diff_to_saved"] = max(float((mine[k].float().cpu() - saved[k].float()).abs().max()) for k in saved)

    out = results_dir(args.out) / f"{args.variant}_{args.arm}_{args.model}_s{target}_r{repeat}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result.get(k) for k in ("variant", "arm", "model", "target", "pre_target_s1", "ft_ft_val",
                                                 "max_abs_diff_to_saved", "seconds")}))


if __name__ == "__main__":
    main()
