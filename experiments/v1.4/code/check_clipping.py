"""Pre-run check of option G (v1.4): does gradient clipping stop the RNN blow-up of a fold that diverged in v1.3?

Rebuilds one fold exactly as benchmark.run_rnn_benchmark does (same seed, same 44 band inputs, same input
standardization), then trains the RNN for a few PRETRAINING epochs twice: without clipping (v1.3) and with the
v1.4 cap. Only pretraining subjects' session 1 is used; the test session is not touched.
CPU numerics differ from the GPU of the real run, so the no-clip run may not explode exactly as it did there.
  usage (from experiments/v1.4/code/):  ../../.venv/bin/python check_clipping.py [subject] [repeat] [epochs]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "rssns_for_mi_EEG" / "src"))
import os
os.chdir(Path(__file__).parent / "rssns_for_mi_EEG")  # data/ paths are relative to the project folder

import torch

from benchmark_config import GRAD_CLIP_NORM_RECURRENT, N_CLASSES, N_RECURRENT_INPUTS, RNN_CFG, RNN_TRAIN, EEGNET_CFG, RSNN_CFG
from download_data import download_data
from folds import build_fold_data, generate_folds
from models import RecurrentELUNet, model_summary
from preprocess_data import preprocess_data
from training import fit

args = [int(a) for a in sys.argv[1:]]
subject, repeat, epochs = args + [3, 0, 5][len(args):]  # default: fold s3_r0, 5 epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed = preprocess_data(download_data(redo=False))
fold = next(f for f in generate_folds(processed) if f.target == subject and f.repeat == repeat)
data = build_fold_data(processed, fold, field="X_bands")
n_hidden = model_summary(EEGNET_CFG, RSNN_CFG)["rsnn_hidden"]
print(f"fold s{subject}_r{repeat}, seed {fold.seed}, device {device}, {n_hidden} hidden, {len(data.pre_train.X)} pretrain trials")

for clip in (None, GRAD_CLIP_NORM_RECURRENT):
    torch.manual_seed(fold.seed)  # same init for both runs
    model = RecurrentELUNet(N_RECURRENT_INPUTS, n_hidden, N_CLASSES, RNN_CFG).to(device)
    model.set_input_stats(data.pre_train.X)
    _, hist = fit(model, data.pre_train, data.pre_val, epochs=epochs, batch_size=RNN_TRAIN.batch_size,
                  learning_rate=RNN_TRAIN.learning_rate, dropout=RNN_TRAIN.pretrain_dropout, seed=fold.seed,
                  device=device, grad_clip_norm=clip)
    print(f"\nclip = {clip}")
    for h in hist:
        extra = f"  clipped {h['clipped_fraction']:.0%} of steps, max grad norm {h['max_grad_norm']:.3g}" if clip else ""
        print(f"  epoch {h['epoch']}: train {h['train_loss']:.4g}  val {h['val_loss']:.4g}{extra}")
