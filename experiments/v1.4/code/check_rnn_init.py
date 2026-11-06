"""Pre-run check for v1.4: which RNN folds blow up before any training step (at initialization)?

For every fold: build the RNN exactly as benchmark.run_rnn_benchmark does (same seed, 44 band inputs, input
standardization from the pretraining subjects), then compute the cross-entropy of the UNTRAINED model on the
pretraining trials (forward pass only, no weight update). Pretraining subjects' session 1 only; no test data.
Compares with the epoch-1 training loss that the v1.3 run logged, to tell a blow-up at initialization (which
gradient clipping cannot fix) from one that builds up during training.
  usage (from experiments/v1.4/code/):  ../../.venv/bin/python check_rnn_init.py
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "rssns_for_mi_EEG" / "src"))
os.chdir(HERE / "rssns_for_mi_EEG")  # data/ paths are relative to the project folder

import torch
import torch.nn.functional as F

from benchmark_config import EEGNET_CFG, N_CLASSES, N_RECURRENT_INPUTS, RNN_CFG, RSNN_CFG
from download_data import download_data
from folds import build_fold_data, generate_folds
from models import RecurrentELUNet, model_summary
from preprocess_data import preprocess_data

V13 = HERE.parent / "local-train-v1.3/rssns_for_mi_EEG/results/benchmark/20261102_163908/histories/rnn"
processed = preprocess_data(download_data(redo=False))
n_hidden = model_summary(EEGNET_CFG, RSNN_CFG)["rsnn_hidden"]
rows = []
for fold in generate_folds(processed):
    data = build_fold_data(processed, fold, field="X_bands")
    torch.manual_seed(fold.seed)
    model = RecurrentELUNet(N_RECURRENT_INPUTS, n_hidden, N_CLASSES, RNN_CFG)
    model.set_input_stats(data.pre_train.X)
    model.eval()
    with torch.no_grad():
        X, y = torch.from_numpy(data.pre_train.X), torch.from_numpy(data.pre_train.y)
        init_loss = sum(F.cross_entropy(model(X[i:i + 256]), y[i:i + 256], reduction="sum").item()
                        for i in range(0, len(X), 256)) / len(X)
        eig = torch.linalg.eigvals(model.recurrent.weight)
    logged = json.loads((V13 / f"s{fold.target}_r{fold.repeat}.json").read_text())["pretrain"][0]["train_loss"]
    rows.append({"fold": f"s{fold.target}_r{fold.repeat}", "init_loss": init_loss, "v13_epoch1_train_loss": logged,
                 "spectral_radius": eig.abs().max().item(), "max_real_eig": eig.real.max().item()})

rows.sort(key=lambda r: -r["init_loss"])
print(f"{'fold':7} {'loss at init':>13} {'v1.3 epoch 1':>13} {'radius':>7} {'max Re':>7}")
for r in rows[:15]:
    print(f"{r['fold']:7} {r['init_loss']:13.3g} {r['v13_epoch1_train_loss']:13.3g} {r['spectral_radius']:7.3f} {r['max_real_eig']:7.3f}")
print("folds with loss > 5 at init:", sum(r["init_loss"] > 5 for r in rows),
      "| folds with v1.3 epoch-1 loss > 5:", sum(r["v13_epoch1_train_loss"] > 5 for r in rows))
(HERE / f"check_rnn_init_{'cap' if RNN_CFG.recurrent_init_max_spectral_norm else 'nocap'}.json").write_text(json.dumps(rows, indent=1))
