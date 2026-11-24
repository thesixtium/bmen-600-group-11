"""Fixed spiking reservoir (and its non-spiking twin) for the v2.3 reservoir screen.

Only the readout is trained. The reservoir turns every trial into a fixed feature vector once, so the readout is
trained on cached features with the pipeline's own loop (training.pretrain_then_finetune).

Reservoir dynamics: the same LIF equations as RecurrentLIFNet (models.py, v1.4 = v1.7 code), per time step t:
    I_t = W_in x_t + W_rec s_{t-1}
    u_t = beta * u_{t-1} + I_t - threshold * s_{t-1}          (reset by subtraction)
    s_t = Heaviside(u_t - threshold)
with beta = 0.9 and threshold = 1.0 (RSNNConfig). x_t is the 44-value band input, standardized per channel with the
pretraining subjects' statistics (as RecurrentLIFNet.set_input_stats).

Non-spiking twin (an echo state network with the same weights and the same leak):
    v_t = beta * v_{t-1} + W_in x_t + W_rec r_{t-1},   r_t = tanh(v_t)
Its feature is the time average of r_t^2 (tanh is odd, so the plain average of r_t is close to 0 for band-passed
input; r_t^2 plays the role of the rectified spike rate).

Weights (seeded by the fold seed, never trained):
    W_in  ~ N(0, 1/n_in) * g_in       g_in fitted per fold and neuron kind on the pretraining subjects only (fit_input_gain)
    W_rec ~ N(0, 1/H), then scaled so its spectral radius (largest |eigenvalue|) equals rho; rho = 0 removes it
"""
import numpy as np
import torch
from torch import nn

BETA = 0.9
THRESHOLD = 1.0
TARGET_RATE = 0.1      # median spikes per step over neurons on the pretraining trials (12.8 Hz at 128 Hz)


def draw_weights(n_in: int, n_hidden: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-gain input matrix (H, n_in) and recurrent matrix (H, H) scaled to spectral radius 1."""
    g = torch.Generator().manual_seed(seed)
    w_in = torch.randn(n_hidden, n_in, generator=g) / np.sqrt(n_in)
    w_rec = torch.randn(n_hidden, n_hidden, generator=g) / np.sqrt(n_hidden)
    radius = torch.linalg.eigvals(w_rec.double()).abs().max().item()
    return w_in, w_rec / radius


@torch.no_grad()
def simulate(X: torch.Tensor, w_in: torch.Tensor, w_rec: torch.Tensor, kind: str, n_windows: int = 1,
             batch: int = 256) -> torch.Tensor:
    """Features of every trial. X: (N, C, T) standardized input on the device. Returns (N, H * n_windows).

    kind 'lif': spike count per window divided by the window length (rate in spikes per step).
    kind 'tanh': mean of r_t^2 per window.
    """
    n, _, t_steps = X.shape
    h = w_in.shape[0]
    win = t_steps // n_windows
    out = []
    for i in range(0, n, batch):
        x = X[i:i + batch].permute(2, 0, 1)            # (T, B, C)
        drive = x @ w_in.T                              # (T, B, H): input current, all steps at once
        b = x.shape[1]
        state = x.new_zeros(b, h)                       # membrane u (lif) or v (tanh)
        prev = x.new_zeros(b, h)                        # spikes s_{t-1} (lif) or rates r_{t-1} (tanh)
        acc = x.new_zeros(n_windows, b, h)
        for t in range(t_steps):
            current = drive[t] + prev @ w_rec.T
            if kind == "lif":
                state = BETA * state + current - THRESHOLD * prev
                prev = (state > THRESHOLD).float()     # as snntorch's fast_sigmoid forward: spike when u - threshold > 0
                acc[min(t // win, n_windows - 1)] += prev
            else:
                state = BETA * state + current
                prev = torch.tanh(state)
                acc[min(t // win, n_windows - 1)] += prev ** 2
        out.append((acc / win).permute(1, 0, 2).reshape(b, n_windows * h))
    return torch.cat(out)


@torch.no_grad()
def fit_input_gain(X_pre: torch.Tensor, w_in: torch.Tensor, kind: str = "lif", n_trials: int = 256,
                   seed: int = 0) -> tuple[float, float]:
    """Input gain g_in so the median activity per neuron (no recurrence, whole trial) on a fixed random subset of the
    pretraining trials equals TARGET_RATE: the LIF spike rate (spikes per step), or for the tanh twin the time-mean
    of r_t^2. The twin needs its own gain: with the LIF gain its tanh units saturate (median r^2 about 0.8 on fold 1).
    Log-bisection; uses no labels and no target-subject data."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X_pre), generator=g)[:n_trials].to(X_pre.device)
    sub = X_pre[idx]
    zero = torch.zeros(w_in.shape[0], w_in.shape[0], device=X_pre.device)
    lo, hi = np.log(1e-3), np.log(1e3)
    for _ in range(30):
        mid = (lo + hi) / 2
        activity = simulate(sub, w_in * np.exp(mid), zero, kind).median().item()
        lo, hi = (mid, hi) if activity < TARGET_RATE else (lo, mid)
    gain = float(np.exp((lo + hi) / 2))
    return gain, simulate(sub, w_in * gain, zero, kind).median().item()


class Readout(nn.Module):
    """Dropout + Linear(n_features -> 4) on standardized reservoir features: the only trained part (4 * F + 4)."""

    def __init__(self, n_features: int, n_classes: int = 4) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.25)  # probability is set by the training schedule (set_dropout)
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(f))
