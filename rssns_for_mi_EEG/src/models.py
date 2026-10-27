"""EEGNet ANN, fully recurrent LIF network (RSNN) and analytic parameter counts.

The converted SNN lives in ann_to_snn.py.
"""

import math

import torch
from numpy.typing import NDArray
from snntorch import surrogate  # only the surrogate gradient is used; the LIF loop below is written out explicitly
from torch import nn

from benchmark_config import EEGNetConfig, RSNNConfig


# ---------------------------------------------------------------- EEGNet


class EEGNet(nn.Module):
    """EEGNet-4,2 with ELU, following Lawhern et al. 2018 (J. Neural Eng. 15, 056013), Table 2.

    Block 1: temporal conv (F1 x (1, 32), 'same') -> BN -> depthwise spatial conv (D*F1, (22, 1), max-norm 1)
             -> BN -> ELU -> AvgPool(1, 4) -> Dropout
    Block 2: separable conv ((1, 16) depthwise + 1x1 pointwise) -> BN -> ELU -> AvgPool(1, 8) -> Dropout
    Classifier: Flatten -> Dense(4, max-norm 0.25). Returns logits (softmax lives inside the cross-entropy loss).
    Input: (batch, 22, 256).
    """

    def __init__(self, cfg: EEGNetConfig = EEGNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.F1 * cfg.D
        bn = {"eps": cfg.bn_eps, "momentum": cfg.bn_momentum}
        self.conv1 = nn.Conv2d(1, cfg.F1, (1, cfg.kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(cfg.F1, **bn)
        self.depthwise = nn.Conv2d(cfg.F1, d, (cfg.n_channels, 1), groups=cfg.F1, bias=False)
        self.bn2 = nn.BatchNorm2d(d, **bn)
        self.pool1 = nn.AvgPool2d((1, cfg.pool1))
        self.drop1 = nn.Dropout(0.25)
        self.sep_depth = nn.Conv2d(d, d, (1, cfg.separable_kernel), padding="same", groups=d, bias=False)
        self.sep_point = nn.Conv2d(d, cfg.F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg.F2, **bn)
        self.pool2 = nn.AvgPool2d((1, cfg.pool2))
        self.drop2 = nn.Dropout(0.25)
        self.fc = nn.Linear(cfg.F2 * (cfg.n_samples // (cfg.pool1 * cfg.pool2)), cfg.n_classes)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, 22, 256)
        x = self.drop1(self.pool1(self.act(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(self.act(self.bn3(self.sep_point(self.sep_depth(x))))))
        return self.fc(x.flatten(1))

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Keras MaxNorm constraints; called by the training loop after every optimizer step."""
        self.depthwise.weight.renorm_(2, 0, self.cfg.depthwise_max_norm)  # per output filter
        self.fc.weight.renorm_(2, 0, self.cfg.dense_max_norm)  # per output unit


# ---------------------------------------------------------------- Fully recurrent LIF network


class RecurrentLIFNet(nn.Module):
    """(input standardization) -> input projection -> ONE dense all-to-all recurrent LIF layer -> linear readout.

    Weights (all trained by surrogate-gradient BPTT):
      input_proj : Linear(n_inputs -> n_hidden), applied to the 22 channel values of one EEG sample per step
      recurrent  : Linear(n_hidden -> n_hidden, no bias), dense H x H; entry [i, j] is the trainable weight of the
                   directed connection from hidden neuron j to hidden neuron i (diagonal = self-connections included)
      readout    : Linear(n_hidden -> n_classes) on the time-averaged hidden spike rate

    Input standardization: per-channel (x - ch_mean) / ch_std with FIXED buffers (not trained, not part of the
    parameter count). They are set once by set_input_stats() from the pretraining subjects' session 1 and stay fixed
    through fine-tuning and testing, so no target-subject or test information enters them. Until set_input_stats() is
    called they are the identity (mean 0, std 1), i.e. the model sees the raw input.

    Dynamics for t = 1..T (reset by subtraction, previous spike used for both feedback and reset):
      I_t   = input_proj(x_t) + recurrent(s_{t-1})
      u_t   = beta * u_{t-1} + I_t - threshold * s_{t-1}.detach()
      s_t   = Heaviside(u_t - threshold)     (backward: snntorch fast-sigmoid surrogate)
    u_0 = 0 and s_0 = 0 are re-created inside every forward() call, so no state leaks between trials, batches,
    folds, subjects or sessions. Hidden-to-hidden edges form directed cycles, so the layer is truly recurrent.
    """

    def __init__(self, n_inputs: int, n_hidden: int, n_classes: int, cfg: RSNNConfig = RSNNConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_hidden = n_hidden
        self.input_proj = nn.Linear(n_inputs, n_hidden)
        self.recurrent = nn.Linear(n_hidden, n_hidden, bias=False)
        self.readout = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(0.25)  # on the readout input; probability is set by the training schedule
        self.spike_fn = surrogate.fast_sigmoid(slope=cfg.surrogate_slope)
        nn.init.normal_(self.recurrent.weight, mean=0.0, std=cfg.recurrent_init_gain / math.sqrt(n_hidden))
        # fixed (non-trainable) input standardization, saved in the state_dict with the weights
        self.register_buffer("ch_mean", torch.zeros(1, n_inputs, 1))
        self.register_buffer("ch_std", torch.ones(1, n_inputs, 1))

    @property
    def recurrent_weight(self) -> torch.Tensor:
        return self.recurrent.weight  # (H, H), trainable

    @torch.no_grad()
    def set_input_stats(self, X: NDArray) -> None:
        """Set the per-channel mean/std from X of shape (n_trials, n_channels, n_samples).

        Call with the PRETRAINING subjects' session-1 trials only. Statistics are taken over trials and time.
        """
        X_t = torch.as_tensor(X, dtype=torch.float32)
        self.ch_mean.copy_(X_t.mean(dim=(0, 2)).view(1, -1, 1))
        self.ch_std.copy_(X_t.std(dim=(0, 2)).clamp_min(1e-6).view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.ch_mean) / self.ch_std  # (B, C, T) standardized per channel
        x = x.permute(2, 0, 1)  # (T, B, C): one EEG sample (22 values) per simulation step
        n_steps = min(self.cfg.n_steps, x.shape[0])
        batch = x.shape[1]
        input_current = self.input_proj(x)  # (T, B, H): feedforward part of I_t, identical to computing it per step
        mem = x.new_zeros(batch, self.n_hidden)
        spk = x.new_zeros(batch, self.n_hidden)
        spike_count = x.new_zeros(batch, self.n_hidden)
        for t in range(n_steps):
            current = input_current[t] + self.recurrent(spk)  # external input + recurrent spikes of step t-1
            mem = self.cfg.beta * mem + current - self.cfg.threshold * spk.detach()
            spk = self.spike_fn(mem - self.cfg.threshold)
            spike_count = spike_count + spk
        return self.readout(self.dropout(spike_count / n_steps))


# ---------------------------------------------------------------- Parameter counts (analytic)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eegnet_param_count(c: EEGNetConfig = EEGNetConfig()) -> int:
    """Trainable parameters of the ANN, BatchNorm gamma/beta included (796 for EEGNet-4,2 on 22 x 256)."""
    d = c.F1 * c.D
    flat = c.F2 * (c.n_samples // (c.pool1 * c.pool2))
    return (
        c.F1 * c.kernel_length + 2 * c.F1  # temporal conv + BN1
        + d * c.n_channels + 2 * d  # depthwise conv + BN2
        + d * c.separable_kernel + d * c.F2 + 2 * c.F2  # separable conv (depthwise + pointwise) + BN3
        + flat * c.n_classes + c.n_classes  # dense
    )


def converted_snn_param_count(c: EEGNetConfig = EEGNetConfig()) -> int:
    """Weights + biases of the converted SNN: BatchNorm is folded into the preceding conv, which therefore gains one
    bias per output channel (776 for EEGNet-4,2). Matches the sum of numel() of the folded layers in ann_to_snn.py."""
    d = c.F1 * c.D
    flat = c.F2 * (c.n_samples // (c.pool1 * c.pool2))
    return (
        c.F1 * c.kernel_length + c.F1
        + d * c.n_channels + d
        + d * c.separable_kernel + d * c.F2 + c.F2
        + flat * c.n_classes + c.n_classes
    )


def rsnn_param_count(n_inputs: int, n_hidden: int, n_classes: int) -> int:
    """input_proj (W + b) + dense recurrent H x H (no bias) + readout (W + b). The fixed input standardization
    buffers are not trainable and not counted."""
    return (n_inputs * n_hidden + n_hidden) + n_hidden * n_hidden + (n_hidden * n_classes + n_classes)


def choose_hidden_size(target_params: int, n_inputs: int, n_classes: int, max_hidden: int = 2000) -> int:
    """Integer hidden size whose RSNN parameter count is closest to target_params (smaller size wins ties)."""
    return min(range(1, max_hidden + 1), key=lambda h: abs(rsnn_param_count(n_inputs, h, n_classes) - target_params))


def model_summary(eegnet_cfg: EEGNetConfig, rsnn_cfg: RSNNConfig) -> dict:
    snn_params = converted_snn_param_count(eegnet_cfg)
    hidden = rsnn_cfg.n_hidden or choose_hidden_size(snn_params, eegnet_cfg.n_channels, eegnet_cfg.n_classes)
    rsnn_params = rsnn_param_count(eegnet_cfg.n_channels, hidden, eegnet_cfg.n_classes)
    return {
        "ann_params": eegnet_param_count(eegnet_cfg),
        "converted_snn_params": snn_params,
        "rsnn_hidden": hidden,
        "rsnn_params": rsnn_params,
        "rsnn_minus_snn_params": rsnn_params - snn_params,
        "rsnn_relative_difference": (rsnn_params - snn_params) / snn_params,
    }