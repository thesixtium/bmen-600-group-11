"""ANN -> feedforward SNN conversion of a trained EEGNet.

Method: rate-coded conversion with data-based weight normalization, after Rueckauer et al. 2017
(Front. Neurosci. 11:682, https://doi.org/10.3389/fnins.2017.00682). Own re-implementation in plain PyTorch (no
conversion library is installed in the project); nothing is adapted from external source files.

Steps
  1. Fold every BatchNorm into the preceding convolution (conv weights x gamma/sqrt(var+eps), plus a bias).
  2. Data-based normalization: for each spiking layer l, lambda_l = percentile (default 99.9) of |activation| of the
     ANN on calibration trials. Weights of the first op of layer l are scaled by lambda_{l-1}/lambda_l and all biases
     by 1/lambda_l, so firing rates stay in [0, 1] spikes/step. Calibration uses ONLY the target subject's
     fine-tuning training trials (never the test session).
  3. Spiking neurons: integrate-and-fire with reset by subtraction (threshold 1). The analog trial, divided by
     lambda_0, is injected as a constant input current at every step (direct encoding). AvgPool and Flatten act on spikes.
  4. ELU: ELU has a negative branch, which plain IF neurons cannot represent. Hidden neurons therefore emit signed
     (+1 / -1) spikes and negative spikes are rate-limited to 1/lambda_l per step (ELU saturates at -1), i.e. the
     converted activation is approx. max(x, -1) on the negative side instead of exp(x)-1.
  5. Output layer: IF neurons without a floor; prediction = class with the highest output spike count; ties are
     broken by the final membrane potential (configurable).

Limitations: rate coding needs many steps (latency); the negative ELU branch is only approximated; signed spikes
need ternary events; the 99.9th-percentile scaling clips rare large activations; conversion cannot recover
information lost in the ANN.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from benchmark_config import SNNConfig
from models import EEGNet


def _fold_batchnorm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a copy of conv (which has no bias) with the eval-mode BatchNorm folded in."""
    folded = copy.deepcopy(conv)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)  # (out_channels,)
    folded.weight = nn.Parameter((conv.weight * scale.view(-1, 1, 1, 1)).detach(), requires_grad=False)
    bias = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        bias = bias + conv.bias * scale
    folded.bias = nn.Parameter(bias.detach(), requires_grad=False)
    return folded


def _percentile(t: torch.Tensor, q: float) -> float:
    return max(float(np.percentile(np.abs(t.detach().cpu().numpy()), q)), 1e-6)


class _IFNeurons:
    """Integrate-and-fire population, reset by subtraction. State is created fresh for every simulation."""

    def __init__(self, threshold: float, signed: bool = False, neg_rate: float = 1.0, v_floor: float = 1.0) -> None:
        self.threshold, self.signed, self.neg_rate, self.v_floor = threshold, signed, neg_rate, v_floor
        self.v: torch.Tensor | None = None
        self.net: torch.Tensor | None = None  # signed spike count so far
        self.t = 0

    def step(self, current: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(current)
            self.net = torch.zeros_like(current)
        th = self.threshold
        self.v = self.v + current
        pos = (self.v >= th).to(current.dtype)
        self.v = self.v - pos * th
        spikes = pos
        if self.signed:
            # negative spike only if the running negative rate stays above -neg_rate (ELU's lower bound)
            allowed = (self.net - 1.0) >= -self.neg_rate * (self.t + 1)
            neg = ((self.v <= -th) & allowed).to(current.dtype)
            self.v = torch.clamp(self.v + neg * th, min=-th * self.v_floor)
            spikes = pos - neg
        self.net = self.net + spikes
        self.t += 1
        return spikes


class ConvertedEEGNetSNN(nn.Module):
    """Feedforward spiking version of a trained (and fine-tuned) EEGNet. Not trained; weights come from the ANN."""

    def __init__(self, ann: EEGNet, calibration_x: torch.Tensor, cfg: SNNConfig = SNNConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        ann = copy.deepcopy(ann).eval()
        self.conv1 = _fold_batchnorm(ann.conv1, ann.bn1)
        self.depthwise = _fold_batchnorm(ann.depthwise, ann.bn2)
        self.sep_depth = copy.deepcopy(ann.sep_depth)
        self.sep_point = _fold_batchnorm(ann.sep_point, ann.bn3)
        self.fc = copy.deepcopy(ann.fc)
        self.pool1, self.pool2 = ann.pool1, ann.pool2
        for p in self.parameters():
            p.requires_grad_(False)

        # --- data-based normalization on calibration trials (unscaled folded ANN) ---
        with torch.no_grad():
            x = calibration_x.to(next(ann.parameters()).device)
            a1 = F.elu(self.depthwise(self.conv1(x.unsqueeze(1))))
            a2 = F.elu(self.sep_point(self.sep_depth(self.pool1(a1))))
            out = self.fc(self.pool2(a2).flatten(1))
            q = cfg.calibration_percentile
            lam0, lam1, lam2, lam3 = _percentile(x, q), _percentile(a1, q), _percentile(a2, q), _percentile(out, q)
            self.lambdas = (lam0, lam1, lam2, lam3)

            # layer 1 = [conv1, depthwise], layer 2 = [sep_depth, sep_point], layer 3 = [fc]
            self.conv1.weight.mul_(lam0 / lam1)
            self.conv1.bias.div_(lam1)
            self.depthwise.bias.div_(lam1)
            self.sep_depth.weight.mul_(lam1 / lam2)
            self.sep_point.bias.div_(lam2)
            self.fc.weight.mul_(lam2 / lam3)
            self.fc.bias.div_(lam3)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())  # includes the folded biases; should equal 776

    @torch.no_grad()
    def simulate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 22, 256). Returns (output spike counts, final output membrane potentials), each (B, n_classes)."""
        cfg = self.cfg
        lam0, lam1, lam2, _ = self.lambdas
        n1 = _IFNeurons(cfg.threshold, cfg.signed_spikes, min(1.0, 1.0 / lam1), cfg.negative_floor)
        n2 = _IFNeurons(cfg.threshold, cfg.signed_spikes, min(1.0, 1.0 / lam2), cfg.negative_floor)
        n3 = _IFNeurons(cfg.threshold, signed=False)
        # constant input current -> the (linear) first layer gives the same current at every step: compute once
        current1 = self.depthwise(self.conv1((x / lam0).unsqueeze(1)))
        for _ in range(cfg.n_steps):
            s1 = n1.step(current1)
            s2 = n2.step(self.sep_point(self.sep_depth(self.pool1(s1))))
            n3.step(self.fc(self.pool2(s2).flatten(1)))
        return n3.net, n3.v

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted class, output spike counts). Prediction = highest spike count."""
        counts, potential = self.simulate(x)
        if self.cfg.tie_break_membrane:
            tied_best = counts == counts.max(dim=1, keepdim=True).values
            potential = torch.where(tied_best, potential, torch.full_like(potential, float("-inf")))
            return potential.argmax(dim=1), counts
        return counts.argmax(dim=1), counts
