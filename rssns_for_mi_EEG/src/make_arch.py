"""Architecture diagram of the implemented models: (a) EEGNet ANN / converted SNN, (b) RNN / RSNN.

Usage:  python make_arch.py [output.png]      (default: model_architectures.png next to this script)
Writes one PNG at 300 dpi, sized for the full text width of an IEEE two-column paper (7.16 in).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.6, "pdf.fonttype": 42})

W, H = 7.16, 4.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

C_LAYER = "#d9d9d9"   # trainable linear / convolution layer
C_FIXED = "#ffffff"   # fixed (non-trainable) operation
C_ANN = "#9ecae1"     # conventional (ELU) units
C_SPK = "#fdae6b"     # spiking units
EDGE = "#333333"
SHAPE = "#555555"
BH = 0.215            # box height
STEP = 0.305          # vertical distance between box centres


def box(x, y, w, text, color, h=BH, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=color, ec=EDGE, lw=0.7))
    ax.text(x, y, text, ha="center", va="center", fontweight="bold" if bold else "normal")


def split_box(x, y, w, left, right, h=BH):
    """Left half conventional (blue), right half spiking (orange)."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w / 2, h, fc=C_ANN, ec="none"))
    ax.add_patch(Rectangle((x, y - h / 2), w / 2, h, fc=C_SPK, ec="none"))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="none", ec=EDGE, lw=0.7))
    ax.plot([x, x], [y - h / 2, y + h / 2], color=EDGE, lw=0.5)
    ax.text(x - w / 4, y, left, ha="center", va="center")
    ax.text(x + w / 4, y, right, ha="center", va="center")


def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 color=EDGE, shrinkA=0, shrinkB=0, **kw))


def shape(x, y, text):
    ax.text(x, y, text, ha="left", va="center", color=SHAPE, fontsize=6.2, style="italic")


# ===================================================================== panel (a): EEGNet ANN / converted SNN
xa, wa, xs_a = 1.30, 2.10, 2.42
top = 4.10
ax.text(0.12, 4.42, "(a) EEGNet (ANN) and converted SNN", fontweight="bold", fontsize=7.4, va="center")
ax.text(xs_a, 4.27, "output shape", color=SHAPE, fontsize=6.0, ha="left", va="center")

rows_a = [
    ("box", "Input EEG trial (22 channels, 2 s)", C_FIXED, "22 × 256"),
    ("box", "Temporal conv, 4 filters 1×32  + BN", C_LAYER, "4 × 22 × 256"),
    ("box", "Depthwise spatial conv 22×1, D = 2  + BN", C_LAYER, "8 × 1 × 256"),
    ("split", ("ELU (ANN)", "IF spiking (SNN)"), None, "2048 units"),
    ("box", "Average pool 1×4  (+ dropout, ANN)", C_FIXED, "8 × 1 × 64"),
    ("box", "Separable conv 1×16 + 1×1  + BN", C_LAYER, "8 × 1 × 64"),
    ("split", ("ELU (ANN)", "IF spiking (SNN)"), None, "512 units"),
    ("box", "Average pool 1×8 (+ dropout, ANN), flatten", C_FIXED, "64"),
    ("box", "Dense layer 64 → 4", C_LAYER, "4"),
    ("split", ("Logits", "IF, spike count"), None, "4 classes"),
]
ys_a = [top - i * STEP for i in range(len(rows_a))]
for (kind, label, color, shp), y in zip(rows_a, ys_a):
    if kind == "box":
        box(xa, y, wa, label, color)
    else:
        split_box(xa, y, wa, *label)
    shape(xs_a, y, shp)
for y0, y1 in zip(ys_a[:-1], ys_a[1:]):
    arrow(xa, y0 - BH / 2, xa, y1 + BH / 2)

note_a = ("SNN: same layers and weights as the fine-tuned ANN; BN folded\n"
          "into the convolutions; the whole trial is injected as a constant\n"
          "current at each of 256 steps; hidden IF neurons emit ±1 spikes.\n"
          "No recurrent connections in either model.\n"
          "Parameters: ANN 796, SNN 776.")
ax.text(0.12, ys_a[-1] - 0.25, note_a, ha="left", va="top", fontsize=6.0, color="#222222", linespacing=1.35)

# ===================================================================== panel (b): RSNN / RNN
xb, wb, xs_b = 4.95, 2.05, 6.05
ax.text(3.72, 4.42, "(b) Recurrent networks: RNN and RSNN", fontweight="bold", fontsize=7.4, va="center")
ax.text(xs_b, 4.27, "output shape", color=SHAPE, fontsize=6.0, ha="left", va="center")

y_in = top
y_norm = top - STEP
y_step = top - 2 * STEP
y_proj = top - 3 * STEP
rec_h = 0.98
y_rec = y_proj - BH / 2 - 0.16 - rec_h / 2
y_avg = y_rec - rec_h / 2 - 0.16 - BH / 2
y_read = y_avg - STEP
y_out = y_read - STEP

box(xb, y_in, wb, "Input EEG trial (22 channels, 2 s)", C_FIXED)
shape(xs_b, y_in, "22 × 256")
box(xb, y_norm, wb, "Per-channel standardization (fixed)", C_FIXED)
shape(xs_b, y_norm, "22 × 256")
box(xb, y_step, wb, "One sample per step: $x_t$", C_FIXED)
shape(xs_b, y_step, "22 per step")
box(xb, y_proj, wb, "Input projection $W_{in}$ (linear)", C_LAYER)
shape(xs_b, y_proj, "H")

# recurrent layer
rx0 = xb - wb / 2
ax.add_patch(FancyBboxPatch((rx0, y_rec - rec_h / 2), wb, rec_h, boxstyle="round,pad=0,rounding_size=0.04",
                            fc="#f4f4f4", ec=EDGE, lw=0.9))
ax.text(xb, y_rec + rec_h / 2 - 0.10, "Recurrent layer: H neurons, all-to-all",
        ha="center", va="center", fontsize=6.4)
# small all-to-all graph of neurons (left half blue = ELU in RNN, right half orange = LIF in RSNN)
cx, cy, R, r = xb - 0.45, y_rec - 0.07, 0.25, 0.075
pts = [(cx + R * math.cos(math.pi / 2 + k * 2 * math.pi / 5), cy + R * math.sin(math.pi / 2 + k * 2 * math.pi / 5))
       for k in range(5)]
for i in range(5):
    for j in range(i + 1, 5):
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color="#888888", lw=0.45, zorder=2)
for (px, py) in pts:
    ax.add_patch(Wedge((px, py), r, 90, 270, fc=C_ANN, ec="none", zorder=3))
    ax.add_patch(Wedge((px, py), r, 270, 90, fc=C_SPK, ec="none", zorder=3))
    ax.add_patch(Circle((px, py), r, fc="none", ec=EDGE, lw=0.6, zorder=4))
# self-connection on the top neuron
tx, ty = pts[0]
ax.add_patch(FancyArrowPatch((tx - 0.05, ty + r * 0.8), (tx + 0.05, ty + r * 0.8), connectionstyle="arc3,rad=-1.9",
                             arrowstyle="-|>", mutation_scale=4, lw=0.5, color=EDGE, zorder=5))
ax.text(xb + 0.42, y_rec + 0.10, "RNN: ELU", ha="center", va="center", fontsize=6.0,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_ANN, ec=EDGE, lw=0.5))
ax.text(xb + 0.42, y_rec - 0.10, "RSNN: LIF spiking", ha="center", va="center", fontsize=6.0,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_SPK, ec=EDGE, lw=0.5))
ax.text(xb + 0.42, y_rec - 0.34, "$W_{rec}$ (H × H): state of\nstep $t-1$ fed back", ha="center", va="center", fontsize=5.6,
        color="#333333")

# external recurrence arrow: layer output back to layer input via W_rec
loop_x = rx0 + wb + 0.12
ax.plot([xb + wb / 2, loop_x], [y_rec - 0.30, y_rec - 0.30], color="#c0392b", lw=1.0)
ax.plot([loop_x, loop_x], [y_rec - 0.30, y_rec + 0.30], color="#c0392b", lw=1.0)
arrow(loop_x, y_rec + 0.30, xb + wb / 2, y_rec + 0.30, )
ax.patches[-1].set_color("#c0392b")
ax.text(loop_x + 0.04, y_rec, "$W_{rec}$", color="#c0392b", fontsize=6.4, ha="left", va="center")
shape(xs_b + 0.18, y_rec + 0.30, "H")

# time loop bracket
bx = rx0 - 0.14
ax.plot([bx, bx], [y_rec - rec_h / 2, y_step + BH / 2], color="#2c7fb8", lw=0.9)
ax.plot([bx, bx + 0.06], [y_step + BH / 2, y_step + BH / 2], color="#2c7fb8", lw=0.9)
ax.plot([bx, bx + 0.06], [y_rec - rec_h / 2, y_rec - rec_h / 2], color="#2c7fb8", lw=0.9)
ax.text(bx - 0.05, (y_rec - rec_h / 2 + y_step + BH / 2) / 2, "repeated for t = 1, …, 256",
        rotation=90, ha="right", va="center", color="#2c7fb8", fontsize=6.0)

box(xb, y_avg, wb, "Average over the 256 steps", C_FIXED)
shape(xs_b, y_avg, "H")
box(xb, y_read, wb, "Dropout + linear readout", C_LAYER)
shape(xs_b, y_read, "4")
box(xb, y_out, wb, "Logits", C_FIXED)
shape(xs_b, y_out, "4 classes")

for y0, y1 in ((y_in, y_norm), (y_norm, y_step), (y_step, y_proj), (y_read, y_out), (y_avg, y_read)):
    arrow(xb, y0 - BH / 2, xb, y1 + BH / 2)
arrow(xb, y_proj - BH / 2, xb, y_rec + rec_h / 2)
arrow(xb, y_rec - rec_h / 2, xb, y_avg + BH / 2)

note_b = ("H = 17: RNN-p and RSNN-p (752 parameters each)\n"
          "H = 2560: RSNN-n (6,622,724 parameters)\n"
          "LIF: decay β = 0.9, threshold 1, reset by subtraction")
ax.text(3.72, y_out - 0.25, note_b, ha="left", va="top", fontsize=6.0, color="#222222", linespacing=1.35)

# ===================================================================== legend
ly = 0.13
items = [(C_LAYER, "trainable linear / convolution layer"), (C_FIXED, "fixed operation"),
         (C_ANN, "conventional (ELU) units"), (C_SPK, "spiking units (IF / LIF)")]
for (col, lab), lx in zip(items, (0.25, 2.30, 3.55, 5.20)):
    ax.add_patch(Rectangle((lx, ly - 0.05), 0.18, 0.10, fc=col, ec=EDGE, lw=0.6))
    ax.text(lx + 0.24, ly, lab, va="center", fontsize=6.2)
ax.plot([3.55, 3.55], [0.35, 4.50], color="#bbbbbb", lw=0.6)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "model_architectures.png"
fig.savefig(out, dpi=300)
print(f"Saved {out}")