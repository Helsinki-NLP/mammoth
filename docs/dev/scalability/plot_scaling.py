#!/usr/bin/env python3
"""
Mammoth scaling efficiency plot.
Usage: python plot_scaling.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

GPUS_PER_NODE = 8

data = {
    1:  16492.82,
    2:  15765.71,
    4:  14876.51,
    8:  12330.42,
    16: 11260.25,
    32: 10189.91,
}

nodes      = np.array(sorted(data.keys()))
per_gpu    = np.array([data[n] for n in nodes])
total      = per_gpu * nodes * GPUS_PER_NODE
baseline   = total[0]
ideal      = baseline * nodes
efficiency = total / ideal * 100  # %

# ── style ──────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
GRAY   = "#94A3B8"
RED    = "#EF4444"
BG     = "#F8FAFC"
GRID   = "#E2E8F0"
TEXT   = "#1E293B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Mammoth Scaling Efficiency  (LUMI MI250X, 8 GPUs/node)",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

# ── left: throughput ───────────────────────────────────────────────────────
total_M = total / 1e6

ax1.plot(nodes, ideal / 1e6, "--", color=GRAY, linewidth=1.5,
         label="Ideal (linear)", zorder=2)
ax1.plot(nodes, total_M, "o-", color=BLUE, linewidth=2.2,
         markersize=7, markerfacecolor="white", markeredgewidth=2,
         label="Actual", zorder=3)

# shade gap
ax1.fill_between(nodes, total_M, ideal / 1e6, alpha=0.08, color=RED)

for n, t in zip(nodes, total_M):
    ax1.annotate(f"{t:.2f}M", (n, t), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9, color=BLUE)

ax1.set_xscale("log", base=2)
ax1.set_xticks(nodes)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax1.set_xlabel("Nodes")
ax1.set_ylabel("Total throughput (M tok/s)")
ax1.set_title("Total throughput vs ideal", fontweight="semibold")
ax1.legend(frameon=False)

# ── right: efficiency % ────────────────────────────────────────────────────
ax2.axhline(100, color=GRAY, linewidth=1.5, linestyle="--", label="100 % (ideal)")
ax2.plot(nodes, efficiency, "o-", color=BLUE, linewidth=2.2,
         markersize=7, markerfacecolor="white", markeredgewidth=2, zorder=3)
ax2.fill_between(nodes, efficiency, 100, alpha=0.08, color=RED)

for n, e in zip(nodes, efficiency):
    ax2.annotate(f"{e:.1f}%", (n, e), textcoords="offset points",
                 xytext=(0, -16), ha="center", fontsize=9, color=BLUE)

ax2.set_xscale("log", base=2)
ax2.set_xticks(nodes)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax2.set_ylim(50, 110)
ax2.set_yticks(range(50, 110, 10))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
ax2.set_xlabel("Nodes")
ax2.set_ylabel("Scaling efficiency")
ax2.set_title("Scaling efficiency  (actual / ideal)", fontweight="semibold")

fig.tight_layout()
out = "scaling_efficiency.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── text summary ───────────────────────────────────────────────────────────
print(f"\n{'Nodes':>6}  {'GPUs':>5}  {'Per-GPU tok/s':>14}  {'Total Mtok/s':>13}  {'Efficiency':>11}")
print("-" * 58)
for n, pg, t, e in zip(nodes, per_gpu, total_M, efficiency):
    print(f"{n:>6}  {n*GPUS_PER_NODE:>5}  {pg:>14.2f}  {t:>13.3f}  {e:>10.1f}%")
