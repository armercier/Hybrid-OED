"""Summary figures for the October 2026 status report (numbers from the runs logged in the repo)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"
C1, C2, C3, C4 = "#2a6fdb", "#c2571a", "#1a9850", "#7b5ea7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                    va="bottom", fontsize=7, color=INK, xytext=(0, 2), textcoords="offset points")


# 1. remat: runtime and memory per method (paper CPU vs GPU + remat), from src/remat/README.md
methods = ["Fixed layout", "Illumination", "Waveform misfit", "Oracle"]
cpu_s, gpu_s = [166.3, 356.5, 503.1, 479.2], [9.8, 37.2, 43.2, 40.5]
cpu_gb, gpu_gb = [1.2, 9.8, 66.1, 64.3], [1.2, 2.4, 5.9, 3.1]
x = np.arange(len(methods))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), layout="constrained")
for ax, a, b, lab, fmt in ((axs[0], cpu_s, gpu_s, "runtime per run (s)", lambda v: f"{v:.0f}"),
                           (axs[1], cpu_gb, gpu_gb, "peak memory (GiB)", lambda v: f"{v:.1f}")):
    b1 = ax.bar(x - 0.2, a, 0.38, color=MUTED, label="paper: CPU, no remat")
    b2 = ax.bar(x + 0.2, b, 0.38, color=C1, label="GPU (RTX 2080 Ti) + remat")
    bar_labels(ax, b1, fmt), bar_labels(ax, b2, fmt)
    ax.set_xticks(x, methods, fontsize=8), ax.set_ylabel(lab), ax.grid(axis="y", color=GRID, lw=0.6)
axs[1].axhline(11, color=C2, lw=0.8, ls="--")
axs[1].text(3.45, 12.5, "GPU memory (11 GB)", color=C2, fontsize=7, ha="right")
axs[0].legend(frameon=False, fontsize=8)
fig.savefig(FIG / "remat.pdf")

# 2. forward-model accuracy vs Salvus 3-D, and cost
rows = [("Plane stress", 0.23, 0.38, 1.4, C1), ("Plane stress + S0 corr.", 0.18, 0.27, 1.4, C1),
        ("JAX 3-D elastic FD", 0.03, 0.10, 50, C3),
        (r"Mindlin, $\kappa=\pi^2/12$", 0.87, 0.99, 9, C2), (r"Mindlin, $\kappa$ matched to A0", 0.20, 0.16, 9, C2)]
fig, axs = plt.subplots(1, 2, figsize=(9, 2.9), layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1]))
y = np.arange(len(rows))[::-1]
axs[0].barh(y + 0.18, [r[1] for r in rows], 0.34, color=[r[4] for r in rows], alpha=0.55, label="early window (t < 300 µs)")
axs[0].barh(y - 0.18, [r[2] for r in rows], 0.34, color=[r[4] for r in rows], label="full record (1 ms)")
for yy, r in zip(y, rows):
    axs[0].text(r[2] + 0.01, yy - 0.18, f"{r[2]:.2f}", va="center", fontsize=7)
    axs[0].text(r[1] + 0.01, yy + 0.18, f"{r[1]:.2f}", va="center", fontsize=7, color=MUTED)
axs[0].set_yticks(y, [r[0] for r in rows]), axs[0].set_xlabel("relative L2 misfit vs Salvus 3-D")
from matplotlib.patches import Patch
axs[0].grid(axis="x", color=GRID, lw=0.6)
axs[0].legend(handles=[Patch(color=MUTED, alpha=0.55, label="early window (t < 300 µs)"),
                       Patch(color=MUTED, label="full record (1 ms)")], frameon=False, fontsize=7, loc="lower right")
axs[0].set_title("in-plane benchmark (S0/SH0) | out-of-plane benchmark (A0)", fontsize=8, loc="left", color=MUTED)
axs[1].barh(y, [r[3] for r in rows], 0.5, color=[r[4] for r in rows])
for yy, r in zip(y, rows):
    axs[1].text(r[3] * 1.1, yy, f"{r[3]:g} s", va="center", fontsize=7)
axs[1].set_xscale("log"), axs[1].set_yticks(y, [""] * len(rows)), axs[1].set_xlabel("seconds per forward solve (GPU)")
axs[1].grid(axis="x", color=GRID, lw=0.6)
fig.savefig(FIG / "forward_models.pdf")

# 3. models vs the real LDV scan (held-out misfit behind / before the cube)
ldv = [("Mindlin plate", 0.550, 0.373, C2), ("Mindlin + cube patch + damping", 0.412, 0.385, C2),
       ("Mindlin + rigid cube 10 mm + damping", 0.399, 0.410, C2),
       ("Mindlin + rigid cube 6 mm eff. + damping", 0.308, 0.327, C2),
       ("Salvus 3-D plate only", 0.557, 0.382, C3), ("Salvus 3-D + steel cube", 0.354, 0.344, C3),
       ("Salvus 3-D + steel cube, Q = 40", 0.330, 0.334, C3)]
fig, ax = plt.subplots(figsize=(9, 3.0), layout="constrained")
y = np.arange(len(ldv))[::-1]
ax.barh(y + 0.18, [r[1] for r in ldv], 0.34, color=[r[3] for r in ldv], label="behind the cube")
ax.barh(y - 0.18, [r[2] for r in ldv], 0.34, color=[r[3] for r in ldv], alpha=0.5, label="before the cube")
for yy, r in zip(y, ldv):
    ax.text(r[1] + 0.005, yy + 0.18, f"{r[1]:.2f}", va="center", fontsize=7)
    ax.text(r[2] + 0.005, yy - 0.18, f"{r[2]:.2f}", va="center", fontsize=7, color=MUTED)
ax.set_yticks(y, [r[0] for r in ldv]), ax.set_xlabel("held-out relative misfit vs LDV out-of-plane velocity")
ax.set_xlim(0, 0.65), ax.grid(axis="x", color=GRID, lw=0.6), ax.legend(frameon=False, fontsize=8)
fig.savefig(FIG / "ldv_models.pdf")

# 4. rigid footprint calibration
rig = json.loads((REPO / "ldv_assessment" / "mindlin_rigid.json").read_text())["sweep"]
sz = np.array([s["size"] for s in rig]) * 1e3
o = np.argsort(sz)
fig, ax = plt.subplots(figsize=(6, 3.1), layout="constrained")
for key, col, lab in (("salvus_Q40", C3, "vs Salvus 3-D (cube, Q = 40)"), ("ldv", C2, "vs LDV data")):
    v = np.array([s[key] for s in rig])
    ax.plot(sz[o], v[o, 0], "o-", color=col, label=lab + ", behind cube")
    ax.plot(sz[o], v[o, 1], "s--", color=col, alpha=0.6, label=lab + ", before cube")
ax.axvline(6, color=MUTED, lw=0.8, ls=":")
ax.text(6.1, 0.39, "chosen on Salvus: 6 mm", fontsize=7, color=MUTED)
ax.axhline(0.33, color=C3, lw=0.6, ls=":")
ax.text(2.0, 0.335, "Salvus 3-D vs LDV", fontsize=7, color=C3)
ax.set_xlabel("effective rigid footprint of the 10 mm cube (mm)"), ax.set_ylabel("held-out misfit")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0.08, 0.45)
fig.savefig(FIG / "rigid_footprint.pdf")

# 5. existing figures (copied; two tall ones cropped)
copies = {"wavefield_traces.png": "elastic_results/wavefields/traces_event_01.png",
          "regularisation_maps.png": "elastic_results/inversion/regularisation_maps.png",
          "diagonal_maps.png": "elastic_results/inversion/diagonal_maps.png",
          "oed_summary.png": "oed_plate_results/oed_summary.png",
          "ldv_components.png": "ldv_assessment/ldv_components.png",
          "mindlin_ldv_amplitude.png": "ldv_assessment/amplitude_maps.png"}
for dst, src in copies.items():
    shutil.copy(REPO / src, FIG / dst)
im = Image.open(REPO / "elastic_results/wavefields/snapshots_event_01.png")
w, h = im.size
im.crop((0, 0, w, int(0.36 * h))).save(FIG / "wavefield_snapshots.png")
im = Image.open(REPO / "ldv_assessment/ldv_salvus_mindlin.png")
w, h = im.size
im.crop((0, int(0.015 * h), w, int(0.375 * h))).save(FIG / "ldv_snapshots.png")
im.crop((0, int(0.635 * h), w, h)).save(FIG / "ldv_traces.png")
print("figures in", FIG)
