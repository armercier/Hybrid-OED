# OBSOLETE (Nov 2026): early placeholder schematics. Do not run: it would overwrite figures/F1_loop.pdf with a placeholder.
"""Schematic (no-data) figure drafts for the MSSP outline."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def stamp(ax):
    ax.text(0.98, 0.97, "SCHEMATIC - no data", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color="0.5", style="italic")

# F1: acquisition loop
fig, ax = plt.subplots(figsize=(7, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.6, 3.0)
boxes = {"LDV measures\nat current points": (0.2, 1.6), "K FWI updates\n(Mindlin, GPU)": (2.7, 1.6),
         "Design objective\n(misfit or illumination)": (5.2, 1.6), "Gradient step on\nLDV coordinates": (7.7, 1.6)}
for t, (x, y) in boxes.items():
    ax.add_patch(FancyBboxPatch((x, y), 2.1, 1.1, boxstyle="round,pad=0.05", fc="#eef3fb", ec="#2b5c9e"))
    ax.text(x + 1.05, y + 0.55, t, ha="center", va="center", fontsize=8.5)
for x0, x1 in [(2.3, 2.7), (4.8, 5.2), (7.3, 7.7)]:
    ax.add_patch(FancyArrowPatch((x0, 2.15), (x1, 2.15), arrowstyle="-|>", mutation_scale=12))
ax.add_patch(FancyArrowPatch((8.75, 1.55), (1.25, 1.55), connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=12))
ax.text(5, -0.45, "repeat until point budget is spent; output: thickness map + chosen points", ha="center", fontsize=8.5)
stamp(ax); fig.savefig("figures/F1_loop.pdf", bbox_inches="tight")

# F8: error vs number of points
fig, ax = plt.subplots(figsize=(4.2, 3))
n = np.linspace(0.02, 1, 100)
for lab, a, c in [("uniform grid", 0.25, "0.4"), ("random", 0.3, "0.6"), ("adaptive, misfit", 0.08, "C0"), ("adaptive, illumination", 0.06, "C3")]:
    ax.plot(n * 100, 0.15 + np.exp(-n / a), label=lab, color=c)
ax.axhline(0.15, ls="--", color="k", lw=0.8); ax.text(60, 0.19, "full dense scan", fontsize=8)
ax.set_xlabel("LDV points used (% of dense scan)"); ax.set_ylabel("defect reconstruction error")
ax.set_yticks([]); ax.legend(fontsize=7, frameon=False); stamp(ax)
fig.savefig("figures/F8_budget.pdf", bbox_inches="tight")

# F9: point trajectories
fig, ax = plt.subplots(figsize=(4.2, 3)); rng = np.random.default_rng(0)
defects = [(-75, 46), (70, 51), (0, -14), (-85, -54), (80, -50)]
for d in defects: ax.add_patch(plt.Circle(d, 6, fill=False, ec="k"))
for i in range(14):
    p0 = rng.uniform([-140, -90], [140, 90]); tgt = np.array(defects[i % 5]) + rng.normal(0, 25, 2)
    path = np.array([p0 + (tgt - p0) * s for s in np.linspace(0, 0.7, 6)])
    ax.plot(*path.T, "-", color="C0", lw=0.8); ax.plot(*path[0], "o", ms=3, color="0.5"); ax.plot(*path[-1], "o", ms=4, color="C3")
ax.set_xlim(-150, 150); ax.set_ylim(-100, 100); ax.set_aspect("equal")
ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)"); ax.set_title("start (grey) -> final (red) LDV points", fontsize=8); stamp(ax)
fig.savefig("figures/F9_paths.pdf", bbox_inches="tight")
