"""Figure: LDV vs Salvus 3-D (no cube / steel cube, elastic / Q = 40). Reads salvus_ldv_compare.npz/json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "ldv_assessment"
MODELS = [("no_cube", "Salvus: plate only"), ("cube_x-20mm", "Salvus: + steel cube"),
          ("cube_x-20mm_Q40", "Salvus: + steel cube, PMMA Q=40")]


def main():
    d = np.load(OUT / "salvus_ldv_compare.npz")
    res = json.loads((OUT / "salvus_ldv_compare.json").read_text())
    P, t, obs, src = d["P"], d["t"], d["obs"], d["src"]
    times = (300e-6, 380e-6, 460e-6)
    vmax = np.percentile(np.abs(obs), 99.5)
    rows = [("LDV v_z (observed)", obs)] + [(lab, d[k]) for k, lab in MODELS]
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    gs = fig.add_gridspec(len(rows) + 1, len(times))
    for r, (lab, a) in enumerate(rows):
        for c, tt in enumerate(times):
            k = int(np.argmin(np.abs(t - tt)))
            ax = fig.add_subplot(gs[r, c])
            ax.scatter(P[:, 0], P[:, 1], c=a[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=1.2, linewidths=0)
            ax.plot(*src, "x", color="#1f1f1f", ms=6)
            if r > 1:
                ax.add_patch(Rectangle((-0.025, -0.010), 0.01, 0.01, fill=False, ec="#1f1f1f", lw=0.8))
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            extra = "" if r == 0 else f"  (misfit behind cube {res[MODELS[r-1][0]]['behind']:.2f})"
            ax.set_title((lab + extra + "\n" if c == 0 else "") + f"t = {tt*1e6:.0f} µs", fontsize=8, loc="left")
    ax = fig.add_subplot(gs[-1, :])
    x = np.arange(0.05, 0.26, 0.02)
    for (k, lab), col in zip(MODELS, ("#6b6b6b", "#c2571a", "#2a6fdb")):
        ax.plot(x * 100, res[k]["axis_ratio"], "o-", color=col, label=lab)
    ax.axhline(1, color="#1f1f1f", lw=0.8, ls="--")
    ax.set_xlabel("distance from source along the axis through the cube (cm)")
    ax.set_ylabel("observed / model RMS")
    ax.legend(fontsize=8, frameon=False), ax.grid(alpha=0.3)
    ax.set_title("amplitude along the source-cube axis (1 = model explains the observed amplitude)", fontsize=9, loc="left")
    fig.savefig(OUT / "salvus_ldv_cube.png", dpi=100)
    print(OUT / "salvus_ldv_cube.png")


if __name__ == "__main__":
    main()
