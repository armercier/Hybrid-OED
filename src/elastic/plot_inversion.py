"""Figure for the plate FWI: true vs recovered vs maps, profiles and convergence.

Usage:  python src/elastic/plot_inversion.py salvus_tmax500us plane_stress_tmax500us
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "elastic_results" / "inversion"
DEFN = json.loads((REPO / "salvus_plate" / "benchmark_definition.json").read_text())
LABELS = {"salvus": "data: Salvus 3-D (model error)", "plane_stress": "data: plane stress (inverse crime)"}
COLOURS = {"salvus": "#2a6fdb", "plane_stress": "#c2571a"}


def main(tags):
    runs = []
    for tag in tags:
        r = np.load(OUT / f"{tag}.npz")
        meta = json.loads((OUT / f"{tag}.json").read_text())
        runs.append((tag, r, meta))
    true_vs, dx = runs[0][1]["true_vs"], float(runs[0][1]["dx"])
    L = DEFN["L"]
    ext = (0, L, 0, L)
    inc = DEFN["inclusion"]
    srcs = [DEFN["sources"][e]["position"] for e in runs[0][2]["args"]["events"]]

    fig = plt.figure(figsize=(4.2 * (len(runs) + 1), 8.4), layout="constrained")
    grid = fig.add_gridspec(2, len(runs) + 1)
    vmin, vmax = 1050, 1380
    maps = [("true vs", true_vs, None)] + [(LABELS[m["args"]["data"]], r["vs"], m) for _, r, m in runs]
    for c, (title, vs, meta) in enumerate(maps):
        ax = fig.add_subplot(grid[0, c])
        im = ax.imshow(vs.T, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((inc["centre"][0] - inc["side"] / 2, inc["centre"][1] - inc["side"] / 2),
                               inc["side"], inc["side"], fill=False, ec="white", lw=1, ls="--"))
        for s in srcs:
            ax.plot(*s, "x", color="#e8e8e8", ms=6)
        sub = title if meta is None else f"{title}\nvs RMSE {meta['history'][-1]['vs_rmse']:.1f} m/s " \
                                         f"(start {meta['history'][0]['vs_rmse']:.1f})"
        ax.set_title(sub, fontsize=9, loc="left")
        ax.set_xlabel("x (m)", fontsize=8)
    fig.colorbar(im, ax=ax, label="vs (m/s)", shrink=0.8)

    ax = fig.add_subplot(grid[1, 0:2])
    x = np.arange(true_vs.shape[0]) * dx
    j = int(round(inc["centre"][1] / dx))
    i = int(round(inc["centre"][0] / dx))
    ax.plot(x, true_vs[:, j], color="#1f1f1f", lw=2, label="true, y = 0.35 m")
    ax.plot(x, true_vs[i, :], color="#1f1f1f", lw=1, ls=":", label="true, x = 0.25 m")
    for _, r, m in runs:
        col = COLOURS[m["args"]["data"]]
        ax.plot(x, r["vs"][:, j], color=col, lw=1.5, label=f"{LABELS[m['args']['data']]}, y = 0.35 m")
        ax.plot(x, r["vs"][i, :], color=col, lw=1, ls=":", label=f"{LABELS[m['args']['data']]}, x = 0.25 m")
    ax.set_xlabel("position along profile (m)"), ax.set_ylabel("vs (m/s)")
    ax.legend(fontsize=7, frameon=False), ax.grid(alpha=0.3)
    ax.set_title("profiles through the inclusion centre", fontsize=9, loc="left")

    ax = fig.add_subplot(grid[1, 2:])
    for _, r, m in runs:
        h = m["history"]
        col = COLOURS[m["args"]["data"]]
        it = [k["iteration"] for k in h]
        ax.plot(it, [k["loss"] / h[0]["loss"] for k in h], color=col, lw=1.5, label=f"data misfit, {m['args']['data']}")
        ax.plot(it, [k["vs_rmse"] / h[0]["vs_rmse"] for k in h], color=col, lw=1.5, ls="--",
                label=f"vs RMSE, {m['args']['data']}")
    ax.set_xlabel("iteration"), ax.set_ylabel("relative to start"), ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False), ax.grid(alpha=0.3, which="both")
    ax.set_title("convergence", fontsize=9, loc="left")
    fig.savefig(OUT / "inversion_summary.png", dpi=110)
    print(OUT / "inversion_summary.png")


if __name__ == "__main__":
    main(sys.argv[1:])
