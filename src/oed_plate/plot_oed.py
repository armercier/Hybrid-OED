"""Figure for the plate OED loop: reconstructions, receiver trajectories, convergence.

Usage:  python src/oed_plate/plot_oed.py fixed_v1 illum_v1 [more run tags]
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
OUT = REPO / "oed_plate_results"
DEFN = json.loads((REPO / "salvus_plate" / "benchmark_definition.json").read_text())
COLOURS = ["#6b6b6b", "#2a6fdb", "#c2571a", "#1a9850"]


def main(tags):
    runs = []
    for tag in tags:
        log = json.loads((OUT / tag / "log.json").read_text())
        res = np.load(OUT / tag / "result.npz")
        runs.append((tag, log, res))
    L, inc = DEFN["L"], DEFN["inclusion"]
    sources = np.array([DEFN["sources"][e]["position"] for e in runs[0][1]["args"]["events"]])
    true_vs = runs[0][2]["true_vs"]

    fig = plt.figure(figsize=(4.0 * (len(runs) + 1), 8.2), layout="constrained")
    gs = fig.add_gridspec(2, len(runs) + 1)
    panels = [("true vs", true_vs, None)] + [(t, r["vs"], (log, r)) for t, log, r in runs]
    for c, (title, vs, extra) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(vs.T, origin="lower", extent=(0, L, 0, L), cmap="viridis", vmin=1050, vmax=1380)
        ax.add_patch(Rectangle((inc["centre"][0] - inc["side"] / 2, inc["centre"][1] - inc["side"] / 2),
                               inc["side"], inc["side"], fill=False, ec="white", lw=0.8, ls="--"))
        ax.scatter(*sources.T, marker="x", c="#d62728", s=40, linewidths=1.5)
        if extra is not None:
            log, r = extra
            traj = r["trajectory"]  # (n_positions, nrec, 2)
            for k in range(traj.shape[1]):
                ax.plot(traj[:, k, 0], traj[:, k, 1], color="#ff9f40", lw=0.8)
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, c="#ffe14d", edgecolors="#1f1f1f", linewidths=0.3)
            f = log["final"]
            title = (f"{title} ({log['args']['design']})\nvs RMSE {f['vs_rmse']:.1f} m/s, "
                     f"inclusion {f['vs_mean_inside']:.0f} m/s")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="vs (m/s)")

    ax = fig.add_subplot(gs[1, : max(1, (len(runs) + 1) // 2)])
    for i, (tag, log, _) in enumerate(runs):
        u = [e["update"] for e in log["fwi"]]
        ax.plot(u, [e["vs_rmse"] for e in log["fwi"]], color=COLOURS[i % 4], lw=1.6, label=f"{tag}: vs RMSE")
    ax.set_xlabel("FWI update"), ax.set_ylabel("vs RMSE (m/s)"), ax.grid(alpha=0.3)
    for b in range(1, runs[0][1]["args"]["rounds"]):
        ax.axvline(b * runs[0][1]["args"]["fwi_per_round"], color="#bbbbbb", lw=0.8, ls=":")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("model error during the loop (dotted: design step + Salvus re-acquisition)", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[1, max(1, (len(runs) + 1) // 2):])
    for i, (tag, log, _) in enumerate(runs):
        u = [e["update"] for e in log["fwi"]]
        ax.plot(u, [e["vs_mean_inside"] for e in log["fwi"]], color=COLOURS[i % 4], lw=1.6, label=tag)
    ax.axhline(true_vs[np.abs(true_vs - DEFN["pmma"]["vs"]) > 1].mean(), color="#1f1f1f", lw=1, ls="--",
               label="true inclusion")
    ax.set_xlabel("FWI update"), ax.set_ylabel("mean vs inside inclusion (m/s)"), ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / "oed_summary.png", dpi=110)
    print(OUT / "oed_summary.png")


if __name__ == "__main__":
    main(sys.argv[1:])
