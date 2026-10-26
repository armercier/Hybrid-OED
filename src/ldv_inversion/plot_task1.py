"""Figure for task 1: cube search landscape (stage A) and blind map FWI (stage B).

Usage:  python src/ldv_inversion/plot_task1.py [maps tag, default maps_v1]
"""

from __future__ import annotations

import json
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ldv_problem import OUT


def main(tag):
    cs = json.loads((OUT / "cube_search.json").read_text())
    mp = np.load(OUT / f"{tag}.npz")
    hist = json.loads((OUT / f"{tag}.json").read_text())["history"]
    best = cs["best"]
    sal = np.asarray(cs["salvus_position"])
    P, src = mp["P"], mp["src"]

    fig = plt.figure(figsize=(13, 9), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1])

    # A: misfit landscape
    ax = fig.add_subplot(gs[0, 0])
    coarse = [r for r in cs["results"] if r["stage"] == "coarse"]
    xs = sorted({round(r["x"] * 1e3) for r in coarse})
    ys = sorted({round(r["y"] * 1e3) for r in coarse})
    Z = np.array([[next(r["total"] for r in coarse if round(r["x"] * 1e3) == x and round(r["y"] * 1e3) == y)
                   for x in xs] for y in ys])
    im = ax.imshow(Z, origin="lower", extent=(xs[0] - 1, xs[-1] + 1, ys[0] - 1, ys[-1] + 1), cmap="viridis_r")
    ax.plot(best["x"] * 1e3, best["y"] * 1e3, "*", color="#ffffff", ms=12, mec="#1f1f1f", label="best (1 mm search)")
    ax.plot(*(sal * 1e3), "s", mfc="none", mec="#d62728", ms=9, label="Salvus 3-D fit")
    ax.set_xlabel("cube x (mm, scan frame)"), ax.set_ylabel("cube y (mm)")
    ax.set_title(f"A: rigid-cube position search\nheld-out misfit (behind + before); plate only = "
                 f"{cs['homogeneous']['behind'] + cs['homogeneous']['before']:.2f}", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # B: convergence
    ax = fig.add_subplot(gs[0, 1:])
    ev = [h for h in hist if "test_behind" in h]
    it = [h["iteration"] for h in ev]
    ax.plot(it, [h["test_behind"] for h in ev], "o-", color="#2a6fdb", label="map FWI, behind cube")
    ax.plot(it, [h["test_before"] for h in ev], "s--", color="#2a6fdb", alpha=0.6, label="map FWI, before cube")
    for v, lab, col in ((cs["homogeneous"]["behind"], "plain plate, behind", "#6b6b6b"),
                        (best["behind"], "rigid cube at stage-A position, behind", "#1a9850")):
        ax.axhline(v, color=col, lw=1, ls=":")
        ax.text(it[-1], v, " " + lab, fontsize=7, va="bottom", ha="right", color=col)
    ax.set_xlabel("FWI iteration"), ax.set_ylabel("held-out misfit"), ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("B: blind FWI of mass / rotary-inertia maps (start: plain plate, no cube prior)", fontsize=8, loc="left")

    # B: maps over the scan
    ext = (mp["Xc"].min(), mp["Xc"].max(), mp["Yc"].min(), mp["Yc"].max())
    for row, (key, lab) in enumerate((("log_inertia", "rotary-inertia scale"), ("log_mass", "mass scale"))):
        ax = fig.add_subplot(gs[1 + row, :])
        v = np.exp(mp[key])
        im = ax.imshow(v.T, origin="lower", extent=ext, cmap="magma", vmin=1.0, vmax=max(v.max(), 1.01))
        for c, mk, col, lab2 in ((sal, "s", "#d62728", "Salvus 3-D fit"), (np.array([best["x"], best["y"]]), "*",
                                                                          "#2a6fdb", "stage A")):
            ax.add_patch(Rectangle(c - 0.005, 0.01, 0.01, fill=False, ec=col, lw=1, label=f"cube, {lab2}"))
        ax.plot(*src, "x", color="#ffffff", ms=7)
        ax.set_aspect("equal"), ax.set_xlabel("x (m, scan frame)"), ax.set_ylabel("y (m)")
        ax.set_title(f"recovered {lab} (x = transducer)", fontsize=8, loc="left")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "task1_summary.png", dpi=110)
    print(OUT / "task1_summary.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "maps_v1")
