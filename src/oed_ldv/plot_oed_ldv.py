"""Figure for the LDV OED runs: mass maps with LDV-point trajectories, metrics per round.

Layout: a grid of mass maps (rows = labs, columns = designs) on top, then one row of
three metric panels (cube localisation error, cube contrast, dense-field prediction error).

Usage:  python src/oed_ldv/plot_oed_ldv.py salvus_fixed salvus_illum mindlin_fixed mindlin_illum
"""

from __future__ import annotations

import json
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from run_oed_ldv import OUT

COL = {"none": "#6b6b6b", "illumination": "#2a6fdb", "waveform": "#c2571a", "illumination_acc": "#1a9850"}
DESIGN_LABEL = {"none": "fixed layout", "illumination": "illumination OED", "waveform": "waveform-misfit OED",
                "illumination_acc": "illumination OED, accumulated (150 points)"}


def dkey(log):
    """Design key; accumulated runs are a separate design (different measurement budget)."""
    return log["args"]["design"] + ("_acc" if log["args"].get("accumulate") else "")
LAB_LABEL = {"salvus": "Salvus 3-D lab", "mindlin": "Mindlin lab", "replay": "Replay lab (real LDV)"}
LAB_STYLE = {"salvus": "-", "mindlin": "--", "replay": "-."}


def main(tags):
    runs = [(t, json.loads((OUT / t / "log.json").read_text()), np.load(OUT / t / "result.npz")) for t in tags]
    labs = list(dict.fromkeys(l["args"]["lab"] for _, l, _ in runs))
    designs = list(dict.fromkeys(dkey(l) for _, l, _ in runs))
    vmax = max(np.exp(r["log_mass"]).max() for _, _, r in runs)

    fig = plt.figure(figsize=(7.2 * len(designs), 2.6 * len(labs) + 3.6), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[2.6 * len(labs), 3.4])
    maps = outer[0].subgridspec(len(labs), len(designs))
    for tag, log, r in runs:
        i, j = labs.index(log["args"]["lab"]), designs.index(dkey(log))
        ax = fig.add_subplot(maps[i, j])
        ext = (r["Xc"].min(), r["Xc"].max(), r["Yc"].min(), r["Yc"].max())
        im = ax.imshow(np.exp(r["log_mass"]).T, origin="lower", extent=ext, cmap="magma", vmin=1, vmax=vmax)
        cube = np.asarray(log["cube_true"])
        ax.add_patch(Rectangle(cube - 0.005, 0.01, 0.01, fill=False, ec="#2a6fdb", lw=1.2))
        tr = r["trajectory"]
        if log["args"].get("accumulate"):  # every measured point, coloured by acquisition round
            for rnd, pts in enumerate(tr[1:]):
                ax.scatter(pts[:, 0], pts[:, 1], s=9, color=plt.cm.YlGn(0.35 + 0.65 * rnd / max(len(tr) - 2, 1)),
                           edgecolors="#1f1f1f", linewidths=0.2)
        else:
            for k in range(tr.shape[1]):
                ax.plot(tr[:, k, 0], tr[:, k, 1], color="#ff9f40", lw=0.7)
            ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=10, c="#ffe14d", edgecolors="#1f1f1f", linewidths=0.3)
        ax.plot(*r["src"], "x", color="#ffffff", ms=7)
        last = log["rounds"][-1]
        ax.set_title(f"{LAB_LABEL[log['args']['lab']]}, {DESIGN_LABEL[dkey(log)]}\n"
                     f"loc. error {last['loc_error_mm']:.0f} mm, contrast {last['contrast']:+.1f}, "
                     f"dense-field error {log['dense_prediction_error']:.3f}", fontsize=8, loc="left")
        ax.set_xlim(ext[0] - 0.003, ext[1] + 0.003), ax.set_ylim(ext[2] - 0.003, ext[3] + 0.006)
        ax.set_facecolor("black")
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    cb = fig.colorbar(im, ax=fig.axes, location="right", shrink=0.7, pad=0.01)
    cb.set_label("recovered mass scale (blue square: true cube, x: transducer)", fontsize=8)

    metrics = outer[1].subgridspec(1, 3)
    for col, (key, lab) in enumerate((("loc_error_mm", "cube localisation error (mm)"), ("contrast", "cube contrast"))):
        ax = fig.add_subplot(metrics[0, col])
        for tag, log, _ in runs:
            ax.plot([e["round"] for e in log["rounds"]], [e[key] for e in log["rounds"]], LAB_STYLE[log["args"]["lab"]],
                    marker="o", ms=4, color=COL[dkey(log)],
                    label=f"{LAB_LABEL[log['args']['lab']]}, {DESIGN_LABEL[dkey(log)]}")
        if key == "contrast":  # common start: plain plate, before any design step or FWI update
            ax.plot([-1], [0.0], "D", color="#1f1f1f", ms=5, zorder=5)
            ax.annotate("start: plain plate\n(same for all runs)", (-1, 0.0), xytext=(4, 6),
                        textcoords="offset points", fontsize=7)
        ax.set_xlabel("after round k (acquire → 10 FWI updates → design)", fontsize=8), ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        n = len(runs[0][1]["rounds"])
        ax.set_xticks(range(-1, n), ["start"] + [str(k) for k in range(n)])
        ax.set_xlim(-1.4, n - 0.6)
        if key == "loc_error_mm":
            ax.set_ylim(0, 60)
            ax.axhline(2.5, color="#bbbbbb", lw=0.8, ls=":")
            ax.text(len(runs[0][1]["rounds"]) - 1, 0.6, "model grid spacing (2.5 mm)", fontsize=7, color="#6b6b6b", ha="right")
        else:
            ax.axhline(0, color="#bbbbbb", lw=0.8)
        if col == 0:
            ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(metrics[0, 2])
    x = np.arange(len(runs))
    ax.bar(x - 0.2, [l["dense_prediction_error_homogeneous"] for _, l, _ in runs], 0.38, color="#d0d0d0",
           label="plain plate (same source filter)")
    ax.bar(x + 0.2, [l["dense_prediction_error"] for _, l, _ in runs], 0.38,
           color=[COL[dkey(l)] for _, l, _ in runs], label="final model")
    ax.set_xticks(x, [f"{l['args']['lab']}\n{DESIGN_LABEL[dkey(l)].split(",")[0].split()[0] + ("\n(150 pts)" if l["args"].get("accumulate") else "")}" for _, l, _ in runs], fontsize=8)
    ax.set_ylabel("dense-field prediction error"), ax.set_ylim(0, 1.15 * max(l["dense_prediction_error_homogeneous"] for _, l, _ in runs)), ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("whole-scan prediction from 30 points (r > 4 cm)", fontsize=8, loc="left")
    fig.savefig(OUT / "oed_ldv_summary.png", dpi=110)
    print(OUT / "oed_ldv_summary.png")


if __name__ == "__main__":
    main(sys.argv[1:])
