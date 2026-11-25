"""Paper figure F11: the two acquisition modes of the laboratory design study, drawn from real runs (4 shots).

(a) Fixed scan budget, adaptive batches (run a30): 30 points in three batches of ten; batch 1 is measured as laid out,
    batches 2 and 3 are moved by the design (arrow: from the planned to the measured position) on the model inverted
    from the batches before. (b) Fixed sensor count, gradient moves (ad_j30a): 30 sensors measured at five successive
    positions; each step follows the illumination gradient on the current model. (c) Same sensors and step lengths,
    random directions (rn_j30a). Grey circles: the five features (for orientation only; no strategy uses them).
Below each panel, the sequence of measurement (M), inversion (I) and design (D) steps; numbers are inversion updates.

Usage:  python src/ldv_pmma/plot_strategies.py
Output: reports/2026-11_mssp_paper/figures/F11_strategies.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from paper_style import ADAPTIVE, CAT, GRID, INK, MUTED, RANDOM, apply

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "ldv_pmma_results" / "inversion"
PREP = REPO / "ldv_work" / "prepared"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
BATCH = [CAT[0], CAT[7], CAT[5]]          # batch 1 near-black, 2 deep blue, 3 mid blue


def traj(tag):
    return json.loads((INV / tag / "settings.json").read_text())["trajectory"]


def frame(ax, R, O, geo):
    lo, hi = R.min(0) - O, R.max(0) - O
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=MUTED, lw=0.8, ls="--"))
    for d in geo.values():
        ax.add_patch(Circle((d["x"] - O[0], d["y"] - O[1]), 6, fc="none", ec=GRID, lw=1.6, zorder=1))
    ax.set_xlim(lo[0] - 8, hi[0] + 8); ax.set_ylim(lo[1] - 8, hi[1] + 8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def timeline(ax, steps):
    """steps: list of (label, kind) with kind in M (measure), I (invert), D (design), R (random step), X (plain text)."""
    col = {"M": CAT[0], "I": "#bdbdbd", "D": ADAPTIVE, "R": RANDOM}
    txt = {"M": "white", "I": INK, "D": "white", "R": "white"}
    x = 0.0
    for lab, k in steps:
        if k == "X":
            ax.text(x + 0.05, 0.5, lab, ha="left", va="center", fontsize=6.5, color=INK)
            x += 1.35
            continue
        w = 0.95
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0,rounding_size=0.2", fc=col[k], ec="none"))
        ax.text(x + w / 2, 0.5, lab, ha="center", va="center", fontsize=6.5, color=txt[k])
        x += w + 0.1
    ax.set_xlim(-0.05, 8.5); ax.set_ylim(-0.1, 1.1); ax.axis("off")


def main():
    apply()
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
    fig = plt.figure(figsize=(8.6, 2.9), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 0.32])
    titles = ["(a) fixed scan budget: adaptive batches", "(b) fixed number of sensors: gradient moves",
              "(c) fixed number of sensors: random moves"]
    # (a) adaptive batches
    ax = fig.add_subplot(gs[0, 0]); frame(ax, R, O, geo)
    for k, t in enumerate(traj("a30")):
        meas = R[t["measured_idx"]] - O
        if k > 0:
            st = np.array(t["start_mm"]) - O
            for p, q in zip(st, meas):
                ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=BATCH[k], lw=0.8, mutation_scale=6,
                                                      shrinkA=0, shrinkB=2.5), zorder=2)
            ax.plot(*st.T, "o", ms=3.2, mfc="white", mec=BATCH[k], mew=0.7, zorder=3)
        ax.plot(*meas.T, "o", ms=4, color=BATCH[k], mec="white", mew=0.4, zorder=4, label=f"batch {k + 1}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2,
              columnspacing=0.8)
    ax.set_title(titles[0], fontsize=8, loc="left")
    timeline(fig.add_subplot(gs[1, 0]), [("M 10", "M"), ("I 33", "I"), ("D", "D"), ("M 10", "M"), ("I 33", "I"),
                                         ("D", "D"), ("M 10", "M"), ("I 34", "I")])
    # (b), (c) moving array
    for c, (tag, col, kind) in enumerate((("ad_j30a", ADAPTIVE, "D"), ("rn_j30a", RANDOM, "R")), start=1):
        ax = fig.add_subplot(gs[0, c]); frame(ax, R, O, geo)
        P = np.stack([R[t["measured_idx"]] - O for t in traj(tag)])      # rounds x sensors x 2
        for s in range(P.shape[1]):
            ax.plot(P[:, s, 0], P[:, s, 1], "-", color=col, lw=0.9, alpha=0.85, zorder=2)
        ax.plot(P[0, :, 0], P[0, :, 1], "o", ms=3.2, mfc="white", mec=col, mew=0.7, zorder=3, label="start")
        ax.plot(P[-1, :, 0], P[-1, :, 1], "o", ms=4, color=col, mec="white", mew=0.4, zorder=4, label="position 5")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=6.5,
                  handletextpad=0.2, columnspacing=0.8)
        ax.set_title(titles[c], fontsize=8, loc="left")
        timeline(fig.add_subplot(gs[1, c]), [("M 30", "M"), ("I 20", "I"), ("then 4 \u00d7", "X"),
                                             ("D" if kind == "D" else "rnd", kind), ("M 30", "M"), ("I 20", "I")])
    ax0 = fig.axes[0]
    ax0.plot([-140, -90], [-112, -112], "-", color=INK, lw=1.2, clip_on=False)
    ax0.text(-115, -110, "50 mm", ha="center", va="bottom", fontsize=6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F11_strategies.{ext}", dpi=250)
    print(OUT / "F11_strategies.png")


if __name__ == "__main__":
    main()
