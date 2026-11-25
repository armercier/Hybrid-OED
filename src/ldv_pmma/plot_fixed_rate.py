"""Paper figure F8d (exploratory, post hoc): features found at a fixed false-indication rate.

For every run the detection threshold is set from the run's own map: the 99th (top row) or 95th (bottom row) percentile of
the anomaly a = |h - 10 mm| over the background cells (3 mm map, |u| < 140, |v| < 90 mm, > 20 mm from every feature).
A feature is found when a exceeds that threshold within 8 mm of it. Every map therefore has the same false-indication rate
(1 % or 5 % of the background), and methods are compared on detection only (CFAR-type rule, as in NDT/radar), whatever
the amplitude scale of the inversion. Complements the fixed 3 mm amplitude rule of v3 (F8b, F8c).

Groups, 30 measured points in the final inversion: static layouts; progressive static (3 groups of 10, no design, 33/33/34
inversion); adaptive design as run; adaptive points re-inverted like a static layout; adaptive with the log criterion;
simulation design; moving array (5 rounds of 30, final 30 inverted with the carried model), gradient and random moves.
Right of the divider, when available (proposal A): all positions visited by the moving arrays (~145) inverted jointly, and a
static jittered layout with the same number of points.
Markers: runs (stacked when equal); bar: mean; numbers: runs finding all five.

Usage:  python src/ldv_pmma/plot_fixed_rate.py
Output: reports/2026-11_mssp_paper/figures/F8d_fixed_rate.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import ADAPTIVE, CAT, GRID, INK, MUTED, RANDOM, STATIC, apply
from run_overview import STATIC30
from score_inversions import INV, PREP, thickness

OUT = Path(__file__).resolve().parents[2] / "reports" / "2026-11_mssp_paper" / "figures"
FEAT = ("D1", "D2", "D3", "D4", "M1")
QS = (0.99, 0.95)
J = ("j30a", "j30b", "j30c", "j30d")
Q1S = ("rc30", "j30a", "j30b", "j30c")
MOVE = {4: ("rc30", "j30a", "x48c"), 2: ("rc30", "j30a", "j30b", "x48a", "x48c"), 1: ("rc30", "j30a", "j30b", "x48a", "x48c")}
PFX = {4: ("ad", "rn"), 2: ("a2", "r2"), 1: ("a1", "r1")}


def groups(k):
    g, r = PFX[k]
    ad = {4: ["a30"] + [f"q4a_{s}" for s in J], 2: ["s2a30"] + [f"q2a_{s}" for s in J], 1: [f"q1a_{s}" for s in Q1S]}[k]
    G = [("static", STATIC30[k], STATIC, "s"),
         ("static,\nprogressive", {4: ["n30"] + [f"q4n_{s}" for s in J], 2: ["s2n30"] + [f"q2n_{s}" for s in J],
                                    1: [f"q1n_{s}" for s in Q1S]}[k], MUTED, "v"),
         ("adaptive", ad, ADAPTIVE, "o"),
         ("adaptive,\nre-inverted", [f"ri_{t}" for t in ad], CAT[7], "^"),
         ("adaptive,\nlog crit.", {4: [f"q4l_{s}" for s in J], 2: [f"q2l_{s}" for s in J], 1: [f"q1l_{s}" for s in Q1S]}[k], CAT[5], "P"),
         ("simulation\ndesign", {4: ["d30a", "d30b", "d30c"], 2: ["s2d30"], 1: []}[k], CAT[2], "D"),
         ("moving,\ngradient", [f"{g}_{s}" for s in MOVE[k]], ADAPTIVE, ">"),
         ("moving,\nrandom", [f"{r}_{s}" for s in MOVE[k]], RANDOM, "<"),
         ("all visited,\ngradient", [f"ev_{g}_{s}" for s in MOVE[k]], ADAPTIVE, "h"),
         ("all visited,\nrandom", [f"ev_{r}_{s}" for s in MOVE[k]], RANDOM, "H"),
         ("static,\nsame n", [f"mj{k}_{s}" for s in MOVE[k]], STATIC, "X"),
         ("regular,\n143 points", [f"rg{k}_143"], STATIC, "s")]
    return [(lab, [t for t in tags if (INV / t / "settings.json").exists()], col, mk) for lab, tags, col, mk in G]


def found(tag, geo, O):
    h, Xc, Yc, _ = thickness(tag)
    X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
    a = np.abs(h - 10.0)
    r = {n: np.hypot(X - d["x"] + O[0], Y - d["y"] + O[1]) for n, d in geo.items()}
    bg = (np.abs(X) < 140) & (np.abs(Y) < 90) & (np.min([r[n] for n in FEAT], 0) > 20.0)
    peak = np.array([a[r[n] < 8.0].max() for n in FEAT])
    return [int((peak > np.quantile(a[bg], q)).sum()) for q in QS]


def main():
    apply()
    geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    G = {k: groups(k) for k in (4, 2, 1)}
    labels = [g[0] for g in G[4]]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.6), sharey=True, sharex=True, layout="constrained")
    res = {}
    for c, k in enumerate((4, 2, 1)):
        for gi, (lab, tags, col, mk) in enumerate(G[k]):
            if not tags:
                continue
            F = np.array([found(t, geo, O) for t in tags])
            res[(k, lab)] = dict(zip(tags, F.tolist()))
            for r in range(2):
                ax = axs[r, c]
                v = F[:, r]
                for val in np.unique(v):                       # stack equal values side by side
                    n = int((v == val).sum())
                    x = gi + (np.arange(n) - (n - 1) / 2) * 0.13
                    ax.scatter(x, [val] * n, s=22, marker=mk, color=col, edgecolors="white", linewidths=0.5, zorder=3)
                ax.plot([gi - 0.34, gi + 0.34], [v.mean()] * 2, color=INK, lw=1.0, zorder=4)
                ax.text(gi, 0.55, f"{int((v == 5).sum())}/{len(v)}", ha="center", fontsize=6.5, color=INK)
        axs[0, c].set_title(f"{k} shot{'s' if k > 1 else ''}", fontsize=9, loc="left")
    for r, q in enumerate(QS):
        axs[r, 0].set_ylabel(f"features found (of 5)\nfalse-indication rate {100 * (1 - q):.0f} %")
        for c in range(3):
            ax = axs[r, c]
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            ax.axvline(7.5, color=MUTED, lw=0.6, ls=":")
    axs[0, 0].set_ylim(0.2, 5.5); axs[0, 0].set_yticks(range(1, 6))
    axs[0, 0].text(-0.45, 1.0, "all 5 found:", fontsize=6.5, color=INK)
    axs[0, 0].text(7.6, 5.35, "~145 points", fontsize=6.5, color=MUTED, va="top")
    axs[0, 0].text(7.4, 5.35, "30 points", fontsize=6.5, color=MUTED, va="top", ha="right")
    for c in range(3):
        axs[1, c].set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
        axs[1, c].set_xlim(-0.6, len(labels) - 0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F8d_fixed_rate.{ext}", dpi=200)
    (OUT / "F8d_fixed_rate.json").write_text(json.dumps({f"{k}|{lab}": v for (k, lab), v in res.items()}, indent=1))
    print(OUT / "F8d_fixed_rate.png")


if __name__ == "__main__":
    main()
