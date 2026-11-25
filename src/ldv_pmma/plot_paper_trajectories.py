"""Paper figure F9: thickness maps and sensor paths for one start (default j30a, 4 shots): full-scan reference, static start,
random moves (same step lengths) and adaptive design. Diverging colour map centred on the nominal 10 mm (red thinner,
blue thicker); as-built features outlined for reading only. Panel notes give the v3 scores (scores_v3.json).

Usage:  python src/ldv_pmma/plot_paper_trajectories.py [--start j30a] [--prefix ad_ rn_ ''] [--full v3]
Output: reports/2026-11_mssp_paper/figures/F9_paths_<adaptive tag>.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

from score_inversions import INV, PREP, thickness

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
from paper_style import THICK_CMAP, INK, MUTED, apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="j30a")
    ap.add_argument("--prefix", nargs=3, default=["ad_", "rn_", ""], help="adaptive, random, static tag prefixes")
    ap.add_argument("--full", default="v3")
    ap.add_argument("--label", default="4 shots", help="setting named in the panel titles")
    a = ap.parse_args()
    S = json.loads((INV / "scores_v3.json").read_text())
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    D = {n: (d, d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    ad, rn, sta = (p + a.start for p in a.prefix)
    panels = [(a.full, f"(a) full scan, {a.label}"), (sta, "(b) static start"),
              (rn, "(c) random moves, same step lengths"), (ad, "(d) adaptive design")]
    apply()
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.3, wspace=0.06))
    norm = TwoSlopeNorm(vmin=4.0, vcenter=10.0, vmax=16.0)
    for ax, (tag, title) in zip(axs.flat, panels):
        h, Xc, Yc, st = thickness(tag)
        X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
        sp = X[1, 0] - X[0, 0]
        im = ax.imshow(np.clip(h, 4, 16).T, origin="lower", cmap=THICK_CMAP, norm=norm, aspect="equal",
                       extent=(X[0, 0] - sp / 2, X[-1, 0] + sp / 2, Y[0, 0] - sp / 2, Y[0, -1] + sp / 2))
        if "trajectory" in st:
            T = st["trajectory"]
            P = np.stack([np.array(T[0]["start_mm"])] + [np.array(t["designed_mm"]) for t in T[1:]]) - O   # rounds x sensors x 2
            for k in range(P.shape[1]):
                ax.plot(P[:, k, 0], P[:, k, 1], color=INK, lw=0.9, alpha=0.85, zorder=3)
            ax.scatter(P[0, :, 0], P[0, :, 1], s=9, color="white", edgecolors=INK, lw=0.6, zorder=4)
            ax.scatter(P[-1, :, 0], P[-1, :, 1], s=12, color=INK, zorder=4)
        elif tag != a.full:
            sub = np.load(REPO / st["subset"]) if st.get("subset") else None
            keep = sub["keep"] if sub is not None else None
            if keep is not None:
                Q = R[keep] - O
                ax.scatter(Q[:, 0], Q[:, 1], s=12, color=INK, zorder=4)
        for n, (d, x, y) in D.items():
            p = Circle((x, y), d["diameter_mm"] / 2) if n.startswith("D") else Rectangle((x - 5, y - 5), 10, 10)
            p.set(fill=False, ec="#111111", lw=1.0, ls="-", zorder=5); ax.add_patch(p)
            if tag == a.full:
                ax.text(x + 7, y + 7, n, fontsize=8, color=INK, zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))
        s = S[tag]
        ax.set_title(f"{title}\nfound {s['found']}/5, false indications {s['false_alarm']}", fontsize=8.5, loc="left", color=INK)
        ax.set_xlim(R[:, 0].min() - O[0] - 3, R[:, 0].max() - O[0] + 3)
        ax.set_ylim(R[:, 1].min() - O[1] - 3, R[:, 1].max() - O[1] + 3)
        ax.tick_params(labelsize=8, colors=INK); ax.set_xticks([-100, -50, 0, 50, 100])
    for ax in axs[1]:
        ax.set_xlabel("x (mm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, extend="both")
    cb.set_label("thickness h (mm); nominal 10 mm")
    fig.text(0.5, 0.005, "paths: open dot = start, line = moves over 5 rounds, filled dot = last position; outlines = as-built features",
             ha="center", fontsize=7.5, color=MUTED)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F9_paths_{ad}.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / f"F9_paths_{ad}.png")


if __name__ == "__main__":
    main()
