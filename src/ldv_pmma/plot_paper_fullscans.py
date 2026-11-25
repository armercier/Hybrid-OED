"""Paper figure F7: blind full-scan thickness maps for 4, 2 and 1 shots (v3, s2full, k1full), same colour scale as F9,
with the v3 scoring geometry: as-built features (solid), the 8 mm detection radius (dotted) and the 20 mm zone outside which
an anomaly counts as a false indication (dashed). Panel notes give found / false indications / background RMS (scores_v3.json).

Usage:  python src/ldv_pmma/plot_paper_fullscans.py
Output: reports/2026-11_mssp_paper/figures/F7_fullscans.{pdf,png}
"""
from __future__ import annotations

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
RUNS = [("v3", "(a) 4 shots, 6418 points"), ("s2full", "(b) 2 shots"), ("k1full", "(c) 1 shot")]


def main():
    S = json.loads((INV / "scores_v3.json").read_text())
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    D = {n: (d, d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    apply()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    norm = TwoSlopeNorm(vmin=4.0, vcenter=10.0, vmax=16.0)
    for ax, (tag, title) in zip(axs, RUNS):
        h, Xc, Yc, st = thickness(tag)
        X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
        sp = X[1, 0] - X[0, 0]
        im = ax.imshow(np.clip(h, 4, 16).T, origin="lower", cmap=THICK_CMAP, norm=norm, aspect="equal",
                       extent=(X[0, 0] - sp / 2, X[-1, 0] + sp / 2, Y[0, 0] - sp / 2, Y[0, -1] + sp / 2))
        for n, (d, x, y) in D.items():
            p = Circle((x, y), d["diameter_mm"] / 2) if n.startswith("D") else Rectangle((x - 5, y - 5), 10, 10)
            p.set(fill=False, ec="#111111", lw=1.0, zorder=5); ax.add_patch(p)
            ax.add_patch(Circle((x, y), 8, fill=False, ec=INK, lw=0.6, ls=":", zorder=5))
            ax.add_patch(Circle((x, y), 20, fill=False, ec=MUTED, lw=0.7, ls="--", zorder=5))
            if tag == "v3":
                ax.text(x + 14, y + 14, n, fontsize=8, color=INK, zorder=6, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))
        s = S[tag]
        ax.set_title(f"{title}\nfound {s['found']}/5, false indications {s['false_alarm']}\nbackground RMS {s['bg_rms']:.2f} mm",
                     fontsize=8.5, loc="left", color=INK)
        ax.set_xlim(R[:, 0].min() - O[0] - 3, R[:, 0].max() - O[0] + 3)
        ax.set_ylim(R[:, 1].min() - O[1] - 3, R[:, 1].max() - O[1] + 3)
        ax.set_xticks([-100, -50, 0, 50, 100]); ax.tick_params(labelsize=8, colors=INK)
        ax.set_xlabel("x (mm)")
    axs[0].set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01, extend="both")
    cb.set_label("thickness h (mm)")
    fig.text(0.45, -0.02, "solid: as-built features; dotted: 8 mm detection radius; dashed: 20 mm zone (anomalies outside count as false indications)",
             ha="center", fontsize=7.5, color=MUTED)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F7_fullscans.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / "F7_fullscans.png")


if __name__ == "__main__":
    main()
