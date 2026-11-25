"""Paper figure F3: LDV vs Mindlin vs Salvus 3-D (waveform_check.py), scan file 1 (S1 alone).
Row 1 pristine plate: snapshots data / Mindlin / Salvus and traces at one point.
Row 2 defect plate: snapshots data / Mindlin as built / Salvus as built and traces at two points (behind D3, beside M1).
Misfits (all files, all points) in the titles. Colours: oslo (paper_style).

Usage:  python src/ldv_pmma/plot_waveform_check.py [--t 650]
Output: reports/2026-11_mssp_paper/figures/F3_waveforms.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from paper_style import DATA, GRID, INK, MODEL_A, MODEL_B, MUTED, SEQ_CMAP, apply


def raster(P, v):
    """Scan points on a regular grid -> image (NaN where no data) and its extent."""
    dx = np.median(np.diff(np.unique(np.round(P[:, 0], 1))))
    dy = np.median(np.diff(np.unique(np.round(P[:, 1], 1))))
    i = np.round((P[:, 0] - P[:, 0].min()) / dx).astype(int)
    j = np.round((P[:, 1] - P[:, 1].min()) / dy).astype(int)
    img = np.full((i.max() + 1, j.max() + 1), np.nan)
    img[i, j] = v
    ext = (P[:, 0].min() - dx / 2, P[:, 0].min() + (i.max() + 0.5) * dx, P[:, 1].min() - dy / 2, P[:, 1].min() + (j.max() + 0.5) * dy)
    return img, ext

REPO = Path(__file__).resolve().parents[2]
W = REPO / "ldv_pmma_results" / "waveform_check"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
PREP = REPO / "ldv_work" / "prepared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=650.0, help="snapshot time after the trigger (us)")
    a = ap.parse_args()
    d = np.load(W / "waveform_check.npz")
    M = json.loads((W / "misfits.json").read_text())
    geo = json.loads((PREP / "geometry.json").read_text())
    apply()
    fig = plt.figure(figsize=(10.2, 5.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.42, wspace=0.12, left=0.04, right=0.99, top=0.9, bottom=0.09)
    rows = [("pristine", [("data", "LDV data"), ("mindlin", "Mindlin"), ("salvus", "Salvus 3-D")], "pristine",
             [("centre", (0.0, 0.0))]),
            ("defect", [("data", "LDV data"), ("mindlin_asbuilt", "Mindlin, as built"), ("salvus", "Salvus 3-D, as built")],
             "scatter", [("behind D3", "D3", (0.0, 22.0)), ("beside M1", "M1", (0.0, 20.0))])]
    for r, (scan, cols, gkey, pts) in enumerate(rows):
        R, t = d[f"{scan}_R"], d[f"{scan}_t"] * 1e6
        m = d[f"{scan}_mask_ex1"] > 0
        it = np.argmin(np.abs(t - a.t))
        O = 0.5 * (R.min(0) + R.max(0))
        defs = geo[gkey].get("defects", {}) if scan == "defect" else {}
        vmax = np.percentile(np.abs(d[f"{scan}_data_ex1"][it, m]), 98)
        ms = 7 if scan == "pristine" else 1.2
        for c, (key, lab) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            v = d[f"{scan}_{key}_ex1"][it]
            img, ext = raster(R[m] - O, v[m])
            cmap = SEQ_CMAP.copy(); cmap.set_bad(GRID)
            ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
            for n, g in defs.items():
                x, y = g["x"] - O[0], g["y"] - O[1]
                p = Circle((x, y), g["diameter_mm"] / 2) if n.startswith("D") else Rectangle((x - 5, y - 5), 10, 10)
                p.set(fill=False, ec="#d9534f" if False else INK, lw=0.6); ax.add_patch(p)
            mk = "" if key == "data" else f", misfit {M[f'{scan}/{key}']['total']:.3f}"
            ax.set_title(f"{lab}{mk}", fontsize=8.5, loc="left")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.text(-0.06, 0.5, "pristine plate" if r == 0 else "defect plate", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=9)
                ax.text(0.0, -0.08, f"t = {t[it]:.0f} µs after trigger, source S1", transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
            for lab_i, p in zip("abc", pts):
                ref = (0.0, 0.0) if len(p) == 2 else (defs[p[1]]["x"] - O[0], defs[p[1]]["y"] - O[1])
                q = np.array(ref) + np.array(p[-1])
                j = np.flatnonzero(m)[np.argmin(np.hypot(*(R[m] - O - q).T))]
                ax.plot(R[j, 0] - O[0], R[j, 1] - O[1], marker="D", ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=6)
                ax.text(R[j, 0] - O[0] + 6, R[j, 1] - O[1] + 4, lab_i if len(pts) > 1 else "", fontsize=7.5, color=INK,
                        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.8), zorder=7)
        sub = gs[r, 3].subgridspec(len(pts), 1, hspace=0.15)
        for i, p in enumerate(pts):
            ax = fig.add_subplot(sub[i])
            ref = (0.0, 0.0) if len(p) == 2 else (defs[p[1]]["x"] - O[0], defs[p[1]]["y"] - O[1])
            q = np.array(ref) + np.array(p[-1])
            j = np.flatnonzero(m)[np.argmin(np.hypot(*(R[m] - O - q).T))]
            sc = np.abs(d[f"{scan}_data_ex1"][:, j]).max()
            ax.plot(t, d[f"{scan}_data_ex1"][:, j] / sc, color=DATA, lw=1.4, label="LDV")
            ax.plot(t, d[f"{scan}_{cols[1][0]}_ex1"][:, j] / sc, color=MODEL_A, lw=1.0, ls="--", label="Mindlin")
            ax.plot(t, d[f"{scan}_salvus_ex1"][:, j] / sc, color=MODEL_B, lw=1.0, label="Salvus 3-D")
            ax.axvline(t[it], color=MUTED, lw=0.6, ls=":")
            ax.set_yticks([]); ax.set_xlim(t[0], t[-1])
            ax.text(0.01, 0.95, (f"({'abc'[i]}) " if len(pts) > 1 else "") + p[0], transform=ax.transAxes, fontsize=7.5, va="top")
            if i < len(pts) - 1:
                ax.set_xticklabels([])
            if i == 0 and r == 0:
                ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=1)
        ax.set_xlabel("time after trigger (µs)")
    fig.text(0.04, 0.955, "vertical velocity, scan file 1 (source S1 alone); colour scale symmetric about zero; diamonds: trace positions; outlines: as-built features",
             fontsize=8, color=MUTED)
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F3_waveforms.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / "F3_waveforms.png")


if __name__ == "__main__":
    main()
