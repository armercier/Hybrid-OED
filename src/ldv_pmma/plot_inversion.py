"""Clear figure of a defect-scan inversion result (CPU only, numpy): maps with the as-built defects
outlined and labelled, colour ranges set to the inverted values, and zooms on each defect.

Usage:  python src/ldv_pmma/plot_inversion.py [--tag v1]
Output: ldv_pmma_results/inversion/<tag>/maps_clear.png
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

REPO = Path(__file__).resolve().parents[2]
PREP = REPO / "ldv_work" / "prepared"
EDGE = {"D1": "#00e5ff", "D2": "#00e5ff", "D3": "#ff2bd6", "D4": "#00e5ff", "M1": "#7CFC00"}


def outline(ax, d, n, x, y, lw=1.6, label=True):
    if n.startswith("D"):
        ax.add_patch(Circle((x, y), d["diameter_mm"] / 2, fill=False, ec=EDGE[n], lw=lw))
    else:
        a = d["size_mm"]
        ax.add_patch(Rectangle((x - a / 2, y - a / 2), a, a, fill=False, ec=EDGE[n], lw=lw))
    if label:
        ax.text(x + 8, y + 8, n, color=EDGE[n], fontsize=10, fontweight="bold",
                bbox=dict(fc="k", ec="none", alpha=0.55, pad=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    out = REPO / "ldv_pmma_results" / "inversion" / a.tag
    r = np.load(out / "result.npz")
    st = json.loads((out / "settings.json").read_text())
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))                    # mm, scatter grid centre
    H, hmin = 10.0, st["hmin"]
    m = r["m"]
    thickness_only = st.get("param") == "log_thickness"
    X, Y = r["Xc"] * 1e3 - O[0], r["Yc"] * 1e3 - O[1]
    inner = (np.abs(X) < 140) & (np.abs(Y) < 90)          # colour range without the grid-edge artefact
    if thickness_only:                                     # invert_thickness.py: h = h_min + (H - h_min) exp(m)
        h = hmin + (H - hmin) * np.exp(m[0])
        mass = h                                           # second row: the same thickness on the full 0-10 mm scale
        m_lo, m_hi = 0.0, 10.5
        labels = ("inverted thickness (mm), fine scale", "inverted thickness (mm), full 0-10 mm scale")
        cmaps = ("viridis_r", "viridis_r")
    else:                                                  # invert_defect_scan.py (v1): thickness + mass
        hmax = st["hmax_mm"]
        b = np.log((H - hmin) / (hmax - H))
        h = hmin + (hmax - hmin) / (1 + np.exp(-(m[0] + b)))
        mass = np.exp(m[1])
        m_lo, m_hi = 0.85, np.percentile(mass[inner], 99.9)
        labels = ("inverted thickness (mm)", "inverted mass scale")
        cmaps = ("viridis_r", "magma")
    sp = X[1, 0] - X[0, 0]
    ext = (X[0, 0] - sp / 2, X[-1, 0] + sp / 2, Y[0, 0] - sp / 2, Y[0, -1] + sp / 2)
    D = {n: (d, d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    h_lo, h_hi = np.percentile(h[inner], 0.5), max(10.05, np.percentile(h[inner], 99.5))

    fig = plt.figure(figsize=(17, 12.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.25, 0.62, 0.62], hspace=0.32, wspace=0.18,
                          left=0.05, right=0.97, top=0.92, bottom=0.04)
    top = gs[0, :].subgridspec(1, 2, wspace=0.12)
    for j, (img, lab, cm, lo, hi) in enumerate(((h, labels[0], cmaps[0], h_lo, h_hi),
                                                (mass, labels[1], cmaps[1], m_lo, m_hi))):
        ax = fig.add_subplot(top[0, j])
        im = ax.imshow(img.T, origin="lower", extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        for n, (d, x, y) in D.items():
            outline(ax, d, n, x, y)
        ax.set_title(lab + "   (outlines: as-built defects)", fontsize=11)
        ax.set_xlabel("u (mm, from scan centre)"), ax.set_ylabel("v (mm)")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.75)
    # zooms: 50 x 50 mm around each defect
    half = 25
    for i, (n, (d, x, y)) in enumerate(D.items()):
        sel_x = np.abs(X[:, 0] - x) <= half; sel_y = np.abs(Y[0, :] - y) <= half
        for row, (img, cm, lo, hi, lab) in enumerate(((h, cmaps[0], h_lo, h_hi, "thickness"),
                                                      (mass, cmaps[1], m_lo, m_hi,
                                                       "thickness (0-10 scale)" if thickness_only else "mass"))):
            ax = fig.add_subplot(gs[1 + row, i])
            sub = img[np.ix_(sel_x, sel_y)]
            e = (X[sel_x, 0][0] - sp / 2, X[sel_x, 0][-1] + sp / 2, Y[0, sel_y][0] - sp / 2, Y[0, sel_y][-1] + sp / 2)
            ax.imshow(sub.T, origin="lower", extent=e, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            outline(ax, d, n, x, y, lw=2, label=False)
            true = (f"{H - d['depth_mm']:.0f} mm left" if n.startswith("D") and d["depth_mm"] < H else
                    "through" if n.startswith("D") else f"cube {d['mass_g']} g")
            use_min = row == 0 or thickness_only
            val = sub.min() if use_min else sub.max()
            ax.set_title(f"{n} ({true}): {lab} {'min' if use_min else 'max'} {val:.2f}" + (" mm" if use_min else ""),
                         fontsize=8.5, color="k")
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Defect-scan inversion {a.tag}: misfit {st['misfit_start']:.3f} -> {st['misfit_end']:.3f}, window "
                 f"{st['window_us'][0]:.0f}-{st['window_us'][1]:.0f} us, h_min {hmin:g} mm.  Top: full scan area; bottom: "
                 f"50 x 50 mm zooms (same colour scales)", fontsize=12)
    fig.savefig(out / "maps_clear.png", dpi=90)
    print(out / "maps_clear.png")


if __name__ == "__main__":
    main()
