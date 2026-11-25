"""Paper figure F4: the PMMA specimen. (a) drawing of the front face (LDV side), oslo palette; (b) photo of the plate
(LDV_photos/IMG_6330, top-down through the transparent plate) resampled into the same x, y frame by a homography fitted
on 9 control points (D1-D4, M1, S1-S4; residuals <= 2 mm), annotated. No titles.
Positions from the scan centre O (as built; same constants as make_drawing_pmma_final.py, which stays the full
appendix drawing). PZT sources are 10 x 10 x 7 mm blocks on the back face. Items on the back face are drawn dashed.

Usage:  python reports/2026-11_specimen/make_drawing_paper.py
Output: reports/2026-11_mssp_paper/figures/F4_specimen.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
from make_drawing_pmma_final import CUBE, HOLES, PZT, SU, SV, rel  # noqa: E402
from paper_style import CAT, INK, apply  # noqa: E402

OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
PZT_SIDE = 10.0                  # 10 x 10 x 7 mm blocks
C_SCAN, C_PZT, C_PIT, C_CUBE = CAT[7], CAT[4], CAT[2], CAT[3]
PHOTO = REPO / "reports" / "2026-11_mssp_paper" / "LDV_photos" / "jpg" / "IMG_6330.jpg"   # from IMG_6330.HEIC
# control points: photo pixel (full resolution, 3024 x 4032) of each feature centre, read off the photo
PIX = {"M1": (1169, 1566), "D2": (1917, 1646), "D3": (1433, 2136), "D1": (1843, 2692), "D4": (1143, 2726),
       "S2": (1631, 513), "S1": (348, 2107), "S4": (1385, 3610), "S3": (2749, 2181)}
XL, YL = (-245, 245), (-195, 195)
LAB = {"D1": (-14, 9), "D2": (9, 9), "D3": (9, -11), "D4": (-14, -12), "M1": (10, -12)}


def positions():
    p = {d: rel(x, y) for d, _, x, y, _, _ in HOLES}
    p.update({s: rel(x, y) for s, x, y in PZT}); p[CUBE[0]] = rel(CUBE[1], CUBE[2])
    return p


def homography():
    mm = positions()
    rows = []
    for n, (u, v) in PIX.items():
        x, y = mm[n]
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    return np.linalg.svd(np.array(rows, float))[2][-1].reshape(3, 3)


def photo_panel(ax):
    img = np.asarray(Image.open(PHOTO).convert("RGB"), float) / 255
    H = homography()
    xs, ys = np.arange(XL[0], XL[1], 0.5), np.arange(YL[0], YL[1], 0.5)
    X, Y = np.meshgrid(xs, ys)
    q = np.stack([X, Y, np.ones_like(X)], -1) @ H.T
    u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    out = np.stack([map_coordinates(img[..., c], [v, u], order=1, cval=1.0) for c in range(3)], -1)
    ax.imshow(np.clip(out, 0, 1), origin="lower", extent=(XL[0], XL[1], YL[0], YL[1]))
    for n, (x, y) in positions().items():
        if n.startswith("S"):          # sources: label beside the block, towards the plate centre
            dx, dy = (-30 if x > 100 else 30 if x < -100 else 26), (0 if abs(x) > 100 else 0)
        else:
            dx, dy = LAB[n][0] * 1.6, LAB[n][1] * 1.6
        ax.annotate(n, (x, y), xytext=(x + dx, y + dy), fontsize=8, ha="center", va="center", color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    ax.add_patch(Rectangle((-SU, -SV), 2 * SU, 2 * SV, fill=False, ec=C_SCAN, lw=1.0, ls="--"))
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_xticks(range(-200, 201, 100)); ax.set_yticks(range(-150, 151, 50)); ax.tick_params(labelleft=False)


def main():
    apply()
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(9.4, 4.4), layout="constrained", sharey=True)
    ax.add_patch(Rectangle((-SU, -SV), 2 * SU, 2 * SV, fc=CAT[6], alpha=0.35, ec=C_SCAN, lw=1.2, zorder=1))
    ax.axhline(0, color=CAT[3], lw=0.5, ls="-.", zorder=0); ax.axvline(0, color=CAT[3], lw=0.5, ls="-.", zorder=0)
    ax.plot(0, 0, "+", color=INK, ms=8, mew=1.2, zorder=4)
    ax.text(3, 3, "O", fontsize=8, color=INK)
    for d, kind, x, y, dia, depth in HOLES:
        u, v = rel(x, y)
        if kind == "through":
            ax.add_patch(Circle((u, v), dia / 2, fc="white", ec=INK, lw=1.0, zorder=3))
        else:
            ax.add_patch(Circle((u, v), dia / 2, fc=C_PIT, ec=INK, lw=0.9, ls="--", zorder=3))
        du, dv = LAB[d]
        ax.text(u + du, v + dv, d, fontsize=8, ha="center", va="center", color=INK)
    m, x, y, a = CUBE
    u, v = rel(x, y)
    ax.add_patch(Rectangle((u - a / 2, v - a / 2), a, a, fc=C_CUBE, ec=INK, lw=0.9, ls="--", zorder=3))
    du, dv = LAB[m]
    ax.text(u + du, v + dv, m, fontsize=8, ha="center", va="center", color=INK)
    for s, x, y in PZT:
        u, v = rel(x, y)
        ax.add_patch(Rectangle((u - PZT_SIDE / 2, v - PZT_SIDE / 2), PZT_SIDE, PZT_SIDE, fc=C_PZT, ec=INK, lw=0.9, ls="--", zorder=3))
        ax.text(u, v + (13 if v > 0 else -14), s, fontsize=8, ha="center", va="center", color=INK)
    h = [Line2D([], [], marker="s", ls="", mfc=C_PZT, mec=INK, ms=7, label="PZT source, 10 x 10 x 7 mm (back)"),
         Line2D([], [], marker="o", ls="", mfc=C_PIT, mec=INK, ms=7, label="blind hole, 6 mm deep (back)"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=7, label="through-hole"),
         Line2D([], [], marker="s", ls="", mfc=C_CUBE, mec=INK, ms=7, label="glued steel cube, 10 mm (back)"),
         Rectangle((0, 0), 1, 1, fc=CAT[6], alpha=0.35, ec=C_SCAN, lw=1.2, label="LDV scan area (front)")]
    fig.legend(handles=h, fontsize=7, frameon=False, loc="outside lower center", ncol=5, columnspacing=1.0)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_xticks(range(-200, 201, 100)); ax.set_yticks(range(-150, 151, 50))
    photo_panel(axp)
    ax.text(0.01, 0.99, "(a)", transform=ax.transAxes, fontsize=9, va="top")
    axp.text(0.01, 0.99, "(b)", transform=axp.transAxes, fontsize=9, va="top", bbox=dict(fc="white", ec="none", pad=1))
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F4_specimen.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.03)
    print(OUT / "F4_specimen.png")


if __name__ == "__main__":
    main()
