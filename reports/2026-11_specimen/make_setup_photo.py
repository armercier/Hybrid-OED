"""Appendix figure A2: laboratory setup photo (LDV_photos/IMG_6341, converted from HEIC) with a few labels and
illustrative green laser beams (drawn, not photographed) from the three head apertures to one scan point.
Usage:  python reports/2026-11_specimen/make_setup_photo.py
Output: reports/2026-11_mssp_paper/figures/A2_setup.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
from paper_style import INK, apply  # noqa: E402

PHOTO = REPO / "reports" / "2026-11_mssp_paper" / "LDV_photos" / "jpg" / "IMG_6341.jpg"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
# exit points (fractions of width, height): the visible aperture window between the heads, and the middle of the
# lower edge of the left and right heads' front (bottom, dark blue) faces
BEAMS_FROM = [(0.366, 0.308), (0.521, 0.296), (0.723, 0.276)]
BEAM_TO = (0.50, 0.625)                                            # one point in the scan area
GREEN = "#3cf03c"
LABELS = [("3-D scanning laser\nvibrometer heads", (0.42, 0.24), (0.18, 0.08)),
          ("robot", (0.78, 0.47), (0.86, 0.40)),
          ("PMMA plate, 10 mm", (0.62, 0.62), (0.66, 0.80)),
          ("scan area", (0.56, 0.63), (0.66, 0.53)),
          ("foam support", (0.30, 0.90), (0.10, 0.97))]


def main():
    apply()
    im = Image.open(PHOTO); im.thumbnail((1600, 1600))
    a = np.asarray(im)
    h, w = a.shape[:2]
    fig, ax = plt.subplots(figsize=(4.2, 4.2 * h / w), layout="constrained")
    ax.imshow(a); ax.axis("off")
    tx_, ty_ = BEAM_TO[0] * w, BEAM_TO[1] * h
    for fx, fy in BEAMS_FROM:
        for lw, al in ((1.8, 0.12), (0.6, 0.9)):                         # faint halo + thin core
            ax.plot([fx * w, tx_], [fy * h, ty_], color=GREEN, lw=lw, alpha=al, solid_capstyle="round", zorder=3)
    for ms, al in ((4, 0.3), (2, 1.0)):
        ax.plot(tx_, ty_, "o", color=GREEN, ms=ms, alpha=al, mec="none", zorder=4)
    for t, (x, y), (tx, ty) in LABELS:
        ax.annotate(t, (x * w, y * h), xytext=(tx * w, ty * h), fontsize=7.5, color=INK, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), arrowprops=dict(arrowstyle="-", lw=0.7, color="white"))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"A2_setup.{ext}", dpi=250)
    print(OUT / "A2_setup.png")


if __name__ == "__main__":
    main()
