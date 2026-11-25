"""Drawing of the PMMA lab plate (900 x 835 x 10 mm): defects, PZT discs, glued cube, LDV scan area.

Views: front face (LDV side), back face (as seen after flipping the plate about its short, vertical
side, so y is unchanged and x is mirrored: x_back = 900 - x), a schematic section (not to scale)
showing which face each item is on, and a table with both coordinate frames.
Sizes for 40 kHz: A0 wavelength about 25 mm in 10 mm PMMA. Scan area 300 x 200 mm, centred.

Usage:  python reports/2026-11_specimen/make_drawing_pmma.py
Output: reports/2026-11_specimen/specimen_pmma_drawing.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
L, W, H = 900.0, 835.0, 10.0
SCAN = (300, 318, 600, 518)
# id, kind, x, y (front frame, mm), size, depth, note
HOLES = [("D1", "through", 375, 465, 20, 10.0), ("D2", "pit", 520, 470, 30, 5.0),
         ("D3", "through", 450, 405, 10, 10.0), ("D4", "pit", 365, 365, 15, 5.0)]
PZT = [("S1", 450, 248), ("S2", 670, 430), ("S3", 450, 588), ("S4", 230, 405)]
CUBE = ("M1", 530, 370, 10)
C_PZT, C_CUBE, C_SCAN, C_PIT = "#e07b00", "#0a7d3b", "#1a3fa0", "#9a9a9a"
# label offsets (dx, dy) in mm in the front frame; dx is mirrored on the back view
LABEL = {"D1": (-30, 40), "D2": (30, 40), "D3": (30, 30), "D4": (-30, -30), "M1": (30, -30)}


def view(ax, back):
    fx = (lambda x: L - x) if back else (lambda x: x)
    ax.add_patch(Rectangle((0, 0), L, W, fc="#f3efe6", ec="k", lw=1.4))
    x0, y0, x1, y1 = SCAN
    ax.add_patch(Rectangle((min(fx(x0), fx(x1)), y0), x1 - x0, y1 - y0, fill=False, ec=C_SCAN, lw=1.1,
                           ls="--" if back else "-"))
    for d, kind, x, y, dia, depth in HOLES:
        on_this_face = kind == "through" or back
        if kind == "through":
            ax.add_patch(Circle((fx(x), y), dia / 2, fc="white", ec="k", lw=1.0))
            ax.plot([fx(x) - dia / 2 - 3, fx(x) + dia / 2 + 3], [y, y], "k-", lw=0.4)
            ax.plot([fx(x), fx(x)], [y - dia / 2 - 3, y + dia / 2 + 3], "k-", lw=0.4)
        else:
            ax.add_patch(Circle((fx(x), y), dia / 2, fc=C_PIT if on_this_face else "none", ec="k", lw=1.0,
                                ls="-" if on_this_face else "--", alpha=0.9))
        lab = f"{d}  " + (f"through-hole {dia:g} mm" if kind == "through" else f"pocket {dia:g} mm, {depth:g} mm deep")
        dx, dy = LABEL[d]
        dx = -dx if back else dx
        ax.annotate(f"{lab}\n({fx(x):.0f}, {y:.0f})", (fx(x), y), xytext=(fx(x) + dx, y + dy), fontsize=6.5,
                    ha="left" if dx > 0 else "right", va="center", arrowprops=dict(arrowstyle="-", lw=0.4))
    for s, x, y in PZT:
        ax.add_patch(Circle((fx(x), y), 5, fc=C_PZT if back else "none", ec=C_PZT, lw=1.2, ls="-" if back else "--"))
        ax.text(fx(x), y - 26 if y < W / 2 else y + 14, f"{s} ({fx(x):.0f}, {y:.0f})", ha="center", fontsize=6.5,
                color=C_PZT)
    m, x, y, a = CUBE
    ax.add_patch(Rectangle((fx(x) - a / 2, y - a / 2), a, a, fc=C_CUBE if back else "none", ec=C_CUBE, lw=1.2,
                           ls="-" if back else "--"))
    dx, dy = LABEL[m]
    dx = -dx if back else dx
    ax.annotate(f"{m}  steel cube {a:g} mm\n({fx(x):.0f}, {y:.0f})", (fx(x), y), xytext=(fx(x) + dx, y + dy),
                fontsize=6.5, color=C_CUBE, ha="left" if dx > 0 else "right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C_CUBE))
    # corner marker: the front-face origin
    cx = L if back else 0.0
    ax.add_patch(Polygon([(cx, 0), (cx + (-30 if back else 30), 0), (cx, 30)], fc="#b0181e", ec="none"))
    ax.text(cx + (-34 if back else 34), 6, "corner A (front origin)", fontsize=6, color="#b0181e",
            ha="right" if back else "left")
    # dimensions
    ax.annotate("", (0, -22), (L, -22), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(L / 2, -45, f"{L:.0f} mm", ha="center", fontsize=7)
    ax.annotate("", (-22, 0), (-22, W), arrowprops=dict(arrowstyle="<->", lw=0.6))
    ax.text(-40, W / 2, f"{W:.0f} mm", va="center", rotation=90, fontsize=7)
    ax.set_xlim(-55, L + 15), ax.set_ylim(-55, W + 15), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(("BACK face (machining side): PZT discs, pockets, glued cube; x_back = 900 - x\n"
                  "blue dashed frame: LDV scan area (on the front face)"
                  if back else "FRONT face (LDV side): simulation coordinates, origin at corner A\n"
                  "blue frame: LDV scan area 300 x 200 mm, x 300-600, y 318-518"),
                 fontsize=10, loc="left", fontweight="bold")


def section(ax):
    """Schematic section (not to scale): which face each item is on."""
    t = 6.0
    ax.add_patch(Rectangle((0, 0), 100, t, fc="#d9d4c7", ec="k", lw=1))
    ax.text(101, t / 2, "PMMA 10 mm\n(not to scale)", fontsize=7, va="center")
    for x in (14, 30, 46, 62, 78, 94):
        ax.add_patch(FancyArrowPatch((x, t + 9), (x, t + 0.3), arrowstyle="-|>", mutation_scale=8, color=C_SCAN, lw=0.8))
    ax.text(50, t + 11, "FRONT face: LDV beam, scan area sprayed white (film removed)", ha="center", fontsize=8, color=C_SCAN)
    ax.text(50, -16, "BACK face: PZT discs, pockets, glued cube (plate lies on foam pads)", ha="center", fontsize=8)
    # PZT disc
    ax.add_patch(Rectangle((6, -1.8), 10, 1.8, fc=C_PZT, ec="k", lw=0.6))
    ax.text(11, -4.2, "PZT disc\n10 mm, 1 g", ha="center", va="top", fontsize=6.5, color=C_PZT)
    # through-hole
    ax.add_patch(Rectangle((28, -0.05), 8, t + 0.1, fc="white", ec="none"))
    ax.plot([28, 28], [0, t], "k-", lw=1), ax.plot([36, 36], [0, t], "k-", lw=1)
    ax.text(32, -4.2, "through-hole\nD1, D3", ha="center", va="top", fontsize=6.5)
    # pit
    ax.add_patch(Rectangle((50, -0.05), 12, t / 2 + 0.05, fc="white", ec="none"))
    ax.plot([50, 50, 62, 62], [0, t / 2, t / 2, 0], "k-", lw=1)
    ax.text(56, -4.2, "flat-bottom pocket\n5 mm deep, D2, D4 (milled)", ha="center", va="top", fontsize=6.5)
    # cube
    ax.add_patch(Rectangle((76, -6), 6, 6, fc=C_CUBE, ec="k", lw=0.6))
    ax.text(79, -7, "steel cube M1\n10 mm, glued", ha="center", va="top", fontsize=6.5, color=C_CUBE)
    # foam pad
    ax.add_patch(Rectangle((93, -6), 7, 6, fc="#f5e6a8", ec="k", lw=0.5, hatch="..."))
    ax.text(96.5, -7, "foam", ha="center", va="top", fontsize=6)
    ax.set_xlim(-2, 118), ax.set_ylim(-20, t + 16), ax.axis("off")
    ax.set_title("Section (schematic): which face", fontsize=10, loc="left", fontweight="bold")


def table(ax):
    rows = [["item", "type", "size", "front (x, y) mm", "back x_back mm", "face"]]
    for s, x, y in PZT:
        rows.append([s, "PZT disc (source)", "10 mm, 1 g", f"({x}, {y})", f"{L - x:.0f}", "back"])
    for d, kind, x, y, dia, depth in HOLES:
        rows.append([d, "through-hole" if kind == "through" else "flat-bottom pocket (milled)",
                     f"{dia} mm" + ("" if kind == "through" else f", {depth} mm deep"), f"({x}, {y})", f"{L - x:.0f}",
                     "through" if kind == "through" else "back"])
    m, x, y, a = CUBE
    rows.append([m, "steel cube (glued)", f"{a} mm, ~8 g", f"({x}, {y})", f"{L - x:.0f}", "back"])
    rows.append(["scan", "LDV grid", "300 x 200 mm", "x 300-600, y 318-518", "x 300-600", "front"])
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="left", colLoc="left",
                  colWidths=[0.08, 0.22, 0.17, 0.22, 0.15, 0.1])
    tb.auto_set_font_size(False), tb.set_fontsize(7.5), tb.scale(1, 1.35)
    ax.axis("off")
    ax.set_title("Features", fontsize=10, loc="left", fontweight="bold")


def main():
    fig = plt.figure(figsize=(16.5, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.42], hspace=0.08, wspace=0.06, left=0.03, right=0.98,
                          top=0.91, bottom=0.03)
    view(fig.add_subplot(gs[0, 0]), back=False)
    view(fig.add_subplot(gs[0, 1]), back=True)
    section(fig.add_subplot(gs[1, 0]))
    table(fig.add_subplot(gs[1, 1]))
    fig.suptitle("Lab specimen: PMMA 900 x 835 x 10 mm, 4 PZT sources (40 kHz), 2 through-holes, 2 milled "
                 "half-depth pockets, 1 glued steel cube.\nSolid = on / through this face, dashed = on the other face.  "
                 "y is the same on both faces, x is mirrored on the back (x_back = 900 - x).  "
                 "Note the real positions, sizes, depths and masses after machining.", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"specimen_pmma_drawing.{ext}", dpi=130)
    print(HERE / "specimen_pmma_drawing.png")


if __name__ == "__main__":
    main()
