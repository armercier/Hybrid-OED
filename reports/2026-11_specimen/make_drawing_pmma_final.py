"""AS-BUILT (final) PMMA lab plate, positions from the CENTRE OF THE LDV SCAN GRID.

Copy of make_drawing_pmma_centred.py with the defects as actually drilled on 2026-11-14
(ldv_data_comprehensive/defects.txt): D1, D2 10 mm blind holes 6 mm deep, D3 10 mm through-hole,
D4 8 mm blind hole 6 mm deep, all drilled from the back face; M1 steel cube unchanged.
Positions are the planned ones (unchanged).

Original description:

Same specimen as make_drawing_pmma.py (10 mm PMMA, 40 kHz), but the scan grid is not centred on
the plate, so all items are placed relative to the scan centre O. Front frame (LDV side): u to the
right, v up. Back face (plate flipped about a vertical axis): u is mirrored, u_back = -u, v unchanged.
Constraint for the plate position: every source at least 150 mm from any plate edge.

Usage:  python reports/2026-11_specimen/make_drawing_pmma_final.py
Output: reports/2026-11_specimen/specimen_pmma_final_drawing.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from make_drawing_pmma import C_CUBE, C_PIT, C_PZT, C_SCAN, CUBE, PZT, SCAN

# as built: id, kind, x, y (plate front frame, mm; same positions as planned), diameter, depth from the back face
HOLES = [("D1", "pit", 375, 465, 10, 6.0), ("D2", "pit", 520, 470, 10, 6.0),
         ("D3", "through", 450, 405, 10, 10.0), ("D4", "pit", 365, 365, 8, 6.0)]

HERE = Path(__file__).resolve().parent
OX, OY = (SCAN[0] + SCAN[2]) / 2, (SCAN[1] + SCAN[3]) / 2      # scan centre in the plate frame
SU, SV = (SCAN[2] - SCAN[0]) / 2, (SCAN[3] - SCAN[1]) / 2      # half sizes: 150 x 100 mm
rel = lambda x, y: (x - OX, y - OY)
EDGE = 150   # minimum source-to-plate-edge distance (mm)
sg = lambda a: f"{a + 0.0:+.0f}" if round(a) != 0 else "0"      # signed, no "-0"
LABEL = {"D1": (-30, 32), "D2": (30, 32), "D3": (40, -2), "D4": (-30, -26), "M1": (30, -28)}


def view(ax, back):
    f = (lambda u: -u) if back else (lambda u: u)
    # keep-out zone for the plate edges: sources >= EDGE from any edge
    us = [rel(x, y)[0] for _, x, y in PZT]
    vs = [rel(x, y)[1] for _, x, y in PZT]
    lo_u, hi_u = min(f(u) for u in us) - EDGE, max(f(u) for u in us) + EDGE
    lo_v, hi_v = min(vs) - EDGE, max(vs) + EDGE
    ax.add_patch(Rectangle((lo_u, lo_v), hi_u - lo_u, hi_v - lo_v, fill=False, ec="#b0181e", lw=1.0, ls=":"))
    ax.text(lo_u + 6, hi_v - 16, f"plate edges must lie outside this frame (sources >= {EDGE} mm from any edge)",
            fontsize=6.5, color="#b0181e")
    # scan grid and axes through O
    ax.add_patch(Rectangle((-SU, -SV), 2 * SU, 2 * SV, fill=False, ec=C_SCAN, lw=1.1, ls="--" if back else "-"))
    ax.plot([lo_u, hi_u], [0, 0], color="#888", lw=0.5, ls="-.", zorder=0)
    ax.plot([0, 0], [lo_v, hi_v], color="#888", lw=0.5, ls="-.", zorder=0)
    ax.plot(0, 0, "+", color=C_SCAN, ms=12, mew=1.6)
    ax.text(4, 5, "O", color=C_SCAN, fontsize=9, fontweight="bold")
    a0u, a0v = (SU + 25 if back else -SU - 25), -SV - 45          # axis arrows outside the scan corner
    ax.annotate("", (a0u + f(40), a0v), (a0u, a0v), arrowprops=dict(arrowstyle="-|>", color=C_SCAN, lw=1))
    ax.text(a0u + f(46), a0v - 3, "u", color=C_SCAN, fontsize=9, ha="center", va="center")
    ax.annotate("", (a0u, a0v + 40), (a0u, a0v), arrowprops=dict(arrowstyle="-|>", color=C_SCAN, lw=1))
    ax.text(a0u, a0v + 47, "v", color=C_SCAN, fontsize=9, ha="center")
    for d, kind, x, y, dia, depth in HOLES:
        u, v = rel(x, y)
        on = kind == "through" or back
        if kind == "through":
            ax.add_patch(Circle((f(u), v), dia / 2, fc="white", ec="k", lw=1.0))
        else:
            ax.add_patch(Circle((f(u), v), dia / 2, fc=C_PIT if on else "none", ec="k", lw=1.0, ls="-" if on else "--"))
        ax.plot([f(u) - dia / 2 - 4, f(u) + dia / 2 + 4], [v, v], "k-", lw=0.4)
        ax.plot([f(u), f(u)], [v - dia / 2 - 4, v + dia / 2 + 4], "k-", lw=0.4)
        lab = f"{d}  " + (f"through-hole {dia:g} mm" if kind == "through" else f"blind hole {dia:g} mm, {depth:g} mm deep")
        du, dv = LABEL[d]
        du = -du if back else du
        ax.annotate(f"{lab}\n(u, v) = ({sg(f(u))}, {sg(v)})", (f(u), v), xytext=(f(u) + du, v + dv), fontsize=7,
                    ha="left" if du > 0 else "right", va="center", arrowprops=dict(arrowstyle="-", lw=0.4))
    for s, x, y in PZT:
        u, v = rel(x, y)
        ax.add_patch(Rectangle((f(u) - 5, v - 5), 10, 10, fc=C_PZT if back else "none", ec=C_PZT, lw=1.2, ls="-" if back else "--"))
        ax.text(f(u), v - 20 if v < 0 else v + 10, f"{s} ({sg(f(u))}, {sg(v)})", ha="center", fontsize=7, color=C_PZT)
    m, x, y, a = CUBE
    u, v = rel(x, y)
    ax.add_patch(Rectangle((f(u) - a / 2, v - a / 2), a, a, fc=C_CUBE if back else "none", ec=C_CUBE, lw=1.2,
                           ls="-" if back else "--"))
    du, dv = LABEL[m]
    du = -du if back else du
    ax.annotate(f"{m}  steel cube {a:g} mm\n(u, v) = ({sg(f(u))}, {sg(v)})", (f(u), v), xytext=(f(u) + du, v + dv),
                fontsize=7, color=C_CUBE, ha="left" if du > 0 else "right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C_CUBE))
    ax.text(-SU + 3, SV + 4, f"scan {2 * SU:.0f} x {2 * SV:.0f} mm: u {-SU:+.0f}..{SU:+.0f}, v {-SV:+.0f}..{SV:+.0f}",
            fontsize=7, color=C_SCAN)
    ax.set_xlim(lo_u - 10, hi_u + 10), ax.set_ylim(lo_v - 10, hi_v + 10), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(("BACK face (machining side): u_back = -u, v unchanged\n"
                  "O = point behind the scan centre; blue dashed: scan area (front face)"
                  if back else "FRONT face (LDV side): origin O = centre of the scan grid\n"
                  "u to the right, v up; blue: LDV scan area"), fontsize=10, loc="left", fontweight="bold")


def section(ax):
    """Schematic section (not to scale): which face each item is on (as built)."""
    t = 6.0
    ax.add_patch(Rectangle((0, 0), 100, t, fc="#d9d4c7", ec="k", lw=1))
    ax.text(101, t / 2, "PMMA 10 mm\n(not to scale)", fontsize=7, va="center")
    for x in (14, 30, 46, 62, 78, 94):
        ax.add_patch(FancyArrowPatch((x, t + 9), (x, t + 0.3), arrowstyle="-|>", mutation_scale=8, color=C_SCAN, lw=0.8))
    ax.text(50, t + 11, "FRONT face: LDV beam, scan area", ha="center", fontsize=8, color=C_SCAN)
    ax.text(50, -16, "BACK face: PZT blocks, blind holes, glued cube (plate lies on foam pads)", ha="center", fontsize=8)
    ax.add_patch(Rectangle((6, -4.2), 6, 4.2, fc=C_PZT, ec="k", lw=0.6))
    ax.text(9, -5.0, "PZT block\n10 x 10 x 7 mm,\n5.3 g", ha="center", va="top", fontsize=6.5, color=C_PZT)
    ax.add_patch(Rectangle((28, -0.05), 8, t + 0.1, fc="white", ec="none"))
    ax.plot([28, 28], [0, t], "k-", lw=1), ax.plot([36, 36], [0, t], "k-", lw=1)
    ax.text(32, -4.2, "through-hole\nD3, 10 mm", ha="center", va="top", fontsize=6.5)
    ax.add_patch(Rectangle((50, -0.05), 12, 0.6 * t + 0.05, fc="white", ec="none"))
    ax.plot([50, 50, 62, 62], [0, 0.6 * t, 0.6 * t, 0], "k-", lw=1)
    ax.text(56, -4.2, "blind hole, 6 mm deep\n(4 mm left) D1, D2 10 mm; D4 8 mm", ha="center", va="top", fontsize=6.5)
    ax.add_patch(Rectangle((76, -6), 6, 6, fc=C_CUBE, ec="k", lw=0.6))
    ax.text(79, -7, "steel cube M1\n10 mm, glued", ha="center", va="top", fontsize=6.5, color=C_CUBE)
    ax.add_patch(Rectangle((93, -6), 7, 6, fc="#f5e6a8", ec="k", lw=0.5, hatch="..."))
    ax.text(96.5, -7, "foam", ha="center", va="top", fontsize=6)
    ax.set_xlim(-2, 118), ax.set_ylim(-20, t + 16), ax.axis("off")
    ax.set_title("Section (schematic): which face", fontsize=10, loc="left", fontweight="bold")


def table(ax):
    rows = [["item", "type", "size", "front (u, v) mm", "back u_back", "face"]]
    for s, x, y in PZT:
        u, v = rel(x, y)
        rows.append([s, "PZT block (source)", "10 x 10 x 7 mm, 5.3 g", f"({sg(u)}, {sg(v)})", sg(-u), "back"])
    for d, kind, x, y, dia, depth in HOLES:
        u, v = rel(x, y)
        rows.append([d, "through-hole" if kind == "through" else "blind hole (drilled from back)",
                     f"{dia:g} mm" + ("" if kind == "through" else f", {depth:g} mm deep"), f"({sg(u)}, {sg(v)})",
                     sg(-u), "through" if kind == "through" else "back"])
    m, x, y, a = CUBE
    u, v = rel(x, y)
    rows.append([m, "steel cube (glued)", f"{a:g} mm, ~8 g", f"({sg(u)}, {sg(v)})", sg(-u), "back"])
    rows.append(["scan", "LDV grid", f"{2 * SU:.0f} x {2 * SV:.0f} mm", f"u {-SU:+.0f}..{SU:+.0f}, v {-SV:+.0f}..{SV:+.0f}",
                 "", "front"])
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="left", colLoc="left",
                  colWidths=[0.07, 0.22, 0.21, 0.22, 0.11, 0.09])
    tb.auto_set_font_size(False), tb.set_fontsize(7.5), tb.scale(1, 1.35)
    ax.axis("off")
    ax.set_title("Features (positions from the scan centre O)", fontsize=10, loc="left", fontweight="bold")


def main():
    fig = plt.figure(figsize=(16.5, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5], hspace=0.08, wspace=0.06, left=0.03, right=0.98,
                          top=0.89, bottom=0.03)
    view(fig.add_subplot(gs[0, 0]), back=False)
    view(fig.add_subplot(gs[0, 1]), back=True)
    section(fig.add_subplot(gs[1, 0]))
    table(fig.add_subplot(gs[1, 1]))
    fig.suptitle("ACTUAL FINAL PLATE (as built, 14 Nov 2026)\n"
                 "PMMA 10 mm, 4 PZT sources (40 kHz), 1 through-hole (D3 10 mm), 3 blind holes 6 mm deep from the back (D1, D2 10 mm; D4 8 mm), 1 glued steel cube.\n"
                 "All positions from O = centre of the LDV scan grid.  Solid = on / through this face, "
                 "dashed = other face.  Back face: u_back = -u.", fontsize=11, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"specimen_pmma_final_drawing.{ext}", dpi=130)
    print(HERE / "specimen_pmma_final_drawing.png")


if __name__ == "__main__":
    main()
