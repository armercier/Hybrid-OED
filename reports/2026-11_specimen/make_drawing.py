"""Machining and set-up drawing for the thickness-loss specimen (LdJ thesis plate, 900 x 835 x 10 mm PMMA).

Plate frame (as in the thesis): origin O at the south-west corner of the FRONT face (the face the LDV
scans), x to the East, y to the North, mm. Defects are milled from the BACK face; seen from the back the
plate is mirrored in x, so the machining coordinates are x_back = 900 - x, y_back = y (corner O ends up
at the bottom right).

Usage:  python reports/2026-11_specimen/make_drawing.py   -> specimen_drawing.pdf / .png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
LX, LY, H = 900.0, 835.0, 10.0
SCAN = (300.0, 290.0, 600.0, 540.0)            # x0, y0, x1, y1 (front frame)
# id, kind, x, y, size (diameter or (length, width)), depth, angle (deg, slot axis from +x, front frame)
DEFECTS = [
    ("D1", "hole", 385.0, 475.0, 30.0, 5.0, 0.0),
    ("D2", "hole", 520.0, 490.0, 20.0, 3.0, 0.0),
    ("D3", "hole", 480.0, 360.0, 12.0, 5.0, 0.0),
    ("D4", "slot", 370.0, 345.0, (40.0, 3.0), 5.0, 30.0),
]
SOURCES = [("S1", 450.0, 200.0), ("S2", 690.0, 430.0), ("S3", 450.0, 630.0), ("S4", 210.0, 400.0)]
OLD = [("old steel block", "rect", 350.0, 450.0, 20.0), ("old Al cylinder", "circ", 530.0, 370.0, 40.0)]
FOAM = [(60, 60), (840, 60), (60, 775), (840, 775), (450, 60), (450, 775)]
MARKS = [(300.0, 290.0), (600.0, 290.0), (300.0, 540.0)]

C_DEF, C_SRC, C_SCAN, C_OLD = "#c0392b", "#1f5fbf", "#2e8b57", "#8c8c8c"
# label offsets (points) in the FRONT view; mirrored in x for the back view
LABEL_OFF = {"D1": (-150, 26), "D2": (16, 26), "D3": (18, -34), "D4": (-150, -62)}
VSCALE = 3.0  # vertical exaggeration of the sections


def slot_corners(x, y, L, W, ang):
    a = np.deg2rad(ang)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([x, y])
    return np.array([c - u * L / 2 - v * W / 2, c + u * L / 2 - v * W / 2, c + u * L / 2 + v * W / 2, c - u * L / 2 + v * W / 2])


def draw_plate(ax, back):
    X = (lambda x: LX - x) if back else (lambda x: x)
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=True, fc="#eef4fb", ec="#1f1f1f", lw=1.4))
    # origin corner O (front frame (0, 0))
    ax.plot(X(0), 0, "ko", ms=7, zorder=5)
    ax.annotate("O", (X(0), 0), xytext=(X(0) + (-22 if back else 10), 12), fontsize=11, fontweight="bold")
    # scan area (front face) and registration marks
    x0, y0, x1, y1 = SCAN
    ax.add_patch(Rectangle((min(X(x0), X(x1)), y0), x1 - x0, y1 - y0, fill=False, ec=C_SCAN, lw=1.4, ls="-" if not back else "--"))
    ax.text(min(X(x0), X(x1)) + 4, y0 - 22, "LDV scan area 300 x 250 mm (front face)", color=C_SCAN, fontsize=8)
    for mx, my in MARKS:
        ax.plot(X(mx), my, marker="+", color=C_SCAN, ms=12, mew=1.6)
    # old glue spots
    for lab, kind, ox, oy, s in OLD:
        if kind == "rect":
            ax.add_patch(Rectangle((X(ox) - s / 2, oy - s / 2), s, s, fill=False, ec=C_OLD, lw=0.8, ls=":"))
        else:
            ax.add_patch(Circle((X(ox), oy), s / 2, fill=False, ec=C_OLD, lw=0.8, ls=":"))
    # defects (solid on the back face, dashed = hidden when seen from the front)
    ls = "-" if back else "--"
    for d, kind, x, y, s, depth, ang in DEFECTS:
        if kind == "hole":
            ax.add_patch(Circle((X(x), y), s / 2, fc=C_DEF if back else "none", alpha=0.85 if back else 1, ec=C_DEF, lw=1.3, ls=ls))
            lab = f"{d}: Ø{s:.0f}, {depth:.0f} deep"
        else:
            pts = slot_corners(X(x), y, s[0], s[1], 180 - ang if back else ang)
            ax.add_patch(Polygon(pts, closed=True, fc=C_DEF if back else "none", ec=C_DEF, lw=1.3, ls=ls))
            lab = f"{d}: slot {s[0]:.0f}x{s[1]:.0f}, {depth:.0f} deep, {ang:.0f}°"
        ax.plot(X(x), y, "+", color="#1f1f1f", ms=6, mew=0.8)
        ox, oy = LABEL_OFF[d]
        if back:
            ox = 16 if ox < 0 else -150
        ax.annotate(f"{lab}\n(x={X(x):.0f}, y={y:.0f})", (X(x), y), xytext=(ox, oy), textcoords="offset points", fontsize=7.5,
                    color=C_DEF, arrowprops=dict(arrowstyle="-", color=C_DEF, lw=0.6))
    # sources (bonded on the back face)
    for s, x, y in SOURCES:
        ax.add_patch(Rectangle((X(x) - 5, y - 5), 10, 10, fc=C_SRC, ec="#0d2f66", lw=0.8, alpha=0.9 if back else 0.5))
        ax.annotate(f"{s} ({X(x):.0f}, {y:.0f})", (X(x), y), xytext=(8, -14), textcoords="offset points", fontsize=7.5, color=C_SRC)
    # foam supports
    for fx, fy in FOAM:
        ax.add_patch(FancyBboxPatch((X(fx) - 25, fy - 25), 50, 50, boxstyle="round,pad=2", fc="#f3e3b5", ec="#b8963a", lw=0.6, alpha=0.55, zorder=1.5))
    ax.set_xlim(-60, LX + 60), ax.set_ylim(-70, LY + 50)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, LX + 1, 100)), ax.set_yticks(np.arange(0, LY + 1, 100))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel(("x_back = 900 − x" if back else "x") + " (mm)", fontsize=8), ax.set_ylabel("y (mm)", fontsize=8)


def section(ax, kind, dia, depth, title):
    w = 80.0
    v, Hs = VSCALE, H * VSCALE
    prof = [(-w / 2, 0), (-dia / 2, 0), (-dia / 2, depth * v), (dia / 2, depth * v), (dia / 2, 0), (w / 2, 0), (w / 2, Hs), (-w / 2, Hs)]
    ax.add_patch(Polygon(prof, closed=True, fc="#dbe7f5", ec="#1f1f1f", lw=1.0, hatch="///"))
    ax.annotate("", (dia / 2 + 4, 0), (dia / 2 + 4, depth * v), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(dia / 2 + 6, depth * v / 2, f"depth {depth:g}", fontsize=8, va="center", fontweight="bold")
    ax.annotate("", (-dia / 2 - 4, depth * v), (-dia / 2 - 4, Hs), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-w / 2 - 2, (depth * v + Hs) / 2, f"{H - depth:g} remaining", fontsize=8, va="center", ha="right", fontweight="bold")
    ax.annotate("", (-dia / 2, -4), (dia / 2, -4), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(0, -7, ("Ø" if kind == "hole" else "width ") + f"{dia:g}", fontsize=8, ha="center", va="top", fontweight="bold")
    ax.text(w / 2 + 2, Hs - 2, "FRONT face\n(LDV, untouched)", fontsize=6.5, va="top")
    ax.text(w / 2 + 2, 2, "BACK face\n(milled)", fontsize=6.5, va="bottom")
    ax.text(-w / 2, -14, f"plate 10 thick; vertical scale x{VSCALE:g}", fontsize=6, color="#666")
    ax.set_xlim(-w / 2 - 38, w / 2 + 32), ax.set_ylim(-17, Hs + 3)
    ax.axis("off")
    ax.set_title(title, fontsize=8, loc="left")


def main():
    fig = plt.figure(figsize=(16.5, 13.5), layout="constrained")       # A3 landscape
    gs = fig.add_gridspec(4, 2, height_ratios=[4.2, 1.1, 1.75, 0.45])
    ax_f = fig.add_subplot(gs[0, 0]); draw_plate(ax_f, back=False)
    ax_f.set_title("1. FRONT face (LDV side) — plate frame of the thesis, defects hidden (dashed)", fontsize=10, loc="left")
    ax_b = fig.add_subplot(gs[0, 1]); draw_plate(ax_b, back=True)
    ax_b.set_title("2. BACK face (machining side) — plate turned over about the y axis, O now bottom RIGHT", fontsize=10, loc="left")

    secs = gs[1, :].subgridspec(1, 4)
    for k, (d, kind, x, y, s, depth, ang) in enumerate(DEFECTS):
        section(fig.add_subplot(secs[0, k]), kind, s if kind == "hole" else s[1], depth,
                f"3.{k + 1} Section {d}: " + (f"flat-bottom hole Ø{s:g}" if kind == "hole" else f"slot {s[0]:g} x {s[1]:g} (across width)"))

    ax_t = fig.add_subplot(gs[2, :]); ax_t.axis("off")
    rows = [["ID", "Type", "Size (mm)", "Depth (mm)", "Remaining (mm)", "Front frame x, y (mm)", "Back face x_back, y (mm)", "Orientation"]]
    for d, kind, x, y, s, depth, ang in DEFECTS:
        rows.append([d, "flat-bottom hole" if kind == "hole" else "slot (flat bottom, round ends ok)",
                     f"Ø {s:g}" if kind == "hole" else f"{s[0]:g} x {s[1]:g}", f"{depth:g}", f"{H - depth:g}",
                     f"{x:g}, {y:g}", f"{LX - x:g}, {y:g}",
                     "-" if kind == "hole" else f"{ang:g}° from +x (front) = {180 - ang:g}° from +x_back"])
    for s, x, y in SOURCES:
        rows.append([s, "PZT, out-of-plane, bonded on BACK face", "10 x 10 footprint", "-", "-", f"{x:g}, {y:g}", f"{LX - x:g}, {y:g}", "poled through thickness (z)"])
    tab = ax_t.table(cellText=rows[1:], colLabels=rows[0], loc="upper left", cellLoc="left", colLoc="left",
                     colWidths=[0.04, 0.2, 0.09, 0.07, 0.08, 0.12, 0.14, 0.22])
    tab.auto_set_font_size(False), tab.set_fontsize(8), tab.scale(1, 1.35)
    notes = ("Notes:  (1) Mark corner O on the FRONT face with a permanent pen BEFORE turning the plate over.  "
             "(2) Use a flat end mill (no twist drill: its 118° point leaves a cone), depth from the back-face surface, tolerance ±0.1 mm; "
             "measure the remaining thickness of each defect with a calliper/ultrasonic gauge and report it.  "
             "(3) PMMA: sharp tool, light passes, air cooling, no melting; deburr; keep the FRONT face clean and unscratched.  "
             "(4) Remove the old steel block and Al cylinder and all adhesive (grey dotted outlines) before scanning.  "
             "(5) Plate on soft foam blocks (beige), not clamped.  (6) Green crosses: registration marks on the FRONT face at three scan corners.  "
             "(7) Photograph the plate with a ruler after machining; report measured positions if they differ from the drawing.")
    ax_n = fig.add_subplot(gs[3, :]); ax_n.axis("off")
    ax_n.text(0, 1, notes, fontsize=8, wrap=True, va="top", transform=ax_n.transAxes)
    fig.suptitle("Thickness-loss specimen on the LdJ PMMA plate (900 x 835 x 10 mm) — all dimensions in mm", fontsize=13, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"specimen_drawing.{ext}", dpi=130)
    print(OUT / "specimen_drawing.pdf")


if __name__ == "__main__":
    main()
