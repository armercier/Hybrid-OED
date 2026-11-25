"""Paper figure F8b (candidate main result): same scan-time budget, 30 measured LDV points (about 2 min of scanning).
Static layouts (regular, jittered, random) vs adaptive illumination design (3 groups of 10, each group designed on the
current inverted model), for 4, 2 and 1 shots; simulation designs (designed once on the defect-free background) and earlier
designs (greedy selection, early adaptive settings) shown for completeness. Rows: features found (of 5), false
indications (cells), background RMS. Panel notes: share of runs that find all five features;
bars: medians (static layouts; adaptive when repeated); single adaptive runs labelled. v3 scores.

Usage:  python src/ldv_pmma/plot_matched_budget.py
Output: reports/2026-11_mssp_paper/figures/F8b_matched_budget.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import ADAPTIVE, CAT, GRID, INK, STATIC, apply
from run_overview import DESIGN30, STATIC30

REPO = Path(__file__).resolve().parents[2]
S = json.loads((REPO / "ldv_pmma_results" / "inversion" / "scores_v3.json").read_text())
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
GROUPS = [("static\nlayouts", None, STATIC, "s", 26, False),
          ("adaptive\ndesign", lambda m: m.startswith("adaptive, 3"), ADAPTIVE, "o", 70, False),
          ("simulation\ndesign", lambda m: m == "simulation design", CAT[2], "D", 26, False),
          ("earlier\ndesigns", lambda m: m.startswith("greedy") or "early" in m, CAT[3], "^", 26, True)]
ROWS = [("found", "features found (of 5)"), ("false_alarm", "false indications (cells)"), ("bg_rms", "background RMS (mm)")]


def main():
    apply()
    fig, axs = plt.subplots(3, 3, figsize=(8.6, 6.2), sharey="row", sharex=True, layout="constrained")
    for c, shots in enumerate((4, 2, 1)):
        for gi, (lab, sel, col, mk, sz, open_) in enumerate(GROUPS):
            tags = STATIC30[shots] if sel is None else [t for t, m in DESIGN30[shots] if sel(m)]
            for r, (k, _) in enumerate(ROWS):
                ax = axs[r, c]
                if not tags:
                    continue
                v = np.array([S[t][k] for t in tags], float)
                x = gi + (np.linspace(-0.22, 0.22, len(v)) if len(v) > 1 else np.zeros(1))
                ax.scatter(x, v, s=sz, marker=mk, facecolors="none" if open_ else col, edgecolors=col if open_ else "white",
                           linewidths=1.0 if open_ else 0.6, zorder=3)
                if sel is None or (gi == 1 and len(v) > 1):          # median bar for static and for repeated adaptive runs
                    ax.plot([gi - 0.32, gi + 0.32], [np.median(v)] * 2, color=INK, lw=1.0, zorder=4)
                if gi == 1 and len(v) == 1:
                    for xx, vv, t in zip(x, v, tags):
                        ax.annotate(t, (xx, vv), xytext=(7, 0), textcoords="offset points", fontsize=7, va="center", color=INK)
                if r == 0:
                    n5 = int(sum(S[t]["found"] == 5 for t in tags))
                    ax.text(gi, 0.75, f"{n5}/{len(tags)}", ha="center", fontsize=8, color=INK,
                            fontweight="bold" if gi == 1 else "normal")
        axs[0, c].set_title(f"{shots} shot{'s' if shots > 1 else ''}, 30 measured points", fontsize=9, loc="left")
    for r, (k, lab) in enumerate(ROWS):
        axs[r, 0].set_ylabel(lab)
        for c in range(3):
            axs[r, c].grid(axis="y", color=GRID, lw=0.6); axs[r, c].set_axisbelow(True)
    axs[0, 0].set_ylim(0.3, 5.6); axs[0, 0].set_yticks(range(1, 6))
    axs[0, 0].text(-0.45, 1.15, "all 5 found:", fontsize=7, color=INK)
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_yticks([100, 200, 400, 800], ["100", "200", "400", "800"]); axs[1, 0].minorticks_off()
    for c in range(3):
        axs[2, c].set_xticks(range(len(GROUPS)), [g[0] for g in GROUPS], fontsize=7)
        axs[2, c].set_xlim(-0.55, len(GROUPS) - 0.45)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F8b_matched_budget.{ext}", dpi=200)
    print(OUT / "F8b_matched_budget.png")


if __name__ == "__main__":
    main()
