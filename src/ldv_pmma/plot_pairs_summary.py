"""Paper figure: adaptive design vs random moves of the same lengths, all 13 pairs (4, 2 and 1 shots), scored with v3
(score_inversions_v3.py). Rows: features found (of 5), false indications (log scale), background RMS. Within each pair:
adaptive (filled circle), random moves (triangle), static start (open square, grey); the dashed line is that setting's full scan.
A tick above a pair ("adaptive wins") marks where adaptive ranks above random (found, then false indications, then background RMS).

Usage:  python src/ldv_pmma/plot_pairs_summary.py
Output: reports/2026-11_mssp_paper/figures/F8_pairs.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
S = json.loads((REPO / "ldv_pmma_results" / "inversion" / "scores_v3.json").read_text())
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
from paper_style import ADAPTIVE as BLUE, RANDOM as ORANGE, STATIC as GREY, INK, MUTED, GRID, apply
GROUPS = [  # label, adaptive/random prefix, static prefix, full scan, starts
    ("4 shots", ("ad_", "rn_"), "", "v3", ["rc30", "j30a", "x48c"]),
    ("2 shots", ("a2_", "r2_"), "s2", "s2full", ["rc30", "j30a", "x48c", "j30b", "x48a"]),
    ("1 shot", ("a1_", "r1_"), "k1", "k1full", ["rc30", "j30a", "x48c", "j30b", "x48a"]),
]
ROWS = [("found", "features found (of 5)", False), ("false_alarm", "false indications (cells)", True),
        ("bg_rms", "background RMS (mm)", False)]


def key(s):
    return (-s["found"], s["false_alarm"], s["bg_rms"])


def main():
    apply()
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(hspace=0.12))
    x, ticks, labels, spans, wins = 0, [], [], [], []
    for g, (ap, rp), sp, full, starts in GROUPS:
        x0 = x
        n_win = 0
        for s in starts:
            A, Rn, St = S[ap + s], S[rp + s], S[sp + s]
            win = key(A) < key(Rn); n_win += win
            for ax, (k, _, logy) in zip(axs, ROWS):
                ax.plot([x - 0.17, x + 0.17], [A[k], Rn[k]], color=GREY, lw=0.8, zorder=1)
                ax.scatter(x, St[k], marker="s", s=34, facecolors="none", edgecolors=GREY, lw=1.2, zorder=2)
                ax.scatter(x - 0.17, A[k], marker="o", s=40, color=BLUE, edgecolors="white", lw=1.0, zorder=3)
                ax.scatter(x + 0.17, Rn[k], marker="^", s=44, color=ORANGE, edgecolors="white", lw=1.0, zorder=3)
            if win:
                axs[0].text(x, 5.75, "✓", ha="center", va="center", color=INK, fontsize=10)
            ticks.append(x); labels.append(s); x += 1
        for ax, (k, _, _) in zip(axs, ROWS):
            ax.plot([x0 - 0.4, x - 0.6], [max(S[full][k], 1) if k == "false_alarm" else S[full][k]] * 2,
                    ls="--", color=MUTED, lw=0.9, zorder=0)
        spans.append((x0, x - 1, g, n_win, len(starts)))
        x += 0.8
    for ax, (k, lab, logy) in zip(axs, ROWS):
        ax.set_ylabel(lab)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(2, 3000)
        for a, b, *_ in spans[1:]:
            ax.axvline(a - 0.9, color=GRID, lw=0.8)
    axs[0].set_ylim(0.5, 6.2); axs[0].set_yticks(range(1, 6))
    axs[2].set_ylim(0, 1.65)
    for a, b, g, n, m in spans:
        axs[0].text((a + b) / 2, 6.6, f"{g}\nadaptive wins {n} of {m}", ha="center", va="bottom",
                    fontsize=9, color=INK, clip_on=False)
    axs[2].set_xticks(ticks); axs[2].set_xticklabels(labels)
    axs[2].set_xlabel("starting layout (30-31 sensors; rc = regular, j = jittered, x = random)")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="adaptive design"),
         plt.Line2D([], [], marker="^", ls="", color=ORANGE, ms=6, label="random moves, same lengths"),
         plt.Line2D([], [], marker="s", ls="", mfc="none", mec=GREY, ms=6, label="static start"),
         plt.Line2D([], [], ls="--", color=MUTED, label="full scan")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=8.5)
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.1, right=0.98)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F8_pairs.{ext}", dpi=200)
    print(OUT / "F8_pairs.png")


if __name__ == "__main__":
    main()
