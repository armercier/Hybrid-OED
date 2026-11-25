"""Paper figure F2: numerical study (2-D acoustic, workshop setting, 10 seeds), paired improvement of the final model
MSE over the fixed receiver layout for the three design objectives (a; dots = seeds, diamond = mean +- sample SD) and
against the total runtime of each run on the workshop CPU (b).
Data: oed_multiseed_full_autodiff_results/scientific/results_per_run.csv (the workshop CPU runs; the GPU + remat rerun
gives the same values to 0.3 %). Colours: osloS (paper_style); objectives also differ by marker.

Usage:  python src/ldv_pmma/plot_paper_numerical.py
Output: reports/2026-11_mssp_paper/figures/F2_numerical.{pdf,png}
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import ADAPTIVE, CAT, GRID, INK, apply

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "oed_multiseed_full_autodiff_results" / "scientific" / "results_per_run.csv"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
METHODS = [("illumination", "Stochastic\nillumination", ADAPTIVE, "o"),
           ("waveform_misfit", "Waveform\nmisfit", CAT[5], "s"),
           ("oracle", "Oracle\n(true model)", CAT[0], "D")]


def main():
    rows = list(csv.DictReader(CSV.open()))
    apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained", gridspec_kw=dict(width_ratios=[1.1, 1]))
    rng = np.random.default_rng(0)
    for i, (key, lab, col, mk) in enumerate(METHODS):
        v = np.array([float(r["improvement_percent"]) for r in rows if r["method"] == key])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)) - 0.12, v, s=22, marker=mk, color=col, alpha=0.8, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.errorbar(i + 0.18, v.mean(), yerr=v.std(ddof=1), fmt="D", ms=6, color=col, mec=INK, mew=0.7, capsize=4, lw=1.4, zorder=4)
        ax.annotate(f"{v.mean():.1f} %\n{int((v > 0).sum())}/{len(v)}", (i + 0.3, v.mean()), fontsize=7.5, va="center", color=INK)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], fontsize=8)
    ax.set_xlim(-0.5, len(METHODS) - 0.2)
    ax.set_ylabel("MSE improvement over\nfixed layout (%)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("(a) paired improvement, 10 seeds", fontsize=9, loc="left")
    for key, lab, col, mk in METHODS:
        sel = [r for r in rows if r["method"] == key]
        ax2.scatter([float(r["runtime_total_seconds"]) for r in sel], [float(r["improvement_percent"]) for r in sel], s=22,
                    marker=mk, color=col, alpha=0.85, edgecolors="white", linewidths=0.5, label=lab.replace("\n", " "))
    base = [float(r["runtime_total_seconds"]) for r in rows if r["method"] == "baseline"]
    ax2.axvline(np.mean(base), color=INK, lw=0.8, ls=":")
    ax2.text(np.mean(base) + 6, 31, "fixed layout", fontsize=7.5, color=INK, va="top")
    ax2.axhline(0, color=INK, lw=0.8, ls="--")
    ax2.set_xlabel("total runtime per run, CPU (s)"); ax2.set_ylabel("MSE improvement (%)")
    ax2.grid(color=GRID, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0.1, 0.55))
    ax2.set_title("(b) runtime vs improvement", fontsize=9, loc="left")
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F2_numerical.{ext}", dpi=200)
    print(OUT / "F2_numerical.png")


if __name__ == "__main__":
    main()
