"""Paper figure F10: cost of one design (outer) gradient, K = 20 unrolled FWI updates, 2-D acoustic study, for
three settings: CPU with the workshop implementation, CPU with rematerialisation, GPU (RTX 2080 Ti) with
rematerialisation. Numbers from src/remat/README.md ("One hypergradient"); gradients agree with the workshop
implementation to 3-6e-6. Illumination on GPU with probe batch 12. Colours: osloS (paper_style).

Usage:  python src/ldv_pmma/plot_paper_remat.py
Output: reports/2026-11_mssp_paper/figures/F10_remat.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import ADAPTIVE, GRID, INK, RANDOM, STATIC, apply

OUT = Path(__file__).resolve().parents[2] / "reports" / "2026-11_mssp_paper" / "figures"
OBJ = ["Oracle", "Waveform\nmisfit", "Illumination"]
SETTINGS = [("CPU, workshop", STATIC, [95, 99, 36], [64.2, 66.0, 9.7]),
            ("CPU + remat", RANDOM, [116, 120, 99], [4.3, 4.8, 2.4]),
            ("GPU + remat", ADAPTIVE, [7.1, 7.4, 3.6], [3.1, 5.9, 2.8])]


def main():
    apply()
    x = np.arange(len(OBJ))
    w = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), layout="constrained")
    for i, (lab, col, t, m) in enumerate(SETTINGS):
        for ax, v, fmt in ((axs[0], t, lambda a: f"{a:.0f}" if a >= 10 else f"{a:.1f}"), (axs[1], m, lambda a: f"{a:.1f}")):
            bars = ax.bar(x + (i - 1) * w, v, w * 0.95, color=col, label=lab)
            for r in bars:
                ax.annotate(fmt(r.get_height()), (r.get_x() + r.get_width() / 2, r.get_height()), ha="center", va="bottom",
                            fontsize=6.5, color=INK, xytext=(0, 2), textcoords="offset points",
                            bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)
    gpu_line = axs[1].axhline(11, color=INK, lw=0.8, ls="--", zorder=1)
    axs[0].set_ylabel("time per design gradient (s)"); axs[1].set_ylabel("peak memory (GiB)")
    axs[0].set_ylim(0, 150); axs[1].set_ylim(0, 80)
    for ax in axs:
        ax.set_xticks(x, OBJ, fontsize=8); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    axs[1].legend([gpu_line], ["GPU memory (11 GB)"], frameon=False, fontsize=7.5, loc="upper right")
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F10_remat.{ext}", dpi=200)
    print(OUT / "F10_remat.png")


if __name__ == "__main__":
    main()
