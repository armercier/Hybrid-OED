"""Paper figure F2b: numerical study maps (workshop example run, oed_single_full_autodiff/*.npz; no new computation).
Panels lettered (c)-(g) because the paper shows this figure under the objective comparison (a, b), as one figure:
(c) true model and initial acquisition (4 fixed sources, 30 receivers on a line); (d) fixed-layout reconstruction;
(e) illumination, (f) waveform misfit, (g) oracle reconstructions with the receiver paths over the 5 design rounds;
(f) legend. True model rebuilt from the run configuration (run_oed_multiseed.py:
c_min 2100 m/s background, 55 mm square inclusion at c_max 3200 m/s centred at (-0.05, 0.05) m). Velocity in oslo;
sources red and receivers gold (outside the colour map so they stay visible).

Usage:  python src/ldv_pmma/plot_paper_numerical_maps.py
Output: reports/2026-11_mssp_paper/figures/F2b_numerical_maps.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from paper_style import ADAPTIVE, CAT, INK, SEQ_CMAP, apply

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "oed_single_full_autodiff"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "figures"
LX, CMIN, CMAX, F0, PPW, FMAXF = 1.1, 2100.0, 3200.0, 25e3, 8, 2.0
SC, SIDE = (-0.05, 0.05), 0.055
RED, GOLD = "#d7191c", "#ffc800"
METHODS = [("illumination", "Stochastic illumination", ADAPTIVE, "o"), ("fwi_data_error", "Waveform misfit", CAT[5], "s"),
           ("fwi_model_error", "Oracle (true model)", CAT[0], "D")]


def grid():
    nx = int(np.floor(LX / (CMIN / (PPW * FMAXF * F0)))) + 1
    dx = LX / (nx - 1)
    return nx, dx, -LX / 2


def true_model(nx, dx, x0):
    m = np.full((nx, nx), CMIN)
    i = lambda x: int((x - x0) / dx)
    m[i(SC[0] - SIDE / 2):i(SC[0] + SIDE / 2), i(SC[1] - SIDE / 2):i(SC[1] + SIDE / 2)] = CMAX
    return m


def main():
    nx, dx, x0 = grid()
    tm = true_model(nx, dx, x0)
    d = {k: np.load(RUN / f"{k}_receivers_full_autodiff_sgd.npz") for k, *_ in METHODS}
    base = d["illumination"]["baseline_model"]
    assert base.shape == (nx, nx)
    g2m = lambda a: x0 + np.asarray(a) * dx
    src = d["illumination"]["final_sources_m"]
    rec0 = g2m(d["illumination"]["receiver_trajectories"][0])
    ext = (x0 - dx / 2, x0 + (nx - 0.5) * dx) * 2
    vmin, vmax = CMIN, 2550.0
    mse_b = float(d["illumination"]["baseline_model_mse"])
    apply()
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.4), sharex=True, sharey=True, layout="constrained")
    panels = [("(c) true model, initial geometry", tm, rec0, None),
              (f"(d) fixed layout\nMSE {mse_b:.0f}", base, rec0, None)]
    for k, lab, *_ in METHODS:
        mse = float(d[k]["final_model_mse"])
        name = "oracle" if "Oracle" in lab else lab.lower()
        panels.append((f"({'efg'[len(panels) - 2]}) {name}\nMSE {mse:.0f} ({100 * (mse - mse_b) / mse_b:+.1f} %)",
                       d[k]["finally_inverted_model"], g2m(d[k]["receiver_trajectories"][-1]), g2m(d[k]["receiver_trajectories"])))
    for ax, (title, m, rec, tr) in zip(axs.flat, panels):
        im = ax.imshow(np.clip(m, vmin, vmax).T, origin="lower", extent=ext, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((SC[0] - SIDE / 2, SC[1] - SIDE / 2), SIDE, SIDE, fill=False, ec="white", lw=0.9, ls="--"))
        if tr is not None:
            for r in range(tr.shape[1]):
                ax.plot(tr[:, r, 0], tr[:, r, 1], color=GOLD, lw=0.7, alpha=0.75, zorder=3)
        ax.scatter(src[:, 0], src[:, 1], marker="x", color=RED, s=26, lw=1.6, zorder=4)
        ax.scatter(rec[:, 0], rec[:, 1], marker="*", color=GOLD, edgecolors=INK, linewidths=0.4, s=34, zorder=5)
        ax.set_title(title, fontsize=8.5, loc="left")
    ax = axs.flat[5]
    ax.axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="x", ls="", color=RED, mew=1.6, ms=6, label="fixed sources"),
         Line2D([], [], marker="*", ls="", mfc=GOLD, mec=INK, mew=0.4, ms=9, label="receivers (final)"),
         Line2D([], [], color=GOLD, lw=1.2, label="receiver paths,\n5 design rounds"),
         Line2D([], [], color="white", lw=1.0, ls="--", label="true inclusion")]
    leg = ax.legend(handles=h, loc="center", fontsize=8, frameon=True, facecolor=SEQ_CMAP(0.0), labelcolor="white", edgecolor="none")
    lim = 0.42
    for a in axs.flat[:5]:
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
    for a in axs[1]:
        a.set_xlabel("x (m)")
    for a in axs[:, 0]:
        a.set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01, extend="max")
    cb.set_label("wave speed (m/s)")
    OUT.mkdir(exist_ok=True)
    for ext_ in ("pdf", "png"):
        fig.savefig(OUT / f"F2b_numerical_maps.{ext_}", dpi=200)
    print(OUT / "F2b_numerical_maps.png")


if __name__ == "__main__":
    main()
