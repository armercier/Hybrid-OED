"""Figure for the Mindlin fit of the LDV scan (reads ldv_assessment/mindlin_ldv_fit.npz).

Usage:  python src/ldv_assessment/plot_mindlin_fit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "ldv_assessment"


def main():
    d = np.load(OUT / "mindlin_ldv_fit.npz")
    meta = json.loads((OUT / "mindlin_ldv_fit.json").read_text())
    P, t, obs, pred, train, src = d["P"], d["t"], d["obs"], d["pred"], d["train"], d["src"]
    res = obs - pred
    times = (300e-6, 380e-6, 460e-6, 540e-6)
    vmax = np.percentile(np.abs(obs), 99.5)

    fig = plt.figure(figsize=(15, 12), layout="constrained")
    gs = fig.add_gridspec(5, len(times))
    for r, (name, a) in enumerate((("LDV v_z (observed)", obs), ("Mindlin prediction", pred),
                                   ("residual (observed - predicted)", res))):
        for c, tt in enumerate(times):
            k = int(np.argmin(np.abs(t - tt)))
            ax = fig.add_subplot(gs[r, c])
            ax.scatter(P[:, 0], P[:, 1], c=a[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=1.2, linewidths=0)
            ax.plot(*src, "x", color="#1f1f1f", ms=6)
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title((name + "\n" if c == 0 else "") + f"t = {tt*1e6:.0f} µs", fontsize=8, loc="left")

    ax = fig.add_subplot(gs[3, :2])
    e_res = np.sqrt(np.mean(res**2, 0)) / np.sqrt(np.mean(obs**2, 0)).max()
    sc = ax.scatter(P[:, 0], P[:, 1], c=e_res, cmap="magma", s=1.5, linewidths=0)
    ax.plot(*src, "x", color="#2a6fdb", ms=7)
    ax.set_aspect("equal"), ax.set_xlabel("x (m, scan frame)"), ax.set_ylabel("y (m)")
    ax.set_title("residual RMS per scan point (relative): unexplained by a homogeneous plate", fontsize=8, loc="left")
    fig.colorbar(sc, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[3, 2:])
    F = d["F"]
    nf = 2 * (len(F) - 1)
    imp = np.fft.irfft(F, nf)
    dt = t[1] - t[0]
    tau = np.arange(nf) * dt
    tau = np.where(tau > nf * dt / 2, tau - nf * dt, tau)
    o = np.argsort(tau)
    ax.plot(tau[o] * 1e6, imp[o], color="#1f1f1f", lw=1)
    ax.set_xlim(-50, 450), ax.set_xlabel("lag (µs)"), ax.grid(alpha=0.3)
    ax.set_title("estimated source filter (wavelet + trigger delay, relative to a 45 kHz Ricker)", fontsize=8, loc="left")

    test = np.where(~train)[0]
    dist = np.hypot(*(P[test] - src).T)
    picks = test[np.argsort(dist)[[len(test) // 10, len(test) // 3, 2 * len(test) // 3, -10]]]
    for c, r in enumerate(picks):
        ax = fig.add_subplot(gs[4, c])
        ax.plot(t * 1e6, obs[:, r], color="#1f1f1f", lw=1.4, label="LDV")
        ax.plot(t * 1e6, pred[:, r], color="#2a6fdb", lw=1.0, label="Mindlin")
        ax.set_title(f"held-out trace, {np.hypot(*(P[r] - src)):.2f} m from source", fontsize=8, loc="left")
        ax.set_xlabel("t (µs)"), ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    b = meta["best"]
    fig.suptitle(f"LDV scan vs JAX Mindlin (open plate, vs {b['vs']:.0f} m/s, h {b['h']*1e3:.1f} mm, "
                 f"kappa {b['kappa']:.3f}); held-out misfit {b['test']:.3f}", fontsize=10)
    fig.savefig(OUT / "mindlin_ldv_fit.png", dpi=105)
    print(OUT / "mindlin_ldv_fit.png")


if __name__ == "__main__":
    main()
