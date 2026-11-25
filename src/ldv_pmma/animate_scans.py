"""Lightweight GIF animations of the band-passed out-of-plane wavefield (vz), one per file.

Defect (scatter) scan by default, 3 mm grid; optional --pristine for the 6 mm pristine scan.
Frames every 4 us from 480 to 1300 us, symmetric colour scale per file (98th percentile),
defect positions overlaid at their planned (u, v) from the grid centre (approximate).

Usage:  python src/ldv_pmma/animate_scans.py [--pristine]
Output: ldv_pmma_results/scatter/anim_{scatter|pristine}_ex{k}.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from assess_scatter import DATA, DEF_UV, load

OUT = Path(__file__).resolve().parents[2] / "ldv_pmma_results" / "scatter"
T0, T1, DT = 480e-6, 1300e-6, 4e-6


def grid(R, step):
    x0, y0 = R[:, 0].min(), R[:, 1].min()
    ix = np.round((R[:, 0] - x0) / step).astype(int)
    iy = np.round((R[:, 1] - y0) / step).astype(int)
    return ix, iy, x0, y0


def animate(path, out, step, title):
    R, vz, t = load(path)
    ix, iy, x0, y0 = grid(R, step)
    nx, ny = ix.max() + 1, iy.max() + 1
    frames = np.searchsorted(t, np.arange(T0, T1, DT))
    G = np.full((len(frames), ny, nx), np.nan, np.float32)
    G[:, iy, ix] = vz[:, frames].T * 1e3          # mm/s
    nrms = np.sqrt(np.mean(vz[:, t < 450e-6] ** 2, 1))
    good = nrms < 3 * np.median(nrms)
    vmax = np.percentile(np.abs(vz[good][:, frames]), 98) * 1e3
    O = 0.5 * (R.min(0) + R.max(0))
    ext = (x0 - step / 2 - O[0], x0 + (nx - 0.5) * step - O[0], y0 - step / 2 - O[1], y0 + (ny - 0.5) * step - O[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=72)
    im = ax.imshow(G[0], origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    for nm, (du, dv, dd) in DEF_UV.items():
        ax.add_patch(plt.Circle((du, dv), dd / 2 + 2, fill=False, ec="k", lw=0.7))
        ax.text(du + 6, dv + 6, nm, fontsize=7)
    ax.set_xlabel("u from grid centre (mm)"), ax.set_ylabel("v (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="vz (mm/s)")
    ttl = ax.set_title("")
    fig.tight_layout()

    def upd(i):
        im.set_data(G[i])
        ttl.set_text(f"{title}   t = {t[frames[i]] * 1e6:5.0f} us")
        return im, ttl

    FuncAnimation(fig, upd, frames=len(frames), blit=False).save(out, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pristine = "--pristine" in sys.argv
    for k in (1, 2, 3, 4):
        if pristine:
            animate(DATA / f"pristine_mat/excitation_{k}_pristine.mat", OUT / f"anim_pristine_ex{k}.gif", 6.0,
                    f"PRISTINE ex{k}")
        else:
            animate(DATA / f"scatt_mat/excitation_{k}_scatt.mat", OUT / f"anim_scatter_ex{k}.gif", 3.0,
                    f"DEFECT scan ex{k}")


if __name__ == "__main__":
    main()
