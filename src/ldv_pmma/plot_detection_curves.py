"""Paper figure F8c: detection curves at the same scan-time budget (30 measured LDV points), ROC style.

A single threshold t on the anomaly a = |h - 10 mm| is swept from 1 to 5 mm. At each t:
  features found   = |h - 10| > t somewhere within 8 mm of the feature (D1-D4, M1; mean over the runs of a method, of 5)
  false indications = background cells (3 mm map, |u| < 140, |v| < 90 mm, > 20 mm from every feature) with a > t
                      (median over the runs of a method)
so detection and false indications use the same threshold, as in a detection test; no threshold has to be chosen.
The v3 standard (t = 3 mm) is marked. Methods: static layouts (regular, jittered, random), adaptive illumination design
(3 groups of 10), simulation design (designed once on the defect-free background) and, when available, the adaptive runs'
final points re-inverted like a static layout (ri_*), and the progressive static control (the same
start groups measured without design, same 33/33/34 inversion: n30, s2n30, q*n_*). Run lists from run_overview.py.

Usage:  python src/ldv_pmma/plot_detection_curves.py
Output: reports/2026-11_mssp_paper/figures/F8c_detection_curves.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_style import ADAPTIVE, CAT, GRID, INK, MUTED, apply
from run_overview import DESIGN30, STATIC30
from score_inversions import INV, PREP, thickness

OUT = Path(__file__).resolve().parents[2] / "reports" / "2026-11_mssp_paper" / "figures"
FEAT = ("D1", "D2", "D3", "D4", "M1")
TH = np.round(np.arange(1.0, 5.01, 0.25), 2)
T_STD = 3.0
SHORT = {"static layouts": "static", "adaptive design": "adaptive", "simulation design": "simulation",
         "adaptive, re-inverted": "adaptive re-inv.", "static, progressive": "static progr.", "adaptive, log criterion": "adaptive log"}
# progressive static control (same 3 groups of 10, no design, same 33/33/34 inversion) paired with each adaptive run
PROG = {"a30": "n30", "s2a30": "s2n30", **{f"q{k}a_{s}": f"q{k}n_{s}" for k in (1, 2, 4) for s in ("rc30", "j30a", "j30b", "j30c", "j30d")}}


def groups(shots):
    ad = [t for t, m in DESIGN30[shots] if m.startswith("adaptive, 3")]
    g = [("static layouts", STATIC30[shots], MUTED, "s", "--"),
         ("adaptive design", ad, ADAPTIVE, "o", "-"),
         ("simulation design", [t for t, m in DESIGN30[shots] if m == "simulation design"], CAT[2], "D", "-."),
         ("adaptive, re-inverted", [f"ri_{t}" for t in ad if (INV / f"ri_{t}" / "settings.json").exists()], CAT[7], "^", ":"),
         ("adaptive, log criterion", [t for t, m in DESIGN30[shots] if "log criterion" in m], CAT[5], "P", "-"),
         ("static, progressive", [n for n in (PROG.get(t) for t in ad) if n and (INV / n / "settings.json").exists()], MUTED, "v", ":")]
    return [x for x in g if x[1]]


def curves(tag, geo):
    h, Xc, Yc, _ = thickness(tag)
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
    a = np.abs(h - 10.0)
    r = {n: np.hypot(X - d["x"] + O[0], Y - d["y"] + O[1]) for n, d in geo.items()}
    peak = np.array([a[r[n] < 8.0].max() for n in FEAT])
    bg = (np.abs(X) < 140) & (np.abs(Y) < 90) & (np.min([r[n] for n in FEAT], 0) > 20.0)
    return np.array([(peak > t).sum() for t in TH]), np.array([(bg & (a > t)).sum() for t in TH])


def main():
    apply()
    geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.4), sharey=True, sharex=True, layout="constrained")
    k3 = int(np.flatnonzero(TH == T_STD)[0])
    handles = {}
    for ax, shots in zip(axs, (4, 2, 1)):
        for lab, tags, col, mk, ls in groups(shots):
            C = [curves(t, geo) for t in tags]
            found = np.mean([c[0] for c in C], 0)
            fi = np.median([c[1] for c in C], 0)
            ln, = ax.plot(np.maximum(fi, 1), found, color=col, ls=ls, lw=1.6, zorder=3)
            ax.plot(np.maximum(fi, 1)[::2], found[::2], mk, color=col, ms=3.5, zorder=3)
            ax.plot(max(fi[k3], 1), found[k3], mk, ms=8, mfc="white", mec=col, mew=1.6, zorder=4)
            handles[lab] = (ln, f"{lab} (n = {len(tags)})" if shots == 4 else lab)
            ax.text(0.98, 0.04 + 0.075 * list(h[0] for h in groups(shots)).index(lab), f"{SHORT[lab]}: n = {len(tags)}",
                    transform=ax.transAxes, ha="right", fontsize=6.5, color=col)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"]); ax.minorticks_off()
        ax.set_title(f"{shots} shot{'s' if shots > 1 else ''}, 30 measured points", fontsize=9, loc="left")
        ax.set_xlabel("false indications (cells, median)")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("features found (of 5, mean)")
    axs[0].set_ylim(1.8, 5.15)
    axs[0].annotate("open markers:\nthreshold 3 mm (v3)\nleft end: 5 mm\nright end: 1 mm", (0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=6.5, color=INK)
    fig.legend([h[0] for h in handles.values()], [k for k in handles], loc="outside lower center", ncol=3,
               frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F8c_detection_curves.{ext}", dpi=200)
    print(OUT / "F8c_detection_curves.png")


if __name__ == "__main__":
    main()
