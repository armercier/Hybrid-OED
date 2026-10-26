"""LDV vs Salvus 3-D (steel cube, Q = 40) vs Mindlin equivalent: snapshots and traces.

Reads salvus_ldv_compare.npz (LDV data, Salvus predictions) and mindlin_equivalent_ldv.npz.
Both model predictions include their own deconvolution filter (wavelet + trigger delay),
estimated on the same training points.

Usage:  python src/ldv_assessment/plot_three_way.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[2] / "ldv_assessment"
CUBE = np.array([-0.020, -0.005])


def main():
    d = np.load(OUT / "salvus_ldv_compare.npz")
    m = np.load(OUT / "mindlin_equivalent_ldv.npz")
    eq = json.loads((OUT / "mindlin_equivalent.json").read_text())
    sal = json.loads((OUT / "salvus_ldv_compare.json").read_text())["cube_x-20mm_Q40"]
    P, t, obs, src, train = d["P"], d["t"], d["obs"], d["src"], d["train"]
    fields = {"LDV (observed v_z)": obs,
              f"Salvus 3-D: plate + steel cube, Q=40   (misfit behind/before {sal['behind']:.2f}/{sal['before']:.2f})":
                  d["cube_x-20mm_Q40"],
              f"Mindlin equivalent: cube patch + damping   (misfit {eq['D_equivalent_vs_ldv'][0]:.2f}/"
              f"{eq['D_equivalent_vs_ldv'][1]:.2f})": m["pred_equivalent"]}
    if (OUT / "mindlin_rigid_best_ldv.npz").exists():  # rigid cube, footprint calibrated on Salvus
        best = json.loads((OUT / "mindlin_rigid.json").read_text())["best_sweep"]
        fields[f"Mindlin + rigid cube ({best['size']*1e3:.0f} mm effective footprint) + damping   "
               f"(misfit {best['ldv'][0]:.2f}/{best['ldv'][1]:.2f})"] = \
            np.load(OUT / "mindlin_rigid_best_ldv.npz")["pred_rigid"]
    labels = list(fields)
    rows = [(k, v) for k, v in fields.items()] + [
        ("residual: LDV - Salvus 3-D", obs - fields[labels[1]]), ("residual: LDV - Mindlin patch equivalent", obs - fields[labels[2]])]
    if len(labels) > 3:
        rows.append(("residual: LDV - Mindlin rigid cube", obs - fields[labels[3]]))
    times = (260e-6, 310e-6, 360e-6, 430e-6)
    vmax = np.percentile(np.abs(obs), 99.5)

    fig = plt.figure(figsize=(16, 2.2 * len(rows) + 9), layout="constrained")
    gs = fig.add_gridspec(len(rows) + 3, len(times), height_ratios=[1] * len(rows) + [1.1, 1.1, 1.1])
    for r, (lab, a) in enumerate(rows):
        for c, tt in enumerate(times):
            k = int(np.argmin(np.abs(t - tt)))
            ax = fig.add_subplot(gs[r, c])
            ax.scatter(P[:, 0], P[:, 1], c=a[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=1.0, linewidths=0)
            ax.add_patch(Rectangle(CUBE - 0.005, 0.01, 0.01, fill=False, ec="#1f1f1f", lw=0.8))
            ax.plot(*src, "x", color="#1f1f1f", ms=6)
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title((lab + "\n" if c == 0 else "") + f"t = {tt*1e6:.0f} µs", fontsize=8, loc="left")

    picks = {"in front of the cube (4.5 cm from source, on axis)": (src[0] + 0.045, src[1] + 0.012),
             "just behind the cube (on axis)": (CUBE[0] + 0.02, CUBE[1]),
             "far behind the cube (on axis)": (0.12, CUBE[1]),
             "beside the cube (off axis, +y)": (CUBE[0], 0.035),
             "behind, off axis (-y)": (0.06, -0.035),
             "far, off axis (+y)": (0.13, 0.035)}
    colours = ("#1f1f1f", "#c2571a", "#2a6fdb", "#1a9850")
    test = ~train
    for i, (title, (x, y)) in enumerate(picks.items()):
        cand = np.where(test)[0]
        r = cand[np.argmin(np.hypot(P[cand, 0] - x, P[cand, 1] - y))]
        ax = fig.add_subplot(gs[len(rows) + i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
        for (lab, a), col, lw in zip(fields.items(), colours, (1.8, 1.1, 1.0, 1.1)):
            ax.plot(t * 1e6, a[:, r], color=col, lw=lw, label=lab.split("   ")[0])
        ax.set_xlim(150, 650), ax.grid(alpha=0.3)
        ax.set_title(f"held-out trace {title}: ({P[r,0]:+.3f}, {P[r,1]:+.3f}) m", fontsize=8, loc="left")
        if i >= 4:
            ax.set_xlabel("t (µs)")
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("LDV scan (10 mm PMMA, 10 mm steel cube; scan frame), band-passed 8-70 kHz. "
                 "x = transducer, square = cube", fontsize=10)
    fig.savefig(OUT / "ldv_salvus_mindlin.png", dpi=95)
    print(OUT / "ldv_salvus_mindlin.png")


if __name__ == "__main__":
    main()
