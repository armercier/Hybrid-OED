"""Map of the LDV points of an OED replay run: where they start, where the design moves them, and which scanned point
was measured, round by round (CPU only, numpy). Works for both counting rules of oed_replay.py:
  --move-all: the same points move every round, so the arrows chain into one path per point;
  rule (b):   each round adds new points; arrows go from their regular starting position to the designed one.

Left: final inverted thickness (grey) with the paths. Right: one small panel per round (that round's arrows and
measured points, earlier measured points in grey). As-built defect outlines are drawn for evaluation only (the
design never sees them).

Usage:  python src/ldv_pmma/plot_oed_trajectory.py --tag o48f
Output: ldv_pmma_results/inversion/<tag>/trajectory.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_inversion import outline

REPO = Path(__file__).resolve().parents[2]
PREP = REPO / "ldv_work" / "prepared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="o48f")
    a = ap.parse_args()
    out = REPO / "ldv_pmma_results" / "inversion" / a.tag
    st = json.loads((out / "settings.json").read_text())
    r = np.load(out / "result.npz")
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    X, Y = r["Xc"] * 1e3 - O[0], r["Yc"] * 1e3 - O[1]
    h = st["hmin"] + (10.0 - st["hmin"]) * np.exp(r["m"][0])
    sp = X[1, 0] - X[0, 0]
    ext = (X[0, 0] - sp / 2, X[-1, 0] + sp / 2, Y[0, 0] - sp / 2, Y[0, -1] + sp / 2)
    D = {n: (d, d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    traj = st["trajectory"]
    nr = len(traj)
    cols = plt.cm.plasma(np.linspace(0.0, 0.85, nr))
    mode = "move-all" if st.get("move_all") else "rule (b), accumulated"
    rounds = st.get("rounds", [])

    def draw_round(ax, k, lw=1.2, ms=22):
        t = traj[k]
        s0 = np.array(t["start_mm"]) - O
        s1 = np.array(t["designed_mm"]) - O
        meas = R[t["measured_idx"]] - O
        mv = np.linalg.norm(s1 - s0, axis=1)
        for p, q, d in zip(s0, s1, mv):
            if d > 0.3:
                ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=cols[k], lw=lw, mutation_scale=10))
        ax.scatter(meas[:, 0], meas[:, 1], s=ms, c=[cols[k]], edgecolors="k", linewidths=0.4, zorder=4)

    fig = plt.figure(figsize=(20, 7.6))
    gs = fig.add_gridspec(2, 1 + (nr + 1) // 2, width_ratios=[2.6] + [1] * ((nr + 1) // 2), hspace=0.3, wspace=0.28,
                          left=0.04, right=0.99, top=0.88, bottom=0.07)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(h.T, origin="lower", extent=ext, cmap="Greys", vmin=np.percentile(h, 1), vmax=11, aspect="equal")
    for k in range(nr):
        draw_round(ax, k)
    for n, (d, x, y) in D.items():
        outline(ax, d, n, x, y)
    ax.set_xlim(R[:, 0].min() - O[0] - 3, R[:, 0].max() - O[0] + 3)
    ax.set_ylim(R[:, 1].min() - O[1] - 3, R[:, 1].max() - O[1] + 3)
    ax.set_xlabel("u (mm, from scan centre)"); ax.set_ylabel("v (mm)")
    ax.set_title(f"{a.tag}: all rounds over the final thickness map (grey, mm)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="thickness (mm), thin = light")
    for k in range(nr):
        ax.scatter([], [], c=[cols[k]], s=30, edgecolors="k", linewidths=0.4, label=f"round {k}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    seen = np.zeros(len(R), bool)
    for k in range(nr):
        axk = fig.add_subplot(gs[k % 2, 1 + k // 2])
        prev = R[seen] - O
        if len(prev) and not st.get("move_all"):
            axk.scatter(prev[:, 0], prev[:, 1], s=10, c="0.6", zorder=3)
        draw_round(axk, k, lw=1.4, ms=16)
        seen[traj[k]["measured_idx"]] = True
        for n, (d, x, y) in D.items():
            outline(axk, d, n, x, y, lw=1.0, label=False)
        axk.set_xlim(R[:, 0].min() - O[0] - 3, R[:, 0].max() - O[0] + 3)
        axk.set_ylim(R[:, 1].min() - O[1] - 3, R[:, 1].max() - O[1] + 3)
        axk.set_aspect("equal"); axk.tick_params(labelsize=7)
        e = rounds[k] if k < len(rounds) else {}
        mv = f"move med {e['move_mm_median']:.1f} / max {e['move_mm_max']:.1f} mm" if "move_mm_median" in e else "no design"
        axk.set_title(f"round {k}: {len(traj[k]['measured_idx'])} measured, {mv}", fontsize=8.5)
    dist = int(seen.sum())
    fig.suptitle(f"OED point trajectories, {a.tag} ({mode}): arrows = design moves (start -> designed), dots = measured "
                 f"scan points; {dist} distinct points measured; outlines = as-built defects (evaluation only)",
                 fontsize=11)
    fig.savefig(out / "trajectory.png", dpi=110)
    print(out / "trajectory.png")


if __name__ == "__main__":
    main()
