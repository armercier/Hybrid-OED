"""Diagnostic figure for one scattered-field (FB)PINN run.

Usage:  python src/pinn_feasibility/plot_run.py fbpinn_s0 [more run tags ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from common import RESULTS, T_SCALE

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scattered_pinn import Problem

INK, MUTED = "#1f1f1f", "#8a8a8a"
REF, PRED, FD = "#1f1f1f", "#2a6fdb", "#c2571a"


def main(tag: str) -> None:
    run = RESULTS / "runs" / tag
    summary = json.loads((run / "summary.json").read_text())
    prob = Problem(summary["config"]["source"])
    pred = np.load(run / "prediction.npz")
    t_us = prob.t * 1e6

    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    grid = fig.add_gridspec(3, 4)

    # Traces at four receivers, in FWI normalisation.
    picks = [0, 8, prob.n_ring - 8, prob.n_ring + 15]
    for col, r in enumerate(picks):
        ax = fig.add_subplot(grid[0, col])
        ax.plot(t_us, prob.du_traces[4][:, r] / prob.norms[4][r], color=REF, lw=1.6, label="FD r4 (reference)")
        ax.plot(t_us, prob.du_traces[1][:, r] / prob.norms[1][r], color=FD, lw=1.0, ls="--", label="FD r1 (paper grid)")
        ax.plot(t_us, pred["du_traces"][:, r] / prob.norms[4][r], color=PRED, lw=1.2, label=summary["config"]["model"])
        x, y = prob.rec_m[r]
        ax.set_title(f"receiver ({x:+.2f}, {y:+.2f}) m", fontsize=9, color=INK)
        ax.set_xlabel("t (µs)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("scattered / max|total|", fontsize=8, color=MUTED)
            ax.legend(fontsize=7, frameon=False)

    # Field snapshots: reference, prediction, difference (shared diverging scale).
    ref, fld = prob.field_ref, pred["field"]
    k_times = [ref.shape[0] * 2 // 5, ref.shape[0] * 3 // 4]
    vmax = np.max(np.abs(ref))
    for row, k in zip((1, 2), k_times):
        tau = prob.field_pts.reshape(*ref.shape, 3)[k, 0, 0, 2]
        for col, (title, img) in enumerate(
            (("FD r4 reference", ref[k]), ("prediction", fld[k]), ("prediction − reference", fld[k] - ref[k]))
        ):
            ax = fig.add_subplot(grid[row, col])
            im = ax.imshow(img.T, origin="lower", extent=(-1, 1, -1, 1), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{title}, t = {tau * T_SCALE * 1e6:.0f} µs", fontsize=9, color=INK)
            ax.tick_params(labelsize=7, colors=MUTED)
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7)

    # Error history.
    hist = summary["history"]
    ax = fig.add_subplot(grid[1:, 3])
    steps = [h["step"] for h in hist]
    ax.semilogy(steps, [h["trace_all"] for h in hist], color=PRED, lw=2, label="receiver traces")
    ax.semilogy(steps, [h["field"] for h in hist], color=MUTED, lw=2, label="box field")
    for name, val in summary["fd_context"].items():
        ax.axhline(val, color=FD if name.endswith("r1") else MUTED, ls=":", lw=1)
        ax.text(steps[-1], val, f" {name.replace('fd_', 'FD ')}", fontsize=7, va="bottom", ha="right", color=INK)
    ax.set_xlabel("training step", fontsize=8, color=MUTED)
    ax.set_ylabel("relative L2 error vs FD r4", fontsize=8, color=MUTED)
    ax.set_title(f"{tag}: {summary['n_params']:,} params, {summary['sec_per_step'] * 1e3:.0f} ms/step",
                 fontsize=9, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, frameon=False)

    fig.savefig(run / "diagnostics.png", dpi=130)
    print(run / "diagnostics.png")


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
