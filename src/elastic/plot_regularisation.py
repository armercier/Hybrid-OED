"""Compare plate-FWI variants (early stopping, TV, window/taper): maps and metrics table.

Usage:  python src/elastic/plot_regularisation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "elastic_results" / "inversion"
DEFN = json.loads((REPO / "salvus_plate" / "benchmark_definition.json").read_text())
RUNS = [
    ("salvus_tmax500us", "baseline (500 µs)"),
    ("tv1e-4_500us", "TV 1e-4 (500 µs)"),
    ("tv3e-4_500us", "TV 3e-4 (500 µs)"),
    ("taper_300us", "300 µs + edge taper"),
    ("combined_tv1e-4_300us", "300 µs + taper + TV 1e-4"),
    ("combined_tv3e-4_300us", "300 µs + taper + TV 3e-4"),
    ("plane_stress_tmax500us", "inverse crime (reference)"),
]


FIG_NAME = "regularisation_maps"


def stop_at(history, window=5, tol=0.02):
    for it in range(window, len(history)):
        prev = history[it - window]["loss"]
        if (prev - history[it]["loss"]) / prev < tol:
            return it
    return len(history) - 1


def main():
    rows, maps = [], []
    true_vs = None
    for tag, label in RUNS:
        if not (OUT / f"{tag}.json").exists():
            continue
        meta = json.loads((OUT / f"{tag}.json").read_text())
        r = np.load(OUT / f"{tag}.npz")
        true_vs = r["true_vs"]
        h = meta["history"]
        s = stop_at(h)
        best = min(h, key=lambda k: k["vs_rmse"])
        rows.append(dict(label=label, iters=h[-1]["iteration"], final_rmse=h[-1]["vs_rmse"],
                         stop_it=s, stop_rmse=h[s]["vs_rmse"], stop_inside=h[s]["vs_mean_inside"],
                         stop_outside=h[s]["vs_std_outside"], best_it=best["iteration"],
                         best_rmse=best["vs_rmse"], sec_per_it=h[-1]["elapsed_s"] / max(h[-1]["iteration"], 1)))
        vs = r["vs_at_stop"] if "vs_at_stop" in r.files else r["vs"]
        maps.append((label, vs, h[s]["vs_rmse"], s))

    print(f"{'variant':30s} {'stop it':>7s} {'RMSE@stop':>9s} {'inside@stop':>11s} {'outside std':>11s} "
          f"{'RMSE final':>10s} {'best (it)':>12s} {'s/it':>5s}")
    for r in rows:
        print(f"{r['label']:30s} {r['stop_it']:7d} {r['stop_rmse']:9.1f} {r['stop_inside']:11.0f} "
              f"{r['stop_outside']:11.1f} {r['final_rmse']:10.1f} {r['best_rmse']:7.1f} ({r['best_it']:2d}) "
              f"{r['sec_per_it']:5.1f}")
    (OUT / f"{FIG_NAME}_summary.json").write_text(json.dumps(rows, indent=2))

    L, inc = DEFN["L"], DEFN["inclusion"]
    first = np.load(OUT / f"{RUNS[0][0]}.npz")
    receivers = first["receivers_m"] if "receivers_m" in first.files else \
        np.load(REPO / "salvus_plate" / "plate_benchmark.npz")["receivers_m"][:, :2]
    events = json.loads((OUT / f"{RUNS[0][0]}.json").read_text())["args"]["events"]
    sources = np.array([DEFN["sources"][e]["position"] for e in events])
    panels = [("true vs", true_vs, None, None)] + [(l, v, e, s) for l, v, e, s in maps]
    cols = 4
    nrow = -(-len(panels) // cols)
    fig, axs = plt.subplots(nrow, cols, figsize=(3.6 * cols, 3.5 * nrow), layout="constrained")
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (label, vs, rmse, s) in zip(axs.ravel(), panels):
        im = ax.imshow(vs.T, origin="lower", extent=(0, L, 0, L), cmap="viridis", vmin=1050, vmax=1380)
        ax.add_patch(Rectangle((inc["centre"][0] - inc["side"] / 2, inc["centre"][1] - inc["side"] / 2),
                               inc["side"], inc["side"], fill=False, ec="white", lw=0.8, ls="--"))
        if rmse is None:
            ax.scatter(receivers[:, 0], receivers[:, 1], s=0.6 if len(receivers) > 500 else 8,
                       c="#ffffff", alpha=0.6 if len(receivers) > 500 else 1.0, linewidths=0,
                       label=f"{len(receivers)} receivers")
            ax.scatter(sources[:, 0], sources[:, 1], marker="x", s=40, c="#d62728", linewidths=1.5,
                       label=f"{len(sources)} sources")
            ax.legend(fontsize=7, loc="lower left", framealpha=0.8)
        ax.set_title(label if rmse is None else f"{label}\nearly stop it {s}: vs RMSE {rmse:.1f} m/s",
                     fontsize=8, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.6, label="vs (m/s)")
    fig.savefig(OUT / f"{FIG_NAME}.png", dpi=110)
    print(OUT / f"{FIG_NAME}.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:  # usage: plot_regularisation.py <figure name> tag=label ...
        FIG_NAME = sys.argv[1]
        RUNS = [tuple(a.split("=", 1)) for a in sys.argv[2:]]
    main()
