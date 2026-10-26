"""Mindlin plate solver vs Salvus 3-D for out-of-plane forces (salvus_plate/flexural_benchmark.npz).

Scores the top-surface normal velocity vz at all receivers (one amplitude factor per
event and solver, since 2-D and 3-D force units differ), total and scattered field,
by time window, and writes a trace figure.

Usage:  python src/elastic/compare_mindlin.py --dx 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "elastic"))
BENCH = REPO / "salvus_plate"
OUT = REPO / "elastic_results" / "mindlin"
WINDOWS_US = (300, 500, 1001)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=1e-3)
    parser.add_argument("--kappa", default="a0", help="'a0' (A0 matched at f0) or a number (pi^2/12 = 0.8225)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from compare_plate import material_grid
    from mindlin import a0_matched_kappa, mindlin_plate

    defn = json.loads((BENCH / "benchmark_definition.json").read_text())
    data = np.load(BENCH / "flexural_benchmark.npz")
    t_obs, rec = data["t"], data["receivers_m"][:, :2]
    L, H = defn["L"], defn["H"]
    n = int(round(L / args.dx)) + 1
    dx = L / (n - 1)
    dt = 0.6 * dx / (defn["pmma"]["vp"] * np.sqrt(2.0))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 1) * dt
    a = (np.pi * defn["f0"] * t_sim) ** 2
    stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
    models = {m: tuple(jnp.asarray(x) for x in material_grid(defn, n, dx, m == "plate_inclusion"))
              for m in ("plate_homogeneous", "plate_inclusion")}
    pm = defn["pmma"]
    kappa = a0_matched_kappa(H, pm["vp"], pm["vs"], pm["rho"], defn["f0"]) if args.kappa == "a0" else float(args.kappa)
    print(f"kappa = {kappa:.4f}")
    run = jax.jit(lambda m, s: mindlin_plate(m[0], m[1], m[2], H, s, stf, dx, dx, dt, jnp.asarray(rec / dx),
                                             kappa=kappa))
    rel = lambda p, o: float(np.linalg.norm(p - o) / np.linalg.norm(o))

    results, saved = {}, {}
    for e in (1, 4):
        s_i = jnp.asarray(np.asarray(defn["sources"][e]["position"]) / dx, jnp.float32)
        pred = {}
        for m, mat in models.items():
            t0 = time.perf_counter()
            tr = np.asarray(jax.block_until_ready(run(mat, s_i)))[..., 0]
            secs = time.perf_counter() - t0
            pred[m] = np.stack([np.interp(t_obs, t_sim, tr[:, r]) for r in range(tr.shape[1])], 1)
        for depth in ("mid", "top"):
            name = f"flex_{depth}_e{e:02d}"
            obs = {m: data[f"{m}__{name}"][..., 2] for m in models}
            scale = np.sum(pred["plate_homogeneous"] * obs["plate_homogeneous"]) / np.sum(pred["plate_homogeneous"] ** 2)
            p = {m: scale * pred[m] for m in models}
            r = {"solve_s": secs}
            for tmax in WINDOWS_US:
                w = t_obs < tmax * 1e-6
                r[f"t<{tmax}us"] = dict(
                    total=rel(p["plate_inclusion"][w], obs["plate_inclusion"][w]),
                    scattered=rel((p["plate_inclusion"] - p["plate_homogeneous"])[w],
                                  (obs["plate_inclusion"] - obs["plate_homogeneous"])[w]))
            results[name] = r
            saved.update({f"{name}__{m}": p[m].astype(np.float32) for m in models})
            print(name, " | ".join(f"{k}: total {v['total']:.3f} scattered {v['scattered']:.3f}"
                                   for k, v in r.items() if k.startswith("t<")), f"| {secs:.1f} s", flush=True)
    tag = f"dx{dx*1e3:.2f}mm_kappa{kappa:.3f}"
    (OUT / f"summary_{tag}.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(OUT / f"mindlin_{tag}.npz", t=t_obs, receivers_m=rec, **saved)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [(0.25, 0.30), (0.45, 0.30), (0.25, 0.45), (0.55, 0.55)]
    fig, axs = plt.subplots(len(picks), 2, figsize=(13, 2.3 * len(picks)), layout="constrained")
    for row, (x, y) in enumerate(picks):
        r = int(np.argmin(np.hypot(rec[:, 0] - x, rec[:, 1] - y)))
        for col, depth in enumerate(("mid", "top")):
            name = f"flex_{depth}_e01"
            ax = axs[row, col]
            ax.plot(t_obs * 1e6, data[f"plate_inclusion__{name}"][:, r, 2], color="#1f1f1f", lw=1.8, label="Salvus 3-D")
            ax.plot(t_obs * 1e6, saved[f"{name}__plate_inclusion"][:, r], color="#2a6fdb", lw=1.1, label="JAX Mindlin")
            ax.set_title(f"v_z, force at {depth}-{'plane' if depth == 'mid' else 'surface'}, receiver ({rec[r,0]:.2f}, {rec[r,1]:.2f}) m",
                         fontsize=9, loc="left")
            ax.grid(alpha=0.3), ax.set_xlim(-60, 1000)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[-1, 0].set_xlabel("t (µs)"), axs[-1, 1].set_xlabel("t (µs)")
    fig.savefig(OUT / f"traces_{tag}.png", dpi=110)


if __name__ == "__main__":
    main()
