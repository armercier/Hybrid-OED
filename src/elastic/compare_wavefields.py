"""Visual and quantitative comparison: Salvus 3-D vs JAX 3-D vs JAX plane stress.

All solvers are sampled at the Salvus benchmark receivers (59 x 59 grid, 10 mm,
top surface), so snapshots and numbers compare identical quantities. Each JAX
result is scaled by one amplitude factor per event (force units differ in 2-D).

Outputs (elastic_results/wavefields/):
  snapshots_<event>.png   top-surface v_y and v_z snapshots, total and scattered field
  traces_<event>.png      traces at a few receivers
  metrics.json            misfits per solver, field, component and time window

Usage:  python src/elastic/compare_wavefields.py --events 1
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
OUT = REPO / "elastic_results" / "wavefields"
MODELS = ("plate_homogeneous", "plate_inclusion")
WINDOWS_US = (300, 500, 1001)


def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def resample(tr, t_sim, t_obs):
    return np.stack([np.stack([np.interp(t_obs, t_sim, tr[:, r, c]) for r in range(tr.shape[1])], 1)
                     for c in range(tr.shape[2])], -1)


def run_solvers(defn, data, event, dx3d=1e-3, dz3d=1e-3, dx2d=1e-3):
    import jax
    import jax.numpy as jnp

    from compare_plate import material_grid
    from elastic3d import plate3d_free
    from plane_stress import plane_stress_free

    t_obs, rec = data["t"], data["receivers_m"][:, :2]
    L, H, f0 = defn["L"], defn["H"], defn["f0"]
    src = defn["sources"][event]
    out, timing = {}, {}

    # plane stress, with and without the S0 correction
    n = int(round(L / dx2d)) + 1
    dt = 0.6 * dx2d / (defn["pmma"]["vp"] * np.sqrt(2.0))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 1) * dt
    stf = jnp.asarray(ricker(t_sim, f0), jnp.float32)
    ps = jax.jit(lambda m, s, f: plane_stress_free(m[0], m[1], m[2], s, f, stf, dx2d, dx2d, dt, jnp.asarray(rec / dx2d)))
    for corr, label in ((False, "plane_stress"), (True, "plane_stress_s0corr")):
        for model in MODELS:
            m = tuple(jnp.asarray(a) for a in material_grid(defn, n, dx2d, model == "plate_inclusion", corr))
            t0 = time.perf_counter()
            tr = np.asarray(jax.block_until_ready(ps(m, jnp.asarray(np.asarray(src["position"]) / dx2d, jnp.float32),
                                                     jnp.asarray(src["force"], jnp.float32))))
            timing[label] = time.perf_counter() - t0
            out[(label, model)] = resample(tr, t_sim, t_obs)

    # full 3-D elastic
    n = int(round(L / dx3d)) + 1
    nz = int(round(H / dz3d)) + 1
    dt = 0.6 / (defn["pmma"]["vp"] * np.sqrt(2 / dx3d**2 + 1 / dz3d**2))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 1) * dt
    stf = jnp.asarray(ricker(t_sim, f0), jnp.float32)
    si = jnp.asarray([src["position"][0] / dx3d, src["position"][1] / dx3d, src["z"] / dz3d], jnp.float32)
    f = jnp.asarray(list(src["force"]) + [0.0], jnp.float32)
    p3 = jax.jit(lambda m: plate3d_free(m[0], m[1], m[2], nz, si, f, stf, dx3d, dz3d, dt, jnp.asarray(rec / dx3d)))
    for model in MODELS:
        m = tuple(jnp.asarray(a) for a in material_grid(defn, n, dx3d, model == "plate_inclusion"))
        t0 = time.perf_counter()
        tr = np.asarray(jax.block_until_ready(p3(m)))
        timing["jax_3d"] = time.perf_counter() - t0
        out[("jax_3d", model)] = resample(tr, t_sim, t_obs)

    # one amplitude factor per solver and event, fitted on the homogeneous in-plane data
    obs = {model: data[f"{model}__{src['event']}"] for model in MODELS}
    for label in ("plane_stress", "plane_stress_s0corr", "jax_3d"):
        p, o = out[(label, MODELS[0])][..., :2], obs[MODELS[0]][..., :2]
        scale = np.sum(p * o) / np.sum(p * p)
        for model in MODELS:
            out[(label, model)] = out[(label, model)] * scale
    return out, obs, timing


def misfits(out, obs, t_obs):
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    res = {}
    for label in ("plane_stress", "plane_stress_s0corr", "jax_3d"):
        r = {}
        for tmax in WINDOWS_US:
            w = t_obs < tmax * 1e-6
            pi, oi = out[(label, "plate_inclusion")][w], obs["plate_inclusion"][w]
            ps, os_ = pi - out[(label, "plate_homogeneous")][w], oi - obs["plate_homogeneous"][w]
            r[f"t<{tmax}us"] = {
                "total_inplane": rel(pi[..., :2], oi[..., :2]),
                "scattered_inplane": rel(ps[..., :2], os_[..., :2]),
                "total_vz": rel(pi[..., 2], oi[..., 2]) if label == "jax_3d" else None,
                "scattered_vz": rel(ps[..., 2], os_[..., 2]) if label == "jax_3d" else None,
            }
        res[label] = r
    return res


def plot_snapshots(out, obs, t_obs, rec, defn, event, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    times_us = (100, 200, 350, 600)
    n = int(round(np.sqrt(len(rec))))
    ext = (rec[:, 0].min(), rec[:, 0].max(), rec[:, 1].min(), rec[:, 1].max())
    comp = 1 if defn["sources"][event]["force"][1] else 0
    rows = [
        ("Salvus 3-D", lambda m: obs[m]),
        ("JAX 3-D", lambda m: out[("jax_3d", m)]),
        ("JAX plane stress + S0 corr.", lambda m: out[("plane_stress_s0corr", m)]),
    ]
    blocks = [("total field, v_" + "xy"[comp], lambda f: f("plate_inclusion")[..., comp]),
              ("scattered field (inclusion − homogeneous), v_" + "xy"[comp],
               lambda f: f("plate_inclusion")[..., comp] - f("plate_homogeneous")[..., comp]),
              ("total field, v_z (out-of-plane, top surface)", lambda f: f("plate_inclusion")[..., 2])]
    fig, axs = plt.subplots(len(blocks) * len(rows), len(times_us), figsize=(3.0 * len(times_us), 2.7 * 9),
                            layout="constrained")
    inc, src = defn["inclusion"], defn["sources"][event]["position"]
    for b, (btitle, getter) in enumerate(blocks):
        ref = getter(rows[0][1])
        for c, tu in enumerate(times_us):
            k = int(np.argmin(np.abs(t_obs - tu * 1e-6)))
            vmax = np.max(np.abs(ref[k])) or 1.0
            for r, (rtitle, f) in enumerate(rows):
                ax = axs[b * len(rows) + r, c]
                if b == 2 and r == 2:
                    ax.text(0.5, 0.5, "plane stress has\nno v_z", ha="center", va="center", transform=ax.transAxes,
                            color="#6b6b6b", fontsize=9)
                    ax.set_xticks([]), ax.set_yticks([])
                    continue
                img = getter(f)[k].reshape(n, n)
                ax.imshow(img.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="bilinear")
                ax.add_patch(Rectangle((inc["centre"][0] - inc["side"] / 2, inc["centre"][1] - inc["side"] / 2),
                                       inc["side"], inc["side"], fill=False, ec="#1f1f1f", lw=0.8, ls="--"))
                ax.plot(*src, marker="x", color="#1f1f1f", ms=5)
                ax.set_xticks([]), ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(rtitle, fontsize=8)
                if r == 0:
                    ax.set_title(f"{btitle}\nt = {tu} µs" if c == 0 else f"t = {tu} µs", fontsize=8, loc="left")
    fig.suptitle(f"{defn['sources'][event]['event']}: top-surface particle velocity on the 10 mm receiver grid "
                 "(colour scale per column = Salvus maximum)", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_traces(out, obs, t_obs, rec, event, defn, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = np.asarray(defn["sources"][event]["position"])
    picks = [(0.25, 0.30), (0.40, 0.35), (0.55, 0.30), (0.10, 0.50)]
    styles = [("Salvus 3-D", None, "#1f1f1f", 1.8), ("JAX 3-D", "jax_3d", "#c2571a", 1.1),
              ("JAX plane stress + S0 corr.", "plane_stress_s0corr", "#2a6fdb", 1.1)]
    fig, axs = plt.subplots(len(picks), 2, figsize=(13, 2.4 * len(picks)), layout="constrained")
    t_us = t_obs * 1e6
    for row, (x, y) in enumerate(picks):
        r = int(np.argmin(np.hypot(rec[:, 0] - x, rec[:, 1] - y)))
        for col, (field, title) in enumerate((("total", "total v_y"), ("scattered", "scattered v_y"))):
            ax = axs[row, col]
            for label, key, colour, lw in styles:
                get = (lambda m: obs[m]) if key is None else (lambda m, key=key: out[(key, m)])
                trace = get("plate_inclusion")[:, r, 1]
                if field == "scattered":
                    trace = trace - get("plate_homogeneous")[:, r, 1]
                ax.plot(t_us, trace, color=colour, lw=lw, label=label)
            ax.set_title(f"{title} at ({rec[r, 0]:.2f}, {rec[r, 1]:.2f}) m, "
                         f"{np.hypot(*(rec[r] - src)):.2f} m from source", fontsize=9, loc="left")
            ax.grid(alpha=0.3)
            ax.set_xlim(-60, 1000)
    axs[0, 0].legend(fontsize=8, frameon=False)
    axs[-1, 0].set_xlabel("t (µs)"), axs[-1, 1].set_xlabel("t (µs)")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    defn = json.loads((BENCH / "benchmark_definition.json").read_text())
    data = np.load(BENCH / "plate_benchmark.npz")
    t_obs, rec = data["t"], data["receivers_m"][:, :2]
    report = {}
    for e in args.events:
        out, obs, timing = run_solvers(defn, data, e)
        name = defn["sources"][e]["event"]
        report[name] = {"misfits": misfits(out, obs, t_obs), "solve_seconds": timing}
        plot_snapshots(out, obs, t_obs, rec, defn, e, OUT / f"snapshots_{name}.png")
        plot_traces(out, obs, t_obs, rec, e, defn, OUT / f"traces_{name}.png")
        print(json.dumps(report[name], indent=1), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
