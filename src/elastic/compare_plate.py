"""Plane-stress (free-edge) JAX solver vs the Salvus 3-D plate benchmark.

Reads salvus_plate/plate_benchmark.npz and benchmark_definition.json (written by
src/salvus_plate/), runs the JAX plane-stress solver for the same plate, sources
and wavelet, and scores in-plane velocities at all receivers:

  * total field, homogeneous and with the inclusion;
  * scattered field (inclusion - homogeneous), which isolates the inclusion physics.

Misfit = relative L2 error over all receivers and samples, and the FWI-style
version with every trace normalised by its own maximum.

Usage:  python src/elastic/compare_plate.py --dx 1e-3 [--events 0 1 ...]
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
RESULTS = REPO / "elastic_results" / "plate"


def ricker(t, f0):
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def material_grid(defn, n, dx, with_inclusion, s0_correction=False):
    """Nodal (vp, vs, rho); edge nodes of the inclusion get the cell-area fraction.

    With ``s0_correction``, each material's vp is replaced by the value whose
    plane-stress plate velocity equals the exact S0 phase velocity at f0 (vs kept).
    """
    if s0_correction:
        from plane_stress import s0_corrected_vp

        defn = json.loads(json.dumps(defn))
        for m in (defn["pmma"], defn["inclusion"]):
            m["vp"] = s0_corrected_vp(m["vp"], m["vs"], defn["H"], defn["f0"])
    vals = {k: np.full((n, n), defn["pmma"][k], np.float32) for k in ("vp", "vs", "rho")}
    if not with_inclusion:
        return vals["vp"], vals["vs"], vals["rho"]
    inc = defn["inclusion"]
    sub = (np.arange(8) + 0.5) / 8 - 0.5  # 8x8 sub-samples of each node's dual cell
    x = np.arange(n) * dx
    frac = np.zeros((n, n))
    for a in sub:
        for b in sub:
            X, Y = np.meshgrid(x + a * dx, x + b * dx, indexing="ij")
            frac += (np.abs(X - inc["centre"][0]) < inc["side"] / 2) & (
                np.abs(Y - inc["centre"][1]) < inc["side"] / 2)
    frac /= sub.size**2
    # mix the moduli (not the velocities) by area fraction
    rho = defn["pmma"]["rho"] + frac * (inc["rho"] - defn["pmma"]["rho"])
    mu = (1 - frac) * defn["pmma"]["rho"] * defn["pmma"]["vs"] ** 2 + frac * inc["rho"] * inc["vs"] ** 2
    m = (1 - frac) * defn["pmma"]["rho"] * defn["pmma"]["vp"] ** 2 + frac * inc["rho"] * inc["vp"] ** 2
    return (np.sqrt(m / rho).astype(np.float32), np.sqrt(mu / rho).astype(np.float32),
            rho.astype(np.float32))


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rel_tracenorm(a, b):
    n = lambda z: z / np.maximum(np.max(np.abs(z), axis=0, keepdims=True), 1e-30)
    return rel(n(a), n(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=1e-3)
    parser.add_argument("--cfl", type=float, default=0.6)
    parser.add_argument("--events", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--s0-correction", action="store_true",
                        help="effective vp matching the exact S0 phase velocity at f0")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from plane_stress import plane_stress_free

    defn = json.loads((BENCH / "benchmark_definition.json").read_text())
    data = np.load(BENCH / "plate_benchmark.npz")
    t_obs, rec = data["t"], data["receivers_m"][:, :2]
    L = defn["L"]
    n = int(round(L / args.dx)) + 1
    dx = L / (n - 1)
    dt = args.cfl * dx / (defn["pmma"]["vp"] * np.sqrt(2.0))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 1) * dt
    stf = jnp.asarray(ricker(t_sim, defn["f0"]), jnp.float32)
    print(f"device={jax.devices()[0]} grid {n}x{n} dx={dx*1e3:.2f} mm dt={dt*1e6:.3f} us steps={len(t_sim)}")

    models = {name: tuple(jnp.asarray(a) for a in material_grid(defn, n, dx, name == "plate_inclusion",
                                                                 args.s0_correction))
              for name in ("plate_homogeneous", "plate_inclusion")}
    run = jax.jit(lambda m, s, f: plane_stress_free(m[0], m[1], m[2], s, f, stf, dx, dx, dt,
                                                    jnp.asarray(rec / dx)))

    results, saved = [], {}
    for e in args.events:
        src = defn["sources"][e]
        s_i = jnp.asarray(np.asarray(src["position"]) / dx, jnp.float32)
        f = jnp.asarray(src["force"], jnp.float32)
        pred, obs = {}, {}
        for name, m in models.items():
            t0 = time.perf_counter()
            tr = np.asarray(jax.block_until_ready(run(m, s_i, f)))
            solve_s = time.perf_counter() - t0
            p = np.stack([np.stack([np.interp(t_obs, t_sim, tr[:, r, c]) for r in range(tr.shape[1])], 1)
                          for c in range(2)], -1)
            o = data[f"{name}__{src['event']}"][:, :, :2]
            scale = np.sum(p * o) / np.sum(p * p)  # one amplitude factor per event (force units)
            pred[name], obs[name] = scale * p, o
        rec_ = {"event": src["event"], "solve_s": solve_s}
        for name in models:
            rec_[name] = dict(misfit=rel(pred[name], obs[name]),
                              misfit_tracenorm=rel_tracenorm(pred[name].reshape(len(t_obs), -1),
                                                             obs[name].reshape(len(t_obs), -1)))
        sp = pred["plate_inclusion"] - pred["plate_homogeneous"]
        so = obs["plate_inclusion"] - obs["plate_homogeneous"]
        rec_["scattered"] = dict(misfit=rel(sp, so),
                                 energy_ratio_obs=float(np.linalg.norm(so) / np.linalg.norm(obs["plate_inclusion"])))
        results.append(rec_)
        saved.update({f"{src['event']}__{k}": v.astype(np.float32) for k, v in pred.items()})
        print(f"{src['event']}: homog {rec_['plate_homogeneous']['misfit']:.3f} "
              f"(trace-norm {rec_['plate_homogeneous']['misfit_tracenorm']:.3f}) | "
              f"inclusion {rec_['plate_inclusion']['misfit']:.3f} "
              f"(trace-norm {rec_['plate_inclusion']['misfit_tracenorm']:.3f}) | "
              f"scattered {rec_['scattered']['misfit']:.3f} "
              f"(scattered/total in Salvus {rec_['scattered']['energy_ratio_obs']:.3f}) | {solve_s:.1f} s",
              flush=True)

    tag = f"dx{dx*1e3:.2f}mm" + ("_s0corr" if args.s0_correction else "")
    np.savez_compressed(RESULTS / f"jax_{tag}.npz", t=t_obs, receivers_m=rec, **saved)
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps({"dx": dx, "dt": dt, "results": results}, indent=2))


if __name__ == "__main__":
    main()
