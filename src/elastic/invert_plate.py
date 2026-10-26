"""FWI of the Salvus 3-D plate benchmark with the JAX plane-stress solver (GPU).

* Observed data: Salvus ``plate_inclusion`` in-plane velocity (vx, vy) at all
  top-surface receivers, for the given events, in the window t < --tmax.
  ``--data plane_stress`` instead inverts synthetic data from the plane-stress solver
  itself (inverse-crime control, same receivers and window).
* Forward model: ``plane_stress_free`` (free edges, S0-corrected vp), remat in time chunks.
* Unknown: vs on a coarse grid (--param-spacing) bilinearly interpolated to the FD grid;
  vp = vs * (corrected PMMA vp/vs ratio), rho fixed. Start: homogeneous PMMA.
  The update is masked within --source-mask of every source and optionally tapered
  to zero over --edge-taper from the plate edges.
* Optional total-variation regularisation (--tv) of the relative vs update.
* Early-stopping rule (data only): stop when the misfit decreased by less than
  --stop-tol (relative) over the last --stop-window iterations. The run continues
  to --iterations so the rule can be evaluated; the model at the stop is saved.
* Source amplitude: one scale per event solved in closed form (variable projection).

Usage:  python src/elastic/invert_plate.py --iterations 60 [--data plane_stress]
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
OUT = REPO / "elastic_results" / "inversion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, nargs="+", default=[1, 4, 7, 10])
    parser.add_argument("--data", choices=("salvus", "plane_stress"), default="salvus")
    parser.add_argument("--tmax", type=float, default=500e-6)
    parser.add_argument("--dx", type=float, default=1e-3)
    parser.add_argument("--param-spacing", type=float, default=5e-3)
    parser.add_argument("--source-mask", type=float, default=0.03)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--vs-bounds", type=float, nargs=2, default=(800.0, 1600.0))
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--edge-taper", type=float, default=0.0)
    parser.add_argument("--tv", type=float, default=0.0)
    parser.add_argument("--tv-eps", type=float, default=1e-3)
    parser.add_argument("--stop-window", type=int, default=5)
    parser.add_argument("--stop-tol", type=float, default=0.02)
    parser.add_argument("--receivers", choices=("all", "diagonal"), default="all",
                        help="all 59x59 grid receivers, or the paper's diagonal from (0.15, 0.15) to (0.45, 0.45) m")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import optax
    from jax.scipy.ndimage import map_coordinates

    from compare_plate import material_grid
    from plane_stress import plane_stress_free, s0_corrected_vp

    defn = json.loads((BENCH / "benchmark_definition.json").read_text())
    data = np.load(BENCH / "plate_benchmark.npz")
    L, H, f0 = defn["L"], defn["H"], defn["f0"]
    t_all, rec = data["t"], data["receivers_m"][:, :2]
    if args.receivers == "diagonal":
        rec_sel = np.where((np.abs(rec[:, 0] - rec[:, 1]) < 1e-6) & (rec[:, 0] > 0.15 - 1e-6) & (rec[:, 0] < 0.45 + 1e-6))[0]
    else:
        rec_sel = np.arange(len(rec))
    rec = rec[rec_sel]
    print(f"{len(rec_sel)} receivers", flush=True)
    win = t_all < args.tmax
    t_obs = t_all[win]

    n = int(round(L / args.dx)) + 1
    dx = L / (n - 1)
    pm = defn["pmma"]
    ratio = s0_corrected_vp(pm["vp"], pm["vs"], H, f0) / pm["vs"]
    vp_max = args.vs_bounds[1] * ratio
    dt = 0.6 * dx / (vp_max * np.sqrt(2.0))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 2) * dt
    a = (np.pi * f0 * t_sim) ** 2
    stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
    # linear interpolation from simulation to observation times (differentiable)
    pos = (t_obs - t_sim[0]) / dt
    i0 = np.floor(pos).astype(int)
    w = jnp.asarray(pos - i0, jnp.float32)[:, None, None]
    i0 = jnp.asarray(i0)

    # coarse vs parameterisation, masked near the sources
    nc = int(round(L / args.param_spacing)) + 1
    I, J = np.meshgrid(np.arange(n) * dx / args.param_spacing, np.arange(n) * dx / args.param_spacing, indexing="ij")
    coords = [jnp.asarray(I, jnp.float32), jnp.asarray(J, jnp.float32)]
    Xc, Yc = np.meshgrid(np.arange(nc) * args.param_spacing, np.arange(nc) * args.param_spacing, indexing="ij")
    mask = np.ones((nc, nc), np.float32)
    for e in args.events:
        sx, sy = defn["sources"][e]["position"]
        mask[np.hypot(Xc - sx, Yc - sy) < args.source_mask] = 0.0
    if args.edge_taper > 0:
        d_edge = np.minimum.reduce([Xc, L - Xc, Yc, L - Yc])
        mask *= np.where(d_edge < args.edge_taper, 0.5 - 0.5 * np.cos(np.pi * d_edge / args.edge_taper), 1.0)
    mask = jnp.asarray(mask)

    def vs_map(delta):
        rel = 1.0 + mask * delta
        vs = pm["vs"] * map_coordinates(rel, coords, order=1)
        return jnp.clip(vs, *args.vs_bounds)

    rho = jnp.full((n, n), pm["rho"], jnp.float32)
    rec_i = jnp.asarray(rec / dx, jnp.float32)

    def simulate(vs, e):
        src = defn["sources"][e]
        tr = plane_stress_free(vs * ratio, vs, rho, jnp.asarray(np.asarray(src["position"]) / dx, jnp.float32),
                               jnp.asarray(src["force"], jnp.float32), stf, dx, dx, dt, rec_i,
                               checkpoint_every=args.checkpoint_every)
        return (1 - w) * tr[i0] + w * tr[i0 + 1]  # (nt_obs, nrec, 2)

    true_vs = np.asarray(material_grid(defn, n, dx, True)[1])
    if args.data == "salvus":
        observed = {e: jnp.asarray(data[f"plate_inclusion__{defn['sources'][e]['event']}"][win][:, rec_sel, :2])
                    for e in args.events}
    else:
        true_vs_sim = jnp.asarray(np.clip(true_vs, *args.vs_bounds))
        observed = {e: jax.jit(simulate, static_argnums=1)(true_vs_sim, e) for e in args.events}

    def event_loss(delta, e):
        p, o = simulate(vs_map(delta), e), observed[e]
        scale = jax.lax.stop_gradient(jnp.sum(p * o) / jnp.sum(p * p))
        return jnp.sum((scale * p - o) ** 2) / jnp.sum(o**2)

    grad_fns = {e: jax.jit(jax.value_and_grad(lambda d, e=e: event_loss(d, e))) for e in args.events}  # e static

    def tv(delta):
        m = mask * delta
        gx, gy = m[1:, :-1] - m[:-1, :-1], m[:-1, 1:] - m[:-1, :-1]
        return jnp.sum(jnp.sqrt(gx**2 + gy**2 + args.tv_eps**2) - args.tv_eps)

    tv_grad = jax.jit(jax.value_and_grad(tv))
    opt = optax.adam(args.lr)
    delta = jnp.zeros((nc, nc), jnp.float32)
    state = opt.init(delta)

    inside = np.abs(true_vs - pm["vs"]) > 1.0
    def model_metrics(vs):
        vs = np.asarray(vs)
        return dict(vs_rmse=float(np.sqrt(np.mean((vs - true_vs) ** 2))),
                    vs_rmse_initial=float(np.sqrt(np.mean((pm["vs"] - true_vs) ** 2))),
                    vs_mean_inside=float(vs[inside].mean()), vs_true_inside=float(true_vs[inside].mean()),
                    vs_std_outside=float(vs[~inside].std()))

    history, stop = [], None
    t_start = time.perf_counter()
    for it in range(args.iterations + 1):
        total, grad = 0.0, jnp.zeros_like(delta)
        for e, fn in grad_fns.items():
            l, g = fn(delta)
            total, grad = total + float(l), grad + g
        data_loss = total / len(grad_fns)
        grad = grad / len(grad_fns)
        reg = 0.0
        if args.tv > 0:
            r, gr = tv_grad(delta)
            reg, grad = float(args.tv * r), grad + args.tv * gr
        rec_ = dict(iteration=it, loss=data_loss, reg=reg, elapsed_s=time.perf_counter() - t_start,
                    **model_metrics(vs_map(delta)))
        history.append(rec_)
        if stop is None and it >= args.stop_window:
            prev = history[it - args.stop_window]["loss"]
            if (prev - data_loss) / prev < args.stop_tol:
                stop = dict(iteration=it, vs=np.asarray(vs_map(delta)), **{k: rec_[k] for k in ("loss", "vs_rmse", "vs_mean_inside", "vs_std_outside")})
                print(f"    early-stopping rule triggered at iteration {it}", flush=True)
        print(f"{it:3d} loss={rec_['loss']:.4f} vs_rmse={rec_['vs_rmse']:.1f} (start {rec_['vs_rmse_initial']:.1f}) "
              f"inside mean={rec_['vs_mean_inside']:.0f} (true {rec_['vs_true_inside']:.0f}) "
              f"outside std={rec_['vs_std_outside']:.1f}  t={rec_['elapsed_s']:.0f}s", flush=True)
        if it == args.iterations:
            break
        updates, state = opt.update(grad, state)
        delta = optax.apply_updates(delta, updates)

    tag = args.tag or f"{args.data}_tmax{int(args.tmax * 1e6)}us"
    extra = {} if stop is None else {"vs_at_stop": stop["vs"]}
    np.savez_compressed(OUT / f"{tag}.npz", vs=np.asarray(vs_map(delta)), true_vs=true_vs, dx=dx,
                        delta=np.asarray(delta), mask=np.asarray(mask), receivers_m=rec, **extra)
    (OUT / f"{tag}.json").write_text(json.dumps({
        "args": vars(args), "dt": dt, "n_steps": len(t_sim), "history": history,
        "early_stop": None if stop is None else {k: v for k, v in stop.items() if k != "vs"}}, indent=2))


if __name__ == "__main__":
    main()
