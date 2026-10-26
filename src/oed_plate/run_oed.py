"""Receiver-design OED loop on the Salvus plate with Salvus 'lab' re-acquisition.

Per round (paper, Fig. 1): design step from the current model -> new Salvus
simulation at the new receiver positions (the 'LDV measurement') -> K FWI updates
with the plane-stress solver. Model and optimiser state carry over between rounds.

  --design illumination   receivers move by the stochastic-illumination hypergradient
  --design none           fixed layout baseline (one acquisition, same FWI budget)

Usage (hybridOED env; Salvus acquisitions run in the Salvus venv via subprocess):
  python src/oed_plate/run_oed.py --design illumination --tag illum_v1
  python src/oed_plate/run_oed.py --design none --tag fixed_v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
SALVUS_PY = "/home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python"
OUT = REPO / "oed_plate_results"


def diagonal_receivers(start=0.15, stop=0.45, spacing=0.01):
    x = np.round(np.arange(start, stop + spacing / 2, spacing), 6)
    return np.stack([x, x], 1)


def acquire(rec, events, name, run_dir):
    """Run Salvus for the true plate at these receivers (the 'lab measurement')."""
    rec_file, out_file = run_dir / f"{name}_receivers.npy", run_dir / f"{name}_data.npz"
    np.save(rec_file, np.asarray(rec, np.float64))
    t0 = time.perf_counter()
    subprocess.run([SALVUS_PY, str(HERE / "acquire_salvus.py"), "--receivers", str(rec_file),
                    "--events", *map(str, events), "--name", name, "--out", str(out_file)],
                   check=True, capture_output=True, text=True)
    return np.load(out_file), time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=("illumination", "none"), required=True)
    parser.add_argument("--tag", required=True, help="unique run name (also names the Salvus events)")
    parser.add_argument("--events", type=int, nargs="+", default=[1, 4, 7, 10])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fwi-per-round", type=int, default=20)
    parser.add_argument("--fwi-lr", type=float, default=0.01)
    parser.add_argument("--tv", type=float, default=3e-4)
    parser.add_argument("--design-lr", type=float, default=0.02, help="Adam step on receiver coordinates (m)")
    parser.add_argument("--design-dx", type=float, default=2e-3)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--probe-seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import optax

    from plate_problem import PlateProblem

    run_dir = OUT / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    t_obs = np.load(REPO / "salvus_plate" / "plate_benchmark.npz")["t"]
    fwi = PlateProblem(args.events, t_obs, tv=args.tv)
    design = PlateProblem(args.events, t_obs, dx=args.design_dx, checkpoint_every=0, tv=args.tv)

    misfit_grads = {e: jax.jit(jax.value_and_grad(lambda d, r, o, e=e: fwi.event_misfit(d, e, r, o)))
                    for e in args.events}
    tv_grad = jax.jit(jax.value_and_grad(fwi.tv))
    illum = jax.jit(jax.value_and_grad(design.illumination))

    rec = jnp.asarray(diagonal_receivers(), jnp.float32)
    probes = design.make_probes(jax.random.PRNGKey(args.probe_seed), args.probes, rec.shape[0])
    delta = jnp.zeros((fwi.nc, fwi.nc), jnp.float32)
    opt_m, opt_r = optax.adam(args.fwi_lr), optax.adam(args.design_lr)
    state_m, state_r = opt_m.init(delta), opt_r.init(rec)

    log = dict(args=vars(args), trajectory=[np.asarray(rec).tolist()], fwi=[], design=[], acquisitions=[])
    observed = None
    t_start = time.perf_counter()
    for rnd in range(args.rounds):
        if args.design == "illumination":
            t0 = time.perf_counter()
            value, g = illum(rec, delta, probes)
            updates, state_r = opt_r.update(g, state_r)
            rec = design.project(optax.apply_updates(rec, updates))
            log["design"].append(dict(round=rnd, illumination=float(value), seconds=time.perf_counter() - t0,
                                      max_move_m=float(jnp.max(jnp.linalg.norm(updates, axis=1)))))
            log["trajectory"].append(np.asarray(rec).tolist())
        if observed is None or args.design != "none":
            data, secs = acquire(rec, args.events, f"{args.tag}_r{rnd}", run_dir)
            observed = fwi.observed_from(data)
            log["acquisitions"].append(dict(round=rnd, seconds=secs))
        for k in range(args.fwi_per_round):
            total, grad = 0.0, jnp.zeros_like(delta)
            for e, fn in misfit_grads.items():
                l, gg = fn(delta, rec, observed[e])
                total, grad = total + float(l), grad + gg
            r, gr = tv_grad(delta)
            grad = grad / len(misfit_grads) + args.tv * gr
            entry = dict(round=rnd, update=rnd * args.fwi_per_round + k, misfit=total / len(misfit_grads),
                         tv=float(args.tv * r), **fwi.metrics(delta))
            log["fwi"].append(entry)
            upd, state_m = opt_m.update(grad, state_m)
            delta = optax.apply_updates(delta, upd)
        m = fwi.metrics(delta)
        print(f"round {rnd}: misfit {entry['misfit']:.4f}  vs RMSE {m['vs_rmse']:.1f}  inside {m['vs_mean_inside']:.0f}"
              f"  outside std {m['vs_std_outside']:.1f}"
              + (f"  | illumination {log['design'][-1]['illumination']:.3f}, max move {log['design'][-1]['max_move_m']*100:.1f} cm"
                 if log["design"] else "")
              + f"  | t={time.perf_counter() - t_start:.0f}s", flush=True)

    log["final"] = fwi.metrics(delta)
    log["total_seconds"] = time.perf_counter() - t_start
    np.savez_compressed(run_dir / "result.npz", vs=np.asarray(fwi.vs_map(delta)), true_vs=fwi.true_vs,
                        delta=np.asarray(delta), trajectory=np.asarray(log["trajectory"]))
    (run_dir / "log.json").write_text(json.dumps(log, indent=1))
    print("final:", log["final"], f"total {log['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
