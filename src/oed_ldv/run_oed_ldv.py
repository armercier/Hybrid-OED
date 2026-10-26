"""Online OED for the LDV plate: design step -> lab acquisition -> K Mindlin FWI updates.

Mirrors the loop of the workshop paper (Fig. 1) for the LDV experiment: one transducer,
a sparse set of LDV points moved continuously inside the scan area, a blind log-mass map
as the unknown (the parameter that located the cube in src/ldv_inversion), and a lab
that re-measures at the new points every round.

  --lab mindlin|salvus|replay calibrated Mindlin truth (fast), Salvus 3-D cube + Q 40, or the real
                              dense LDV scan (nearest measured trace, FWI at the snapped points)
  --design illumination|none  stochastic illumination of the mass map, or the fixed layout

Metrics per round: cube localisation error (mass-map peak), cube contrast, data misfit;
at the end, the full-field prediction error: the final model, with its source filter
fitted on the measured points, predicts the dense scan (truth from the lab).

Usage:  python src/oed_ldv/run_oed_ldv.py --lab salvus --design illumination --tag salvus_illum
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# The Salvus lab runs on the same GPU in a subprocess: do not let JAX pre-allocate 75 % of it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from problem import REPO, OEDProblem

import jax
import jax.numpy as jnp
import optax

OUT = REPO / "oed_ldv_results"


def initial_layout(n=30):
    """Line along the top of the scan, away from the source and the cube."""
    return np.stack([np.linspace(-0.05, 0.15, n), np.full(n, 0.04)], 1)


def dense_prediction_error(fwi, log_mass, rec, obs, dense_P, dense_obs, chunk=2500):
    """Final model + source filter fitted on the measured points -> dense field error (r > 4 cm)."""
    sim = jax.jit(fwi.simulate)
    syn_meas = np.asarray(sim(log_mass, jnp.asarray(rec, jnp.float32)))
    S = np.fft.rfft(syn_meas, fwi.nf, axis=0) * np.asarray(fwi.H2)
    O = np.fft.rfft(obs, fwi.nf, axis=0)
    num, den = np.sum(O * np.conj(S), 1), np.sum(np.abs(S) ** 2, 1)
    F = num / (den + 1e-3 * den.max())
    keep = np.hypot(*(dense_P - fwi.src).T) > fwi.near_source
    P, D = dense_P[keep], dense_obs[:, keep]
    err_num = err_den = 0.0
    for s in range(0, len(P), chunk):
        syn = np.asarray(sim(log_mass, jnp.asarray(P[s:s + chunk], jnp.float32)))
        pred = np.fft.irfft(F[:, None] * np.fft.rfft(syn, fwi.nf, axis=0) * np.asarray(fwi.H2), fwi.nf, axis=0)[: len(fwi.t)]
        err_num += np.sum((pred - D[:, s:s + chunk]) ** 2)
        err_den += np.sum(D[:, s:s + chunk] ** 2)
    return float(np.sqrt(err_num / err_den))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", choices=("mindlin", "salvus", "replay"), default="mindlin")
    parser.add_argument("--design", choices=("illumination", "waveform", "none"), required=True)
    parser.add_argument("--design-K", type=int, default=5, help="unrolled FWI updates in the waveform objective")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fwi-per-round", type=int, default=10)
    parser.add_argument("--fwi-lr", type=float, default=0.05)
    parser.add_argument("--design-lr", type=float, default=0.01, help="Adam step on LDV point coordinates (m)")
    parser.add_argument("--design-dx", type=float, default=1e-3)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--n-points", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--order", choices=("acquire-first", "design-first"), default="acquire-first",
                        help="per round: acquire -> FWI -> design (lab order, default) or design -> acquire -> FWI "
                             "(workshop paper)")
    parser.add_argument("--accumulate", action="store_true",
                        help="keep all acquisitions: FWI on every point measured so far (one source filter per "
                             "acquisition); illumination of new points evaluated together with the fixed old ones")
    args = parser.parse_args()
    run_dir = OUT / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    from labs import MindlinLab, ReplayLab, SalvusLab

    lab = {"mindlin": MindlinLab, "salvus": lambda: SalvusLab(run_dir), "replay": ReplayLab}[args.lab]()
    # points actually measured (the replay lab snaps to the scan grid); the design stays continuous
    snap = (lambda r: jnp.asarray(lab.snap(np.asarray(r)), jnp.float32)) if hasattr(lab, "snap") else (lambda r: r)
    fwi = OEDProblem()
    design = OEDProblem(dx=args.design_dx, checkpoint_every=64 if args.design == "waveform" else 0)
    assert design.n_model == fwi.n_model

    fwi_grad = jax.jit(jax.value_and_grad(fwi.fwi_loss))
    illum = jax.jit(jax.value_and_grad(design.illumination))
    illum_with_old = jax.jit(jax.value_and_grad(
        lambda new, old, m, q: design.illumination(jnp.concatenate([new, old]), m, q)))
    wave = jax.jit(jax.value_and_grad(design.waveform_objective(args.design_K, args.fwi_lr)))
    rec = fwi.project(jnp.asarray(initial_layout(args.n_points), jnp.float32))
    probes = design.make_probes(jax.random.PRNGKey(args.seed), args.probes, args.n_points)
    log_mass = jnp.zeros(fwi.n_model, jnp.float32)
    opt_m, opt_r = optax.adam(args.fwi_lr), optax.adam(args.design_lr)
    state_m, state_r = opt_m.init(log_mass), opt_r.init(rec)

    log = dict(args=vars(args), cube_true=lab.cube.tolist(), trajectory=[np.asarray(rec).tolist()], rounds=[])
    obs = None
    acquired = []  # (points, data) of every acquisition, used with --accumulate
    fwi_grad_groups = jax.jit(jax.value_and_grad(fwi.fwi_loss_groups), static_argnums=4)
    started = time.perf_counter()
    def design_step(rnd, rec, state_r, log_mass, obs, entry):
        """Move the LDV points with the design objective (current model, current points)."""
        t0 = time.perf_counter()
        if args.design == "illumination" and args.accumulate and acquired:
            old = jnp.asarray(np.concatenate([a for a, _ in acquired]), jnp.float32)
            probes_all = design.make_probes(jax.random.PRNGKey(args.seed + 1000 * rnd), args.probes,
                                            args.n_points + old.shape[0])
            value, g = illum_with_old(rec, old, log_mass, probes_all)
        elif args.design == "illumination":
            value, g = illum(rec, log_mass, probes)
        else:  # waveform: needs the data at the current points (none before the first acquisition)
            d0 = obs if obs is not None else lab.acquire(np.asarray(rec), f"{args.tag}_pre")[0]
            value, g = wave(rec, log_mass, rec, jnp.asarray(d0))
        upd, state_r = opt_r.update(g, state_r)
        rec = fwi.project(optax.apply_updates(rec, upd))
        entry.update(objective=float(value), design_s=time.perf_counter() - t0,
                     max_move_mm=float(1e3 * jnp.max(jnp.linalg.norm(upd, axis=1))))
        log["trajectory"].append(np.asarray(rec).tolist())
        return rec, state_r

    for rnd in range(args.rounds):
        entry = dict(round=rnd)
        if args.design != "none" and args.order == "design-first":
            rec, state_r = design_step(rnd, rec, state_r, log_mass, obs, entry)
        if obs is None or args.design != "none":
            obs, secs = lab.acquire(np.asarray(rec), f"{args.tag}_r{rnd}")
            rec_meas = snap(rec)
            entry["acquire_s"] = secs
            entry["snap_max_mm"] = float(1e3 * jnp.max(jnp.linalg.norm(rec_meas - rec, axis=1)))
        if args.accumulate:
            acquired.append((np.asarray(rec_meas), obs))
            rec_all = jnp.asarray(np.concatenate([a for a, _ in acquired]), jnp.float32)
            obs_all = jnp.asarray(np.concatenate([d for _, d in acquired], axis=1))
            groups = jnp.asarray(np.repeat(np.arange(len(acquired)), [len(a) for a, _ in acquired]))
            entry["n_points_total"] = int(rec_all.shape[0])
        obs_j = jnp.asarray(obs)
        for _ in range(args.fwi_per_round):
            if args.accumulate:
                loss, g = fwi_grad_groups(log_mass, rec_all, obs_all, groups, len(acquired))
            else:
                loss, g = fwi_grad(log_mass, rec_meas, obs_j)
            upd, state_m = opt_m.update(g, state_m)
            log_mass = optax.apply_updates(log_mass, upd)
        entry.update(loss=float(loss), **fwi.cube_metrics(log_mass, lab.cube), t=time.perf_counter() - started)
        log["rounds"].append(entry)
        measured_rec = np.asarray(rec_meas)  # points measured this round (before any design step)
        if args.design != "none" and args.order == "acquire-first" and rnd < args.rounds - 1:
            rec, state_r = design_step(rnd, rec, state_r, log_mass, obs, entry)
        print(f"round {rnd}: loss {entry['loss']:.4f} | cube loc. error {entry['loc_error_mm']:.1f} mm, "
              f"contrast {entry['contrast']:.2f}, peak mass x{entry['peak_mass']:.2f}"
              + (f" | design objective {entry['objective']:.3f}, max move {entry['max_move_mm']:.1f} mm" if "objective" in entry else "")
              + f" | t={entry['t']:.0f}s", flush=True)

    dense_P, dense_obs = lab.dense()
    fit_rec, fit_obs = measured_rec, obs
    if args.accumulate:  # source filter for the dense prediction fitted on every measured point
        fit_rec, fit_obs = np.asarray(rec_all), np.asarray(obs_all)
    log["dense_prediction_error"] = dense_prediction_error(fwi, log_mass, fit_rec, fit_obs, dense_P, dense_obs)
    log["dense_prediction_error_homogeneous"] = dense_prediction_error(
        fwi, jnp.zeros_like(log_mass), fit_rec, fit_obs, dense_P, dense_obs)
    log["total_s"] = time.perf_counter() - started
    print(f"dense-field prediction error: {log['dense_prediction_error']:.3f} "
          f"(plain plate with the same filter: {log['dense_prediction_error_homogeneous']:.3f}); total {log['total_s']:.0f}s")
    np.savez_compressed(run_dir / "result.npz", log_mass=np.asarray(log_mass), Xc=fwi.Xc, Yc=fwi.Yc,
                        trajectory=np.asarray(log["trajectory"]), src=fwi.src)
    (run_dir / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
