"""Calibrate a Mindlin-plate equivalent of the LDV experiment against Salvus 3-D, then test on LDV.

Targets (salvus_ldv_cube.npz, noise free): plate only; steel cube at (-0.020, -0.005) m;
the same with PMMA Q = 40. The cube becomes a 10 mm x 10 mm patch with scaled mass,
rotary inertia and stiffness; attenuation is a velocity damping rate.

Every comparison uses the LDV pipeline: 8-70 kHz band-pass, one deconvolution filter
(absorbs source-type and 2-D/3-D source scaling differences) estimated on training
points away from the source and the bad row, misfit on held-out points behind / before
the cube.

  A  Mindlin plate          vs Salvus plate only        (baseline: plate theory + surface source)
  B  + cube patch (grid)    vs Salvus + cube (elastic)
  C  + damping (grid)       vs Salvus + cube, Q = 40
  D  best equivalent        vs the LDV data             (Salvus 3-D reaches 0.33 / 0.33)

Usage:  python src/ldv_assessment/mindlin_equivalent.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "elastic"))
from fit_mindlin_ldv import OUT, RHO, VP_VS, load_ldv  # noqa: E402

CUBE = np.array([-0.020, -0.005])
CUBE_SIDE = 0.010
VS, H = 1340.0, 0.010
MASS_FACTOR = 1.0 + 7850.0 * 0.010 / (RHO * H)                                   # 7.65
INERTIA_FACTOR = 1.0 + 7850.0 * 0.010 * (0.010**2 / 12 + 0.010**2) / (RHO * H**3 / 12)  # ~89


def main() -> None:
    import jax
    import jax.numpy as jnp

    from mindlin import a0_matched_kappa, mindlin_plate

    P_all, t_ldv, obs_ldv, _, sos = load_ldv(650e-6)
    sal = np.load(OUT / "salvus_ldv_cube.npz")
    src = np.asarray(json.loads((OUT / "salvus_ldv_cube.json").read_text())["source"])
    idx = sal["rec_idx"]
    P = P_all[idx]
    r = np.hypot(*(P - src).T)
    usable = (r > 0.04) & (np.abs(P[:, 1] - src[1]) > 0.004)
    train = (np.random.default_rng(0).random(len(P)) < 0.5) & usable
    behind, before = ~train & usable & (P[:, 0] > -0.035), ~train & usable & (P[:, 0] <= -0.035)

    dx, margin, sponge = 0.5e-3, 0.12, 0.06
    lo = P.min(0) - margin
    nx, ny = np.round((P.max(0) + margin - lo) / dx).astype(int) + 1
    X, Y = np.meshgrid(lo[0] + np.arange(nx) * dx, lo[1] + np.arange(ny) * dx, indexing="ij")
    in_cube = (np.abs(X - CUBE[0]) <= CUBE_SIDE / 2) & (np.abs(Y - CUBE[1]) <= CUBE_SIDE / 2)
    kappa = a0_matched_kappa(H, VP_VS * VS, VS, RHO, 39e3)
    one = jnp.ones((nx, ny), jnp.float32)
    s_i = jnp.asarray((src - lo) / dx, jnp.float32)
    rec_i = jnp.asarray((P - lo) / dx, jnp.float32)
    t_sal = sal["t__cube_x-20mm"]
    t_end = 650e-6

    def run(stiff_max, **kw):
        # time step bound: the patch may raise the wave speed by sqrt(stiffness / mass or inertia)
        dt = 0.6 * dx / (VP_VS * VS * np.sqrt(2.0) * max(1.0, np.sqrt(stiff_max)))
        t_sim = t_sal[0] + np.arange(int(np.ceil((t_end - t_sal[0]) / dt)) + 2) * dt
        a = (np.pi * 45e3 * t_sim) ** 2
        stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        tr = mindlin_plate(VP_VS * VS * one, VS * one, RHO * one, H, s_i, stf, dx, dx, dt, rec_i,
                           sponge_cells=int(round(sponge / dx)), kappa=kappa, **kw)[..., 0]
        return t_sim, tr

    nf = 2 * len(t_ldv)
    last_pred = {}

    def score(t_syn, syn, t_tar, target):
        resample = lambda tt, a: np.stack([np.interp(t_ldv, tt, a[:, k], left=0.0) for k in range(a.shape[1])], 1)
        s, o = sosfiltfilt(sos, resample(t_syn, syn), axis=0), target
        S, O = np.fft.rfft(s, nf, axis=0), np.fft.rfft(o, nf, axis=0)
        num, den = np.sum(O[:, train] * np.conj(S[:, train]), 1), np.sum(np.abs(S[:, train]) ** 2, 1)
        pred = np.fft.irfft((num / (den + 1e-3 * den.max()))[:, None] * S, nf, axis=0)[: len(t_ldv)]
        mis = lambda sel: float(np.linalg.norm(pred[:, sel] - o[:, sel]) / np.linalg.norm(o[:, sel]))
        last_pred["pred"] = pred
        return mis(behind), mis(before)

    def target(name):  # Salvus traces on the LDV time axis, band-passed like the data
        tt, v = sal[f"t__{name}"], sal[name]
        return sosfiltfilt(sos, np.stack([np.interp(t_ldv, tt, v[:, k], left=0.0) for k in range(v.shape[1])], 1), axis=0)

    jrun = {}

    def mindlin(mass=1.0, inertia=1.0, stiff=1.0, damping=0.0):
        key = round(max(1.0, stiff / min(mass, inertia)), 3)
        if key not in jrun:
            jrun[key] = jax.jit(lambda ms, is_, ss, dmp: run(key, mass_scale=ms, inertia_scale=is_,
                                                              stiffness_scale=ss, damping=dmp)[1])
        dt = 0.6 * dx / (VP_VS * VS * np.sqrt(2.0) * max(1.0, np.sqrt(key)))
        t_sim = t_sal[0] + np.arange(int(np.ceil((t_end - t_sal[0]) / dt)) + 2) * dt
        sc = lambda f: jnp.asarray(np.where(in_cube, f, 1.0), jnp.float32)
        return t_sim, np.asarray(jrun[key](sc(mass), sc(inertia), sc(stiff), jnp.float32(damping)))

    if "--rigid" in sys.argv:  # rigid-body cube (no patch stiffness, no internal resonance)
        M_c = 7850.0 * 0.010**3
        J_c = M_c * (0.010**2 / 12 + (H / 2 + 0.010 / 2) ** 2)
        rigid_run = jax.jit(lambda dmp: run(1.0, rigid_mask=in_cube, rigid_added_mass=M_c,
                                            rigid_added_inertia=J_c, damping=dmp)[1])
        dt1 = 0.6 * dx / (VP_VS * VS * np.sqrt(2.0))
        t_r = t_sal[0] + np.arange(int(np.ceil((t_end - t_sal[0]) / dt1)) + 2) * dt1
        rig = {}
        ss = np.asarray(rigid_run(jnp.float32(0.0)))
        rig["vs_salvus_cube_elastic"] = score(t_r, ss, None, target("cube_x-20mm"))
        print(f"R  rigid cube (elastic) vs Salvus + cube: behind {rig['vs_salvus_cube_elastic'][0]:.3f}, "
              f"before {rig['vs_salvus_cube_elastic'][1]:.3f}", flush=True)
        best = None
        for dmp in (3000.0, 6000.0, 9000.0):
            ss = np.asarray(rigid_run(jnp.float32(dmp)))
            b_, f_ = score(t_r, ss, None, target("cube_x-20mm_Q40"))
            rig[f"vs_salvus_cube_Q40_damping{dmp:.0f}"] = (b_, f_)
            print(f"R  rigid cube + damping {dmp:.0f}/s vs Salvus + cube, Q=40: behind {b_:.3f}, before {f_:.3f}", flush=True)
            if best is None or b_ + f_ < best[0]:
                best = (b_ + f_, dmp, ss)
        rig["damping"] = best[1]
        rig["vs_ldv"] = score(t_r, best[2], None, obs_ldv[:, idx])
        print(f"R  rigid cube + damping {best[1]:.0f}/s vs LDV: behind {rig['vs_ldv'][0]:.3f}, before {rig['vs_ldv'][1]:.3f}"
              " | Salvus 3-D: 0.330/0.334 | patch equivalent: 0.412/0.385", flush=True)
        np.savez_compressed(OUT / "mindlin_rigid_ldv.npz", pred_rigid=last_pred["pred"].astype(np.float32), rec_idx=idx)
        # effective-coupling sweep: fraction of the cube's rotary inertia coupled into bending
        # (glue compliance / in-plane motion not in Mindlin) and size of the rigid footprint
        sweep, best_s = [], None
        for size, frac in itertools.product((0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002), (1.0,)):
            mask = (np.abs(X - CUBE[0]) <= size / 2) & (np.abs(Y - CUBE[1]) <= size / 2)
            ss = np.asarray(jax.jit(lambda dmp, mask=mask, frac=frac: run(
                1.0, rigid_mask=mask, rigid_added_mass=M_c, rigid_added_inertia=frac * J_c, damping=dmp)[1])(
                jnp.float32(rig["damping"])))
            bq, fq = score(t_r, ss, None, target("cube_x-20mm_Q40"))
            bl, fl = score(t_r, ss, None, obs_ldv[:, idx])
            sweep.append(dict(size=size, inertia_fraction=frac, salvus_Q40=(bq, fq), ldv=(bl, fl)))
            print(f"S  rigid footprint {size*1e3:.0f} mm, inertia x{frac}: vs Salvus Q40 {bq:.3f}/{fq:.3f}, "
                  f"vs LDV {bl:.3f}/{fl:.3f}", flush=True)
            if best_s is None or bq + fq < best_s[0]:
                best_s = (bq + fq, sweep[-1], last_pred["pred"].copy())
        rig["sweep"], rig["best_sweep"] = sweep, best_s[1]
        np.savez_compressed(OUT / "mindlin_rigid_best_ldv.npz", pred_rigid=best_s[2].astype(np.float32), rec_idx=idx)
        (OUT / "mindlin_rigid.json").write_text(json.dumps(dict(rig, added_mass=M_c, added_inertia=J_c), indent=1))
        return

    final = _final_factory(obs_ldv=obs_ldv, idx=idx, mindlin=mindlin, score=score, last_pred=last_pred,
                           kappa=kappa)
    if "--final-only" in sys.argv:  # reuse the stored calibration, only redo stage D
        stored = json.loads((OUT / "mindlin_equivalent.json").read_text())
        bestB, bestC = stored["B_best"], stored["C_best"]
        log = stored
        tA, sA = mindlin()
    else:
        bestB = bestC = None
        log = {}
    started = time.perf_counter()
    if bestB is not None:
        return final(log, bestB, bestC, tA, sA)
    # A: plate theory baseline
    tA, sA = mindlin()
    log["A_plate_vs_salvus_plate"] = score(tA, sA, None, target("no_cube"))
    print(f"A  Mindlin plate vs Salvus plate only: behind {log['A_plate_vs_salvus_plate'][0]:.3f}, "
          f"before {log['A_plate_vs_salvus_plate'][1]:.3f}  t={time.perf_counter()-started:.0f}s", flush=True)
    log["A_plate_vs_salvus_cube"] = score(tA, sA, None, target("cube_x-20mm"))
    print(f"   (Mindlin plate vs Salvus + cube: behind {log['A_plate_vs_salvus_cube'][0]:.3f})", flush=True)

    # B: cube patch, mass fixed at the physical value, inertia and stiffness searched
    tgt_B, grid_B = target("cube_x-20mm"), []
    for inertia, stiff in itertools.product((1.0, 0.3 * INERTIA_FACTOR, INERTIA_FACTOR), (1.0, 10.0, 50.0)):
        tt, ss = mindlin(MASS_FACTOR, inertia, stiff)
        b, f = score(tt, ss, None, tgt_B)
        grid_B.append(dict(mass=MASS_FACTOR, inertia=inertia, stiff=stiff, behind=b, before=f))
        print(f"B  patch mass x{MASS_FACTOR:.2f} inertia x{inertia:5.1f} stiffness x{stiff:4.0f}: behind {b:.3f}, "
              f"before {f:.3f}  t={time.perf_counter()-started:.0f}s", flush=True)
    bestB = min(grid_B, key=lambda g: g["behind"] + g["before"])  # balanced: the cube also reflects
    log["B_grid"], log["B_best"] = grid_B, bestB

    # C: damping, with the best patch
    tgt_C, grid_C = target("cube_x-20mm_Q40"), []
    for damping in (0.0, 3000.0, 6000.0, 9000.0):  # Q = 40 at 39 kHz <-> about 6100 / s
        tt, ss = mindlin(bestB["mass"], bestB["inertia"], bestB["stiff"], damping)
        b, f = score(tt, ss, None, tgt_C)
        grid_C.append(dict(damping=damping, behind=b, before=f))
        print(f"C  best patch + damping {damping:5.0f}/s vs Salvus Q=40: behind {b:.3f}, before {f:.3f}", flush=True)
    bestC = min(grid_C, key=lambda g: g["behind"] + g["before"])
    log["C_grid"], log["C_best"] = grid_C, bestC

    return final(log, bestB, bestC, tA, sA)


def _final_factory(**ctx):
    def final(log, bestB, bestC, tA, sA):
        """Stage D: equivalent vs the LDV data (and the plain plate); saves the predictions."""
        obs = ctx["obs_ldv"][:, ctx["idx"]]
        tt, ss = ctx["mindlin"](bestB["mass"], bestB["inertia"], bestB["stiff"], bestC["damping"])
        log["D_equivalent_vs_ldv"] = ctx["score"](tt, ss, None, obs)
        pred_eq = ctx["last_pred"]["pred"]
        log["D_plate_vs_ldv"] = ctx["score"](tA, sA, None, obs)
        np.savez_compressed(OUT / "mindlin_equivalent_ldv.npz", pred_equivalent=pred_eq.astype(np.float32),
                            pred_plate=ctx["last_pred"]["pred"].astype(np.float32), rec_idx=ctx["idx"])
        print(f"D  Mindlin equivalent vs LDV: behind {log['D_equivalent_vs_ldv'][0]:.3f}, before {log['D_equivalent_vs_ldv'][1]:.3f}"
              f" | plain Mindlin plate {log['D_plate_vs_ldv'][0]:.3f}/{log['D_plate_vs_ldv'][1]:.3f}"
              f" | Salvus 3-D cube+Q40: 0.330/0.334", flush=True)
        log["kappa"], log["mass_factor"], log["inertia_factor"] = ctx["kappa"], MASS_FACTOR, INERTIA_FACTOR
        (OUT / "mindlin_equivalent.json").write_text(json.dumps(log, indent=1))
    return final


if __name__ == "__main__":
    main()
