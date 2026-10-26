"""Reconstruct the LDV out-of-plane wavefield with the JAX Mindlin plate solver.

No metadata exist for this scan, so the model is inferred from the data:
* plate frame: scan points projected on their best-fit plane (ldv_analysis.py);
* open plate (sponge), homogeneous PMMA-like plate, thickness h, shear speed vs
  (vp = 2.037 vs), kappa matched to exact A0 dispersion;
* normal point source at (sx, sy) with a broadband Ricker; the true source wavelet
  and trigger delay are absorbed by one deconvolution filter estimated on a random
  half of the scan points ('train') and scored on the other half ('test');
* data: out-of-plane velocity, 8-70 kHz band-pass, window t < --tmax, traces with
  dropout spikes removed.
A grid search over (sx, sy, vs, h) picks the model with the lowest held-out misfit.

Usage:  python src/ldv_assessment/fit_mindlin_ldv.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "elastic"))
OUT = REPO / "ldv_assessment"
MAT = REPO / "grid_scan_dx_dy_1_5mm_fc_25kHz_fs_625kHz_100avg_1250nt_200V_scatter.mat"
RHO = 1180.0
VP_VS = 2730.0 / 1340.0


def load_ldv(tmax):
    m = loadmat(MAT, simplify_cells=True)
    xyz, t = np.asarray(m["XYZ"], float), np.asarray(m["t"], float)
    v = np.stack([m["vib_x"], m["vib_y"], m["vib_z"]], -1).astype(float)
    c = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - c, full_matrices=False)
    e1, e2, nrm = vt
    if nrm[2] < 0:
        nrm = -nrm
    P = (xyz - c) @ np.stack([e1, e2], 1)
    vz = v @ nrm  # out-of-plane velocity (npos, nt)
    fs = 1 / (t[1] - t[0])
    # LDV dropouts: traces whose energy above 100 kHz is anomalous relative to the
    # 10-60 kHz band (robust outlier test on the log ratio)
    spec = np.abs(np.fft.rfft(vz, axis=1)) ** 2
    f = np.fft.rfftfreq(vz.shape[1], 1 / fs)
    ratio = np.log(spec[:, f > 100e3].sum(1) / spec[:, (f > 10e3) & (f < 60e3)].sum(1))
    med = np.median(ratio)
    mad = np.median(np.abs(ratio - med))
    bad = ratio > med + 5 * 1.4826 * mad
    sos = butter(4, [8e3, 70e3], btype="band", fs=fs, output="sos")
    vz = sosfiltfilt(sos, vz, axis=1)
    w = t < tmax
    return P[~bad], t[w], vz[~bad][:, w].T, int(bad.sum()), sos  # data (nt, nrec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmax", type=float, default=650e-6)
    parser.add_argument("--dx", type=float, default=0.5e-3)
    parser.add_argument("--margin", type=float, default=0.12, help="plate margin around the scan (m)")
    parser.add_argument("--sponge", type=float, default=0.06)
    parser.add_argument("--vs", type=float, nargs="+", default=[1300.0, 1340.0, 1380.0])
    parser.add_argument("--h", type=float, nargs="+", default=[9e-3, 10e-3, 11e-3])
    parser.add_argument("--src-offsets", type=float, nargs="+", default=[-0.004, 0.0, 0.004])
    parser.add_argument("--src", type=float, nargs=2, default=None, help="default: arrival-time fit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from mindlin import a0_matched_kappa, mindlin_plate

    P, t, obs, n_bad, sos = load_ldv(args.tmax)
    rng = np.random.default_rng(0)
    train = rng.random(len(P)) < 0.5
    src0 = np.array(args.src) if args.src else np.array([-0.097, -0.005])
    print(f"{len(P)} traces kept ({n_bad} with dropout spikes removed), window t < {args.tmax*1e6:.0f} us; "
          f"start source {src0}")

    lo = P.min(0) - args.margin
    hi = P.max(0) + args.margin
    nx, ny = (np.round((hi - lo) / args.dx).astype(int) + 1)
    dx = args.dx
    rec_i = jnp.asarray((P - lo) / dx, jnp.float32)
    dt = 0.6 * dx / (VP_VS * max(args.vs) * np.sqrt(2.0))
    t_sim = np.arange(int(np.ceil(t[-1] / dt)) + 2) * dt
    f_src = 45e3
    a = (np.pi * f_src * (t_sim - 1.5 / f_src)) ** 2
    stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
    sponge = int(round(args.sponge / dx))
    print(f"grid {nx} x {ny} at {dx*1e3:.2f} mm, {len(t_sim)} steps, sponge {sponge} cells")

    @jax.jit
    def simulate(vs, h, kappa, s_i):
        f = jnp.full((nx, ny), 1.0, jnp.float32)
        return mindlin_plate(VP_VS * vs * f, vs * f, RHO * f, h, s_i, stf, dx, dx, dt, rec_i,
                             sponge_cells=sponge, kappa=kappa)[..., 0]

    nf = 2 * len(t)

    def score(syn):
        """Deconvolution filter on train traces, misfit on test traces (both band-passed)."""
        syn = sosfiltfilt(sos, syn, axis=0)
        S, O = np.fft.rfft(syn, nf, axis=0), np.fft.rfft(obs, nf, axis=0)
        num = np.sum(O[:, train] * np.conj(S[:, train]), 1)
        den = np.sum(np.abs(S[:, train]) ** 2, 1)
        F = num / (den + 1e-3 * den.max())
        pred = np.fft.irfft(F[:, None] * S, nf, axis=0)[: len(t)]
        mis = lambda sel: float(np.linalg.norm(pred[:, sel] - obs[:, sel]) / np.linalg.norm(obs[:, sel]))
        return mis(~train), mis(train), pred, F

    results, best = [], None
    started = time.perf_counter()
    for vs, h in itertools.product(args.vs, args.h):
        kappa = a0_matched_kappa(h, VP_VS * vs, vs, RHO, 39e3)
        for ox, oy in itertools.product(args.src_offsets, args.src_offsets):
            s = src0 + np.array([ox, oy])
            syn = np.asarray(simulate(vs, h, kappa, jnp.asarray((s - lo) / dx, jnp.float32)))
            syn = np.stack([np.interp(t, t_sim, syn[:, r]) for r in range(syn.shape[1])], 1)
            test, tr, pred, F = score(syn)
            results.append(dict(vs=vs, h=h, kappa=kappa, src=s.tolist(), test=test, train=tr))
            if best is None or test < best[0]["test"]:
                best = (results[-1], pred, F)
        r = min((x for x in results if x["vs"] == vs and x["h"] == h), key=lambda x: x["test"])
        print(f"vs={vs:.0f} h={h*1e3:.1f} mm kappa={kappa:.3f}: best test misfit {r['test']:.3f} "
              f"(train {r['train']:.3f}) src=({r['src'][0]:+.3f},{r['src'][1]:+.3f})  t={time.perf_counter()-started:.0f}s",
              flush=True)

    b, pred, F = best
    print("best:", b)
    np.savez_compressed(OUT / "mindlin_ldv_fit.npz", P=P, t=t, obs=obs, pred=pred, train=train,
                        F=F, src=np.array(b["src"]))
    (OUT / "mindlin_ldv_fit.json").write_text(json.dumps({"args": vars(args), "best": b, "grid": results,
                                                           "n_traces": len(P), "n_removed": n_bad}, indent=1))


if __name__ == "__main__":
    main()
