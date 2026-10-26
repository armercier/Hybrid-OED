"""Parametric scatterer fit to the LDV scan with the JAX Mindlin solver.

Starting from the best homogeneous fit (fit_mindlin_ldv.py), adds a circular scatterer
near the shadow apex seen in the residual amplitude map. Scatterer types: thickness change
(``h``), added mass (``rho``), stiffness change (``vs``, vp/vs kept), near through-hole
(h and stiffness tiny). The wavelet filter is estimated on training points outside the
near-source zone (r < 4 cm) and the bad scan row; the misfit is reported on held-out
points (a) around/behind the scatterer and (b) in the clean region before it.

Usage:  python src/ldv_assessment/fit_scatterer_ldv.py
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


def main() -> None:
    import jax
    import jax.numpy as jnp

    from mindlin import mindlin_plate

    best_h = json.loads((OUT / "mindlin_ldv_fit.json").read_text())["best"]
    vs, h0, kappa, src = best_h["vs"], best_h["h"], best_h["kappa"], np.array(best_h["src"])
    tmax, dx, margin, sponge_m = 650e-6, 0.5e-3, 0.12, 0.06
    P, t, obs, _, sos = load_ldv(tmax)
    r_src = np.hypot(*(P - src).T)
    usable = (r_src > 0.04) & (np.abs(P[:, 1] - src[1]) > 0.004)
    train = (np.random.default_rng(0).random(len(P)) < 0.5) & usable
    test_behind = ~train & usable & (P[:, 0] > -0.035)
    test_before = ~train & usable & (P[:, 0] <= -0.035)
    lo = P.min(0) - margin
    nx, ny = np.round((P.max(0) + margin - lo) / dx).astype(int) + 1
    X, Y = np.meshgrid(lo[0] + np.arange(nx) * dx, lo[1] + np.arange(ny) * dx, indexing="ij")
    dt = 0.6 * dx / (VP_VS * vs * np.sqrt(2.0))
    t_sim = np.arange(int(np.ceil(t[-1] / dt)) + 2) * dt
    a = (np.pi * 45e3 * (t_sim - 1.5 / 45e3)) ** 2
    stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
    one = jnp.ones((nx, ny), jnp.float32)
    s_i = jnp.asarray((src - lo) / dx, jnp.float32)
    rec_i = jnp.asarray((P - lo) / dx, jnp.float32)

    @jax.jit
    def simulate(vsmap, rhomap, hmap):
        return mindlin_plate(VP_VS * vsmap, vsmap, rhomap, hmap, s_i, stf, dx, dx, dt, rec_i,
                             sponge_cells=int(round(sponge_m / dx)), kappa=kappa)[..., 0]

    nf = 2 * len(t)

    def score(vsmap, rhomap, hmap):
        syn = np.asarray(simulate(*(jnp.asarray(a, jnp.float32) for a in (vsmap, rhomap, hmap))))
        syn = np.stack([np.interp(t, t_sim, syn[:, r]) for r in range(syn.shape[1])], 1)
        syn = sosfiltfilt(sos, syn, axis=0)
        S, O = np.fft.rfft(syn, nf, axis=0), np.fft.rfft(obs, nf, axis=0)
        num, den = np.sum(O[:, train] * np.conj(S[:, train]), 1), np.sum(np.abs(S[:, train]) ** 2, 1)
        pred = np.fft.irfft((num / (den + 1e-3 * den.max()))[:, None] * S, nf, axis=0)[: len(t)]
        mis = lambda sel: float(np.linalg.norm(pred[:, sel] - obs[:, sel]) / np.linalg.norm(obs[:, sel]))
        return mis(test_behind), mis(test_before), pred

    ones = np.ones((nx, ny))
    base_behind, base_before, _ = score(vs * ones, RHO * ones, h0 * ones)
    print(f"homogeneous: held-out misfit behind scatterer {base_behind:.3f}, before it {base_before:.3f}", flush=True)
    kinds = {  # name: (vs factor, rho factor, h factor) inside the scatterer
        "hole (h x0.05, vs x0.3)": (0.3, 1.0, 0.05), "thinner h x0.5": (1.0, 1.0, 0.5),
        "thicker h x2": (1.0, 1.0, 2.0), "added mass rho x4": (1.0, 4.0, 1.0),
        "stiffer vs x2": (2.0, 1.0, 1.0), "softer vs x0.5": (0.5, 1.0, 1.0)}
    results, best = [], None
    started = time.perf_counter()
    for (cx, cy), radius, (kind, (fv, fr, fh)) in itertools.product(((-0.022, -0.005),), (0.006, 0.012), kinds.items()):
        inside = np.hypot(X - cx, Y - cy) < radius
        behind, before, pred = score(vs * np.where(inside, fv, 1.0), RHO * np.where(inside, fr, 1.0),
                                     h0 * np.where(inside, fh, 1.0))
        results.append(dict(cx=cx, cy=cy, radius=radius, kind=kind, behind=behind, before=before))
        print(f"r={radius*1e3:.0f} mm {kind:24s}: behind {behind:.3f} (homog {base_behind:.3f}), before {before:.3f}"
              f"  t={time.perf_counter()-started:.0f}s", flush=True)
        if best is None or behind < best[0]["behind"]:
            best = (results[-1], pred)
    b, pred = best
    print("best:", b)
    np.savez_compressed(OUT / "mindlin_ldv_scatterer_fit.npz", P=P, t=t, obs=obs, pred=pred, train=train, src=src)
    (OUT / "mindlin_ldv_scatterer_fit.json").write_text(json.dumps(
        {"homogeneous_behind": base_behind, "homogeneous_before": base_before, "best": b, "grid": results}, indent=1))


if __name__ == "__main__":
    main()
