"""Finite-difference references for the scattered-field PINN feasibility study.

For each source, with homogeneous density (the FWI forward model), the paper
solver ``acoustic2D_cpml_minmem`` is run (unmodified) at three resolutions:

  * r = 1: the paper grid (8 points per minimum wavelength);
  * r = 2, 4: refined grids with the *same continuous* velocity c(x), i.e. the
    bilinear interpolant of the coarse grid, which is also what the PINN sees.

Each level records receiver traces for the incident (c = c0) and total (true c)
fields.  Level r = 4 additionally records a dense set of "receivers" on the
nodes of the target box (spacing dx/2), which gives the continuum reference
field and the incident field u0 that drives the PINN.  2-vs-4 measures how well
the r = 4 level is converged; 1-vs-4 measures the paper solver's own error.

Sample k of every level corresponds to t = k * dt (source-time convention).

Usage:  python src/pinn_feasibility/fd_reference.py [--sources 0 1 2 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import BOX_CENTER, BOX_HALF, REFERENCE_DIR, eval_receivers_m

import jax
import jax.numpy as jnp
from scipy.ndimage import map_coordinates

from hybridoed.forward import acoustic2D_cpml_minmem
from run_oed_multiseed import Simulation, scientific_config

LEVELS = (1, 2, 4)


def nrm(traces: np.ndarray) -> np.ndarray:
    """FWI trace normalisation (per receiver max-abs); traces are (nt, nrec)."""
    return traces / np.maximum(np.max(np.abs(traces), axis=0), 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    sim = Simulation(scientific_config())
    cfg = sim.config
    print(f"device={jax.devices()[0]}  grid={sim.nx}x{sim.ny}  dx={sim.dx*1e3:.3f} mm  "
          f"dt={sim.dt*1e6:.4f} us  n_steps={sim.n_steps}  pml={sim.pml_width}")

    coarse_c = {"homo": np.asarray(sim.homo_model), "true": np.asarray(sim.true_model)}
    t_samples = np.arange(sim.n_steps) * sim.dt

    receivers = eval_receivers_m()
    rec_m = np.concatenate([receivers["ring"], receivers["diag"]])

    # Dense box nodes at spacing dx/2, with a 3-coarse-cell margin.
    lo, hi = BOX_CENTER - BOX_HALF - 3 * sim.dx, BOX_CENTER + BOX_HALF + 3 * sim.dx
    i_lo, i_hi = np.floor(sim.x_to_i(lo[0]) * 2), np.ceil(sim.x_to_i(hi[0]) * 2)
    j_lo, j_hi = np.floor(sim.y_to_i(lo[1]) * 2), np.ceil(sim.y_to_i(hi[1]) * 2)
    gi = np.arange(i_lo, i_hi + 1) / 2.0  # coarse index units
    gj = np.arange(j_lo, j_hi + 1) / 2.0
    box_x = sim.x_min_m + gi * sim.dx
    box_y = sim.y_min_m + gj * sim.dy
    GI, GJ = np.meshgrid(gi, gj, indexing="ij")
    dense_idx = np.stack([GI.ravel(), GJ.ravel()], axis=1)

    solvers = {}
    for r in LEVELS:
        n = r * (sim.nx - 1) + 1
        ii, jj = np.meshgrid(np.arange(n) / r, np.arange(n) / r, indexing="ij")
        vel = {k: jnp.asarray(map_coordinates(v, [ii, jj], order=1).astype(np.float32))
               for k, v in coarse_c.items()}
        rec = rec_m if r < 4 else None
        rec_idx = np.asarray(sim.xy_to_indices(rec_m))
        if r == 4:
            rec_idx = np.concatenate([rec_idx, dense_idx])
        rec_idx = jnp.asarray(r * rec_idx, dtype=jnp.float32)
        del rec

        def solve(v, s, r=r, rec_idx=rec_idx):
            return acoustic2D_cpml_minmem(
                v, jnp.full_like(v, cfg.density), r * s, cfg.f0,
                sim.dx / r, sim.dy / r, sim.dt / r, r * sim.n_steps,
                receiver_is=rec_idx, output_wavefield=False, pml_width=r * sim.pml_width,
                R_coeff=cfg.pml_reflection_coefficient, m=cfg.pml_polynomial_order,
                kappa_max=cfg.pml_kappa_max, alpha_max=None,
            )[0][::r]  # decimate to the coarse time samples
        solvers[r] = (jax.jit(solve), vel)

    n_rec = len(rec_m)
    for s in args.sources:
        src = sim.source_indices[s]
        out: dict[str, np.ndarray] = {}
        timings: dict[str, float] = {}
        for r, (solve, vel) in solvers.items():
            for name in ("homo", "true"):
                res = solve(vel[name], src)
                jax.block_until_ready(res)
                t0 = time.perf_counter()
                res = np.asarray(jax.block_until_ready(solve(vel[name], src)))
                timings[f"time_r{r}_{name}"] = time.perf_counter() - t0
                out[f"tr_{name}_r{r}"] = res[:, :n_rec]
                if r == 4:
                    out[f"field_{name}"] = res[:, n_rec:].reshape(len(t_samples), len(gi), len(gj))

        np.savez_compressed(
            REFERENCE_DIR / f"source_{s}.npz",
            source_m=sim.indices_to_meters(np.asarray(src)),
            box_x=box_x, box_y=box_y, t=t_samples, dt=sim.dt,
            c_true=map_coordinates(coarse_c["true"], [GI, GJ], order=1), c0=cfg.c_min,
            rec_m=rec_m, n_ring=len(receivers["ring"]), **out, **timings,
        )

        def sc(r):
            return nrm(out[f"tr_true_r{r}"]) - nrm(out[f"tr_homo_r{r}"])

        def rel(a, b):
            return np.linalg.norm(a - b) / np.linalg.norm(b)

        print(f"source {s}: solve time r1/r2/r4 = {timings['time_r1_true']:.3f}/"
              f"{timings['time_r2_true']:.3f}/{timings['time_r4_true']:.3f} s | "
              f"scattered/total (r4) {np.linalg.norm(sc(4)) / np.linalg.norm(nrm(out['tr_true_r4'])):.3f} | "
              f"incident rel err r1:{rel(nrm(out['tr_homo_r1']), nrm(out['tr_homo_r4'])):.3f} "
              f"r2:{rel(nrm(out['tr_homo_r2']), nrm(out['tr_homo_r4'])):.3f} | "
              f"scattered rel err r1:{rel(sc(1), sc(4)):.3f} r2:{rel(sc(2), sc(4)):.3f}")


if __name__ == "__main__":
    main()
