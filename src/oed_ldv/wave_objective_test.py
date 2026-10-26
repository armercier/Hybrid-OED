"""Feasibility of the lab-consistent waveform-misfit design objective (memory, time).

d~(p) = d_meas + F_p(m_t) - F_{p_t}(m_t): measured data extrapolated to nearby points with the
current model (exact at p_t; its p-derivative is the model-predicted spatial derivative).
Objective: misfit after K unrolled (rematerialised) gradient-descent FWI updates on d~(p),
differentiated with respect to the LDV point coordinates.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from problem import OEDProblem
from labs import MindlinLab
from run_oed_ldv import initial_layout

import jax
import jax.numpy as jnp


def make_objective(prob, K, lr):
    def objective(rec, m_t, rec_t, d_meas):
        d_tilde = d_meas + prob.simulate(m_t, rec) - jax.lax.stop_gradient(prob.simulate(m_t, rec_t))

        @jax.checkpoint
        def update(m, _):
            return m - lr * jax.grad(prob.fwi_loss)(m, rec, d_tilde), None

        m_K, _ = jax.lax.scan(update, m_t, None, length=K)
        return prob.misfit(m_K, rec, d_tilde)
    return objective


def main():
    dx = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
    prob = OEDProblem(dx=dx, checkpoint_every=64)
    rec = prob.project(jnp.asarray(initial_layout(), jnp.float32))
    d_meas = jnp.asarray(MindlinLab().acquire(np.asarray(rec))[0])
    m_t = jnp.zeros(prob.n_model, jnp.float32)
    for K in (1, 2, 3):
        f = jax.jit(jax.value_and_grad(make_objective(prob, K, lr=0.5)))
        try:
            t0 = time.perf_counter(); v, g = f(rec, m_t, rec, d_meas); g.block_until_ready(); t1 = time.perf_counter() - t0
            t0 = time.perf_counter(); v, g = f(rec, m_t, rec, d_meas); g.block_until_ready(); t2 = time.perf_counter() - t0
            peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
            print(f"dx {dx*1e3:.1f} mm, K={K}: objective {float(v):.4f}, |grad| {float(jnp.linalg.norm(g)):.2e}, "
                  f"finite {bool(jnp.all(jnp.isfinite(g)))}, first {t1:.0f}s, run {t2:.1f}s, peak GPU {peak:.1f} GiB", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"dx {dx*1e3:.1f} mm, K={K}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)
            break


if __name__ == "__main__":
    main()
