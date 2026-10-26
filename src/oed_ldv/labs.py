"""'Labs' for the LDV OED loop: return measured out-of-plane velocity at requested points.

Every lab returns band-passed traces on the LDV time axis, shape (nt, nrec), and the true
cube position (for evaluation only). The inversion never sees the lab's wavelet, delay or
attachment model.

* MindlinLab: calibrated Mindlin truth (rigid steel cube, 6 mm effective footprint, physical
  mass and rotary inertia, damping), Ricker 39 kHz with a 170 us trigger delay.
* SalvusLab: Salvus 3-D plate + bonded steel cube, PMMA Q = 40 (ldv_assessment/salvus_project,
  configuration cube_x-20mm_Q40), one new simulation per acquisition (~80 s on the GPU).
* ReplayLab: the real dense LDV scan as the instrument; a requested point returns the measured
  trace of the nearest usable scan point (1.5 mm grid). ``snap`` gives the points actually
  measured, which the FWI uses. Points within 4 cm of the transducer and on the scan row
  through it are not usable (as in src/ldv_inversion). Truth for evaluation: the cube found
  from the full scan (stage A).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt

from problem import BAND, REPO, RHO, SCAN_P, VP_VS, OEDProblem

import jax
import jax.numpy as jnp

SALVUS_PY = "/home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python"
HERE = Path(__file__).resolve().parent


class MindlinLab:
    name = "mindlin"
    cube = np.array([-0.020, -0.005])

    def __init__(self, footprint=0.006, delay=170e-6, f0=39e3):
        self.prob = OEDProblem()
        p = self.prob
        t_sim = np.arange(len(p.stf)) * p.dt
        a = (np.pi * f0 * (t_sim - delay)) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        X, Y = np.meshgrid(p.lo[0] + np.arange(p.nx) * p.dx, p.lo[1] + np.arange(p.ny) * p.dx, indexing="ij")
        self.mask = jnp.asarray((np.abs(X - self.cube[0]) <= footprint / 2) & (np.abs(Y - self.cube[1]) <= footprint / 2))
        m_c = 7850.0 * 0.010**3
        self.m_c, self.j_c = m_c, m_c * (0.010**2 / 12 + 0.010**2)
        self._run = jax.jit(self._simulate)

    def _simulate(self, rec_m):
        from mindlin import mindlin_plate

        p = self.prob
        one = jnp.ones((p.nx, p.ny), jnp.float32)
        rec_i = (rec_m - jnp.asarray(p.lo, jnp.float32)) / p.dx
        tr = mindlin_plate(VP_VS * p.vs * one, p.vs * one, RHO * one, p.h, p.s_i, self.stf, p.dx, p.dx, p.dt, rec_i,
                           sponge_cells=p.sponge_cells, kappa=p.kappa, damping=p.damping, rigid_mask=self.mask,
                           rigid_added_mass=self.m_c, rigid_added_inertia=self.j_c)[..., 0]
        return (1 - p.w) * tr[p.i0] + p.w * tr[p.i0 + 1]

    def acquire(self, rec_m, name=None):
        t0 = time.perf_counter()
        out = sosfiltfilt(BAND, np.asarray(self._run(jnp.asarray(rec_m, jnp.float32))), axis=0)
        return out.astype(np.float32), time.perf_counter() - t0

    def dense(self):
        """Truth at all scan points (for the full-field prediction metric)."""
        out = []
        for chunk in np.array_split(SCAN_P, 4):
            out.append(self.acquire(chunk)[0])
        return SCAN_P, np.concatenate(out, 1)


class SalvusLab:
    name = "salvus"
    cube = np.array([-0.020, -0.005])
    config = "cube_x-20mm_Q40"

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.t = OEDProblem().t

    def acquire(self, rec_m, name):
        # Salvus event names are content-addressed: the same name always means the same points
        # (an earlier run with a different layout under the same tag cannot be reused by mistake).
        import hashlib

        rec64 = np.round(np.asarray(rec_m, np.float64), 9)
        name = f"{name}_{hashlib.sha1(rec64.tobytes()).hexdigest()[:8]}"
        rec_file, out_file = self.run_dir / f"{name}_receivers.npy", self.run_dir / f"{name}_data.npz"
        np.save(rec_file, rec64)
        t0 = time.perf_counter()
        subprocess.run([SALVUS_PY, str(HERE / "acquire_salvus_ldv.py"), "--receivers", str(rec_file),
                        "--name", name, "--config", self.config, "--out", str(out_file)],
                       check=True, capture_output=True, text=True)
        d = np.load(out_file)
        v = np.stack([np.interp(self.t, d["t"], d["vz"][:, k], left=0.0, right=0.0) for k in range(d["vz"].shape[1])], 1)
        return sosfiltfilt(BAND, v, axis=0).astype(np.float32), time.perf_counter() - t0

    def dense(self):
        """Dense truth: the Salvus run at the scan points (ldv_assessment/salvus_ldv_cube.npz)."""
        s = np.load(REPO / "ldv_assessment" / "salvus_ldv_cube.npz")
        idx, ts, v = s["rec_idx"], s[f"t__{self.config}"], s[self.config]
        v = np.stack([np.interp(self.t, ts, v[:, k], left=0.0, right=0.0) for k in range(v.shape[1])], 1)
        return SCAN_P[idx], sosfiltfilt(BAND, v, axis=0).astype(np.float32)


class ReplayLab:
    name = "replay"
    cube = np.array([-0.018, -0.004])

    def __init__(self, near_source=0.04):
        from fit_mindlin_ldv import load_ldv

        prob = OEDProblem()
        P, t, obs, _, _ = load_ldv(prob.t[-1] + 1e-9)
        assert len(t) == len(prob.t) and np.allclose(P, SCAN_P)
        self.P, self.obs = P, obs.astype(np.float32)
        usable = (np.hypot(*(P - prob.src).T) > near_source) & (np.abs(P[:, 1] - prob.src[1]) > 0.004)
        self.usable = np.where(usable)[0]

    def _nearest(self, rec_m):
        d = np.linalg.norm(self.P[self.usable][None] - np.asarray(rec_m, float)[:, None], axis=2)
        return self.usable[np.argmin(d, 1)]

    def snap(self, rec_m):
        return self.P[self._nearest(rec_m)]

    def acquire(self, rec_m, name=None):
        return self.obs[:, self._nearest(rec_m)], 0.0

    def dense(self):
        return self.P, self.obs
