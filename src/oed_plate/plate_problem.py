"""Plate FWI problem for the OED loop: plane-stress forward model, misfit, TV, illumination.

Same choices as src/elastic/invert_plate.py (best sparse-receiver settings), but with
the receiver positions as a differentiable argument:

* forward: ``plane_stress_free`` (free edges, S0-corrected vp), 1 mm grid, remat in time;
* unknown: relative vs update on a 5 mm grid, masked near the sources, tapered near edges;
* misfit: in-plane velocity, one closed-form amplitude per event, window t < tmax;
* TV regularisation of the update;
* stochastic illumination (paper, Appendix A): diag(J^T J) from Rademacher probes,
  scored by variance of the normalised field minus beta * log(mean).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "src" / "elastic"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from compare_plate import material_grid
from plane_stress import plane_stress_free, s0_corrected_vp

BENCH = REPO / "salvus_plate"


class PlateProblem:
    def __init__(self, events, t_obs, tmax=300e-6, dx=1e-3, param_spacing=5e-3, source_mask=0.03,
                 edge_taper=0.03, vs_bounds=(800.0, 1600.0), checkpoint_every=64, tv=3e-4, tv_eps=1e-3,
                 illum_beta=0.05, receiver_margin=0.02):
        self.defn = defn = json.loads((BENCH / "benchmark_definition.json").read_text())
        self.events = list(events)
        self.L, H, f0 = defn["L"], defn["H"], defn["f0"]
        self.pm = pm = defn["pmma"]
        self.window = t_obs < tmax
        t_obs = t_obs[self.window]
        self.tv_weight, self.tv_eps, self.beta = tv, tv_eps, illum_beta
        self.vs_bounds, self.margin = vs_bounds, receiver_margin

        n = int(round(self.L / dx)) + 1
        self.dx = dx = self.L / (n - 1)
        self.ratio = s0_corrected_vp(pm["vp"], pm["vs"], H, f0) / pm["vs"]
        self.dt = dt = 0.6 * dx / (vs_bounds[1] * self.ratio * np.sqrt(2.0))
        t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 2) * dt
        a = (np.pi * f0 * t_sim) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = (t_obs - t_sim[0]) / dt
        i0 = np.floor(pos).astype(int)
        self.w = jnp.asarray(pos - i0, jnp.float32)[:, None, None]
        self.i0 = jnp.asarray(i0)
        self.checkpoint_every = checkpoint_every

        nc = int(round(self.L / param_spacing)) + 1
        self.nc = nc
        I, J = np.meshgrid(np.arange(n) * dx / param_spacing, np.arange(n) * dx / param_spacing, indexing="ij")
        self.coords = [jnp.asarray(I, jnp.float32), jnp.asarray(J, jnp.float32)]
        Xc, Yc = np.meshgrid(np.arange(nc) * param_spacing, np.arange(nc) * param_spacing, indexing="ij")
        mask = np.ones((nc, nc), np.float32)
        for e in self.events:
            sx, sy = defn["sources"][e]["position"]
            mask[np.hypot(Xc - sx, Yc - sy) < source_mask] = 0.0
        if edge_taper > 0:
            d_edge = np.minimum.reduce([Xc, self.L - Xc, Yc, self.L - Yc])
            mask *= np.where(d_edge < edge_taper, 0.5 - 0.5 * np.cos(np.pi * d_edge / edge_taper), 1.0)
        self.mask = jnp.asarray(mask)
        self.region = jnp.asarray(mask > 0.5)
        self.rho = jnp.full((n, n), pm["rho"], jnp.float32)
        self.true_vs = np.asarray(material_grid(defn, n, dx, True)[1])
        self.inside = np.abs(self.true_vs - pm["vs"]) > 1.0

    # ------------------------------------------------------------------ model and forward
    def vs_map(self, delta):
        rel = 1.0 + self.mask * delta
        return jnp.clip(self.pm["vs"] * map_coordinates(rel, self.coords, order=1), *self.vs_bounds)

    def simulate(self, delta, e, rec_m):
        src = self.defn["sources"][e]
        vs = self.vs_map(delta)
        tr = plane_stress_free(vs * self.ratio, vs, self.rho,
                               jnp.asarray(np.asarray(src["position"]) / self.dx, jnp.float32),
                               jnp.asarray(src["force"], jnp.float32), self.stf, self.dx, self.dx, self.dt,
                               rec_m / self.dx, checkpoint_every=self.checkpoint_every)
        return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]

    # ---------------------------------------------------------------------- inversion
    def event_misfit(self, delta, e, rec_m, observed):
        p = self.simulate(delta, e, rec_m)
        scale = jax.lax.stop_gradient(jnp.sum(p * observed) / jnp.sum(p * p))
        return jnp.sum((scale * p - observed) ** 2) / jnp.sum(observed**2)

    def tv(self, delta):
        m = self.mask * delta
        gx, gy = m[1:, :-1] - m[:-1, :-1], m[:-1, 1:] - m[:-1, :-1]
        return jnp.sum(jnp.sqrt(gx**2 + gy**2 + self.tv_eps**2) - self.tv_eps)

    def observed_from(self, acquisition):
        """Window and select the in-plane components of an acquisition npz."""
        return {e: jnp.asarray(acquisition[f"v_{e}"][self.window][..., :2]) for e in self.events}

    # -------------------------------------------------------------------- illumination
    def illumination(self, rec_m, delta, probes):
        """Stochastic Gauss-Newton illumination score (lower is better); probes: {e: (P, nt, nrec, 2)}."""
        h = 0.0
        for e in self.events:
            out, pull = jax.vjp(lambda d: self.simulate(d, e, rec_m), delta)
            sens = jax.lax.map(lambda q: pull(q)[0], probes[e])
            h = h + jnp.mean(sens**2, axis=0)
        h = h / len(self.events)
        # one 3x3 smoothing pass, then statistics over the unmasked region
        hp = jnp.pad(h, 1, mode="edge")
        h = sum(hp[i:i + self.nc, j:j + self.nc] for i in range(3) for j in range(3)) / 9.0
        vals = jnp.where(self.region, h, 0.0)
        count = jnp.sum(self.region)
        mean = jnp.sum(vals) / count
        var = jnp.sum(jnp.where(self.region, (h / mean - 1.0) ** 2, 0.0)) / count
        return var - self.beta * jnp.log(mean)

    def make_probes(self, key, n_probes, n_rec):
        nt = int(self.window.sum())
        probes = {}
        for e in self.events:
            key, sub = jax.random.split(key)
            probes[e] = jax.random.rademacher(sub, (n_probes, nt, n_rec, 2)).astype(jnp.float32) \
                / np.sqrt(nt * n_rec * 2)
        return probes

    def project(self, rec_m):
        return jnp.clip(rec_m, self.margin, self.L - self.margin)

    # ------------------------------------------------------------------------- metrics
    def metrics(self, delta):
        vs = np.asarray(self.vs_map(delta))
        return dict(vs_rmse=float(np.sqrt(np.mean((vs - self.true_vs) ** 2))),
                    vs_mean_inside=float(vs[self.inside].mean()),
                    vs_std_outside=float(vs[~self.inside].std()))
