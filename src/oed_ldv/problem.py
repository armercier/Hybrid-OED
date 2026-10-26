"""OED problem for the LDV plate: Mindlin forward model, mass-map FWI, illumination design.

Geometry, source position, time axis and data band follow the LDV scan (scan-plane frame,
src/ldv_assessment, src/ldv_inversion). Receivers are a differentiable argument (metres).

* forward: Mindlin plate, kappa matched to A0, damping ~ Q 40, sponge edges, normal point
  source with a broadband Ricker; ``dx`` sets the grid (0.5 mm for FWI, 2 mm for design);
* model: log mass scale on a coarse grid over the scan area (zero within ``source_mask``
  of the transducer), bilinear to the FD grid;
* misfit: closed-form source filter (wavelet + trigger delay) in the frequency domain
  (variable projection, fixed in the gradient), data band |H|^2 applied to synthetics;
* illumination: diag of the Gauss-Newton matrix w.r.t. the log-mass map from Rademacher
  probes (paper, Appendix A), score = var(h / mean) - beta log(mean) over the scan area.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfreqz

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src" / "elastic", REPO / "src" / "ldv_assessment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from fit_mindlin_ldv import RHO, VP_VS
from mindlin import a0_matched_kappa, mindlin_plate

SCAN = np.load(REPO / "ldv_assessment" / "mindlin_ldv_fit.npz")
SCAN_P, SCAN_T = SCAN["P"], SCAN["t"]
SOURCE = np.asarray(json.loads((REPO / "ldv_assessment" / "mindlin_ldv_fit.json").read_text())["best"]["src"])
BAND = butter(4, [8e3, 70e3], btype="band", fs=1 / (SCAN_T[1] - SCAN_T[0]), output="sos")


class OEDProblem:
    def __init__(self, dx=0.5e-3, margin=0.08, sponge=0.05, vs=1340.0, h=0.010, damping=6000.0, f_src=45e3,
                 spacing=2.5e-3, source_mask=0.03, near_source=0.04, tv=1e-4, beta=0.05, checkpoint_every=64,
                 tmax=650e-6):
        self.t = SCAN_T[SCAN_T < tmax]
        self.src, self.dx, self.vs, self.h, self.damping = SOURCE, dx, vs, h, damping
        self.scan_lo, self.scan_hi = SCAN_P.min(0), SCAN_P.max(0)
        self.near_source, self.tv_weight, self.beta = near_source, tv, beta
        self.lo = self.scan_lo - margin
        self.nx, self.ny = (np.round((self.scan_hi + margin - self.lo) / dx).astype(int) + 1)
        self.kappa = a0_matched_kappa(h, VP_VS * vs, vs, RHO, 39e3)
        self.sponge_cells = int(round(sponge / dx))
        self.checkpoint_every = checkpoint_every
        self.dt = 0.6 * dx / (VP_VS * vs * np.sqrt(2.0))
        t_sim = np.arange(int(np.ceil(self.t[-1] / self.dt)) + 2) * self.dt
        a = (np.pi * f_src * (t_sim - 1.5 / f_src)) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = self.t / self.dt
        i0 = np.floor(pos).astype(int)
        self.i0, self.w = jnp.asarray(i0), jnp.asarray(pos - i0, jnp.float32)[:, None]
        self.nf = 2 * len(self.t)
        _, Hf = sosfreqz(BAND, worN=np.fft.rfftfreq(self.nf, self.t[1] - self.t[0]), fs=1 / (self.t[1] - self.t[0]))
        self.H2 = jnp.asarray(np.abs(Hf) ** 2, jnp.float32)[:, None]
        self.s_i = jnp.asarray((self.src - self.lo) / dx, jnp.float32)
        # coarse model grid over the scan area
        self.spacing = spacing
        nc = np.round((self.scan_hi - self.scan_lo) / spacing).astype(int) + 1
        self.Xc, self.Yc = np.meshgrid(self.scan_lo[0] + np.arange(nc[0]) * spacing,
                                       self.scan_lo[1] + np.arange(nc[1]) * spacing, indexing="ij")
        self.model_mask = jnp.asarray(np.hypot(self.Xc - self.src[0], self.Yc - self.src[1]) > source_mask,
                                      jnp.float32)
        X, Y = np.meshgrid(self.lo[0] + np.arange(self.nx) * dx, self.lo[1] + np.arange(self.ny) * dx, indexing="ij")
        self.coords = [jnp.asarray((X - self.scan_lo[0]) / spacing, jnp.float32),
                       jnp.asarray((Y - self.scan_lo[1]) / spacing, jnp.float32)]
        self.n_model = tuple(int(n) for n in nc)

    # ------------------------------------------------------------------- forward
    def mass_field(self, log_mass):
        return jnp.exp(map_coordinates(self.model_mask * log_mass, self.coords, order=1, mode="constant", cval=0.0))

    def simulate(self, log_mass, rec_m):
        one = jnp.ones((self.nx, self.ny), jnp.float32)
        rec_i = (jnp.asarray(rec_m, jnp.float32) - jnp.asarray(self.lo, jnp.float32)) / self.dx
        tr = mindlin_plate(VP_VS * self.vs * one, self.vs * one, RHO * one, self.h, self.s_i, self.stf, self.dx,
                           self.dx, self.dt, rec_i, sponge_cells=self.sponge_cells, kappa=self.kappa,
                           mass_scale=self.mass_field(log_mass), damping=self.damping,
                           checkpoint_every=self.checkpoint_every)[..., 0]
        return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]

    # -------------------------------------------------------------------- misfit
    def predict(self, syn, obs):
        """Source-filtered synthetics (filter fitted to obs on the same traces) and the filter."""
        S = jnp.fft.rfft(syn, self.nf, axis=0) * self.H2
        O = jnp.fft.rfft(obs, self.nf, axis=0)
        num, den = jnp.sum(O * jnp.conj(S), axis=1), jnp.sum(jnp.abs(S) ** 2, axis=1)
        F = jax.lax.stop_gradient(num / (den + 1e-3 * jnp.max(den)))
        return jnp.fft.irfft(F[:, None] * S, self.nf, axis=0)[: len(self.t)], F

    def misfit(self, log_mass, rec_m, obs):
        pred, _ = self.predict(self.simulate(log_mass, rec_m), obs)
        return jnp.sum((pred - obs) ** 2) / jnp.sum(obs**2)

    def misfit_groups(self, log_mass, rec_m, obs, groups, n_groups):
        """Misfit over several acquisitions (one simulation for all points, one source filter
        per acquisition; ``groups`` gives the acquisition index of every point)."""
        syn = self.simulate(log_mass, rec_m)
        S = jnp.fft.rfft(syn, self.nf, axis=0) * self.H2
        O = jnp.fft.rfft(obs, self.nf, axis=0)
        onehot = jax.nn.one_hot(groups, n_groups)                          # (nrec, n_groups)
        num = (O * jnp.conj(S)) @ onehot                                    # (nf, n_groups)
        den = (jnp.abs(S) ** 2) @ onehot
        F = jax.lax.stop_gradient(num / (den + 1e-3 * jnp.max(den, axis=0, keepdims=True)))
        pred = jnp.fft.irfft(F[:, groups] * S, self.nf, axis=0)[: len(self.t)]
        err = jnp.sum((pred - obs) ** 2, axis=0) @ onehot
        energy = jnp.sum(obs**2, axis=0) @ onehot
        return jnp.mean(err / energy)

    def fwi_loss_groups(self, log_mass, rec_m, obs, groups, n_groups):
        return self.misfit_groups(log_mass, rec_m, obs, groups, n_groups) + self.tv_weight * self.tv(log_mass)

    def tv(self, log_mass):
        m = self.model_mask * log_mass
        gx, gy = m[1:, :-1] - m[:-1, :-1], m[:-1, 1:] - m[:-1, :-1]
        return jnp.sum(jnp.sqrt(gx**2 + gy**2 + 1e-6) - 1e-3)

    def fwi_loss(self, log_mass, rec_m, obs):
        return self.misfit(log_mass, rec_m, obs) + self.tv_weight * self.tv(log_mass)

    # -------------------------------------------------------------- illumination
    def make_probes(self, key, n_probes, n_rec):
        nt = len(self.t)
        return jax.random.rademacher(key, (n_probes, nt, n_rec)).astype(jnp.float32) / np.sqrt(nt * n_rec)

    def illumination(self, rec_m, log_mass, probes):
        out, pull = jax.vjp(lambda m: self.simulate(m, rec_m), log_mass)
        sens = jax.lax.map(lambda q: pull(q)[0], probes)
        h = jnp.mean(sens**2, axis=0)
        n0, n1 = h.shape
        hp = jnp.pad(h, 1, mode="edge")
        h = sum(hp[i:i + n0, j:j + n1] for i in range(3) for j in range(3)) / 9.0
        region = self.model_mask > 0.5
        count = jnp.sum(region)
        mean = jnp.sum(jnp.where(region, h, 0.0)) / count
        var = jnp.sum(jnp.where(region, (h / mean - 1.0) ** 2, 0.0)) / count
        return var - self.beta * jnp.log(mean)

    # ------------------------------------------------- waveform-misfit design
    def waveform_objective(self, K, step):
        """Lab-consistent waveform-misfit design objective (needs ``checkpoint_every`` > 0).

        The measured data are extrapolated to nearby points with the current model,
        d~(p) = d_meas + F_p(m_t) - F_{p_t}(m_t) (exact at p_t), then K normalised-gradient
        FWI updates (step size ~ the Adam step of the real FWI) are unrolled and
        rematerialised; the objective is the misfit after them, differentiated w.r.t. p.
        """
        def objective(rec, m_t, rec_t, d_meas):
            d_tilde = d_meas + self.simulate(m_t, rec) - jax.lax.stop_gradient(self.simulate(m_t, rec_t))

            @jax.checkpoint
            def update(m, _):
                g = jax.grad(self.fwi_loss)(m, rec, d_tilde)
                return m - step * g / (jnp.sqrt(jnp.mean(g**2)) + 1e-12), None

            m_K, _ = jax.lax.scan(update, m_t, None, length=K)
            return self.misfit(m_K, rec, d_tilde)
        return objective

    # ----------------------------------------------------------------- receivers
    def project(self, rec_m):
        """Keep receivers inside the scan area and outside the near-source disc."""
        r = jnp.clip(jnp.asarray(rec_m), jnp.asarray(self.scan_lo), jnp.asarray(self.scan_hi))
        d = r - jnp.asarray(self.src)
        dist = jnp.linalg.norm(d, axis=1, keepdims=True)
        return jnp.where(dist < self.near_source, jnp.asarray(self.src) + d / jnp.maximum(dist, 1e-9) * self.near_source, r)

    # -------------------------------------------------------------------- metrics
    def cube_metrics(self, log_mass, cube_xy):
        """Localisation error of the mass-map peak and contrast at the true cube."""
        lm = np.asarray(self.model_mask) * np.asarray(log_mass)
        k = np.unravel_index(np.argmax(lm), lm.shape)
        peak = np.array([self.Xc[k], self.Yc[k]])
        inside = (np.abs(self.Xc - cube_xy[0]) <= 0.005) & (np.abs(self.Yc - cube_xy[1]) <= 0.005)
        outside = (~inside) & (np.asarray(self.model_mask) > 0.5)
        return dict(peak_xy=peak.tolist(), loc_error_mm=float(1e3 * np.linalg.norm(peak - cube_xy)),
                    peak_mass=float(np.exp(lm.max())),
                    contrast=float((lm[inside].mean() - lm[outside].mean()) / (lm[outside].std() + 1e-9)))
