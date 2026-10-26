"""LDV plate problem: data, Mindlin forward model and a source-free misfit, shared by the
inversions (and later the OED loop).

* data: out-of-plane velocity of the LDV scan in the scan-plane frame, 8-70 kHz
  band-pass, window t < tmax; points within ``near_source`` of the transducer and the
  scan row through it are excluded; a fixed random half is 'train', the rest 'test';
* forward: Mindlin plate (kappa matched to A0 at 39 kHz, damping ~ Q 40, sponge edges),
  normal point source with a broadband Ricker, optional rigid bonded body and/or
  mass / rotary-inertia maps, remat over time chunks;
* misfit: the unknown source wavelet and trigger delay are a filter F(omega) solved in
  closed form for every evaluation (variable projection, held fixed in the gradient):
      F = sum_r O_r conj(S_r) / (sum_r |S_r|^2 + eps),  L = ||F S - O||^2 / ||O||^2.
  The data band-pass is applied to the synthetics in the frequency domain (|H|^2,
  the zero-phase response of the forward-backward filter used on the data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import sosfreqz

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src" / "elastic", REPO / "src" / "ldv_assessment"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from fit_mindlin_ldv import RHO, VP_VS, load_ldv
from mindlin import a0_matched_kappa, mindlin_plate

OUT = REPO / "ldv_inversion_results"
CUBE_SALVUS = np.array([-0.020, -0.005])  # Salvus 3-D fit (5 mm search), for reference only


class LDVProblem:
    def __init__(self, tmax=650e-6, dx=0.5e-3, margin=0.08, sponge=0.05, near_source=0.04, vs=1340.0,
                 h=0.010, damping=6000.0, f_src=45e3, checkpoint_every=64, seed=0):
        P, t, obs, _, sos = load_ldv(tmax)
        fit = json.loads((REPO / "ldv_assessment" / "mindlin_ldv_fit.json").read_text())["best"]
        self.src = np.asarray(fit["src"])
        r = np.hypot(*(P - self.src).T)
        usable = (r > near_source) & (np.abs(P[:, 1] - self.src[1]) > 0.004)
        rng = np.random.default_rng(seed)
        split = rng.random(len(P)) < 0.5
        self.train = np.where(usable & split)[0]
        self.test = np.where(usable & ~split)[0]
        self.P, self.t, self.obs_all = P, t, obs
        self.behind_test = self.test[P[self.test, 0] > -0.035]
        self.before_test = self.test[P[self.test, 0] <= -0.035]

        self.dx, self.vs, self.h, self.damping = dx, vs, h, damping
        self.lo = P.min(0) - margin
        self.nx, self.ny = (np.round((P.max(0) + margin - self.lo) / dx).astype(int) + 1)
        self.X, self.Y = np.meshgrid(self.lo[0] + np.arange(self.nx) * dx, self.lo[1] + np.arange(self.ny) * dx,
                                     indexing="ij")
        self.kappa = a0_matched_kappa(h, VP_VS * vs, vs, RHO, 39e3)
        self.sponge_cells = int(round(sponge / dx))
        self.checkpoint_every = checkpoint_every
        self.dt = 0.6 * dx / (VP_VS * vs * np.sqrt(2.0))
        t_sim = np.arange(int(np.ceil(t[-1] / self.dt)) + 2) * self.dt
        a = (np.pi * f_src * (t_sim - 1.5 / f_src)) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = t / self.dt
        i0 = np.floor(pos).astype(int)
        self.i0, self.w = jnp.asarray(i0), jnp.asarray(pos - i0, jnp.float32)[:, None]
        self.nf = 2 * len(t)
        f = np.fft.rfftfreq(self.nf, t[1] - t[0])
        _, Hf = sosfreqz(sos, worN=f, fs=1 / (t[1] - t[0]))
        self.H2 = jnp.asarray(np.abs(Hf) ** 2, jnp.float32)[:, None]
        self.s_i = jnp.asarray((self.src - self.lo) / dx, jnp.float32)

    # ----------------------------------------------------------------------- forward
    def receivers(self, idx):
        return jnp.asarray((self.P[idx] - self.lo) / self.dx, jnp.float32)

    def simulate(self, rec_i, mass_scale=None, inertia_scale=None, rigid_mask=None, rigid_mass=0.0,
                 rigid_inertia=0.0):
        one = jnp.ones((self.nx, self.ny), jnp.float32)
        tr = mindlin_plate(VP_VS * self.vs * one, self.vs * one, RHO * one, self.h, self.s_i, self.stf, self.dx,
                           self.dx, self.dt, rec_i, sponge_cells=self.sponge_cells, kappa=self.kappa,
                           mass_scale=mass_scale, inertia_scale=inertia_scale, damping=self.damping,
                           rigid_mask=rigid_mask, rigid_added_mass=rigid_mass, rigid_added_inertia=rigid_inertia,
                           checkpoint_every=self.checkpoint_every)[..., 0]
        return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]  # LDV time axis, (nt, nrec)

    # ------------------------------------------------------------------------ misfit
    def spectra(self, syn):
        return jnp.fft.rfft(syn, self.nf, axis=0) * self.H2

    def source_filter(self, S, O):
        num = jnp.sum(O * jnp.conj(S), axis=1)
        den = jnp.sum(jnp.abs(S) ** 2, axis=1)
        return jax.lax.stop_gradient(num / (den + 1e-3 * jnp.max(den)))

    def predict(self, syn, O_fit):
        """Synthetics filtered with the closed-form source filter fitted to O_fit (same traces)."""
        S = self.spectra(syn)
        F = self.source_filter(S, O_fit)
        return jnp.fft.irfft(F[:, None] * S, self.nf, axis=0)[: len(self.t)], F

    def misfit(self, syn, obs):
        O = jnp.fft.rfft(obs, self.nf, axis=0)
        pred, _ = self.predict(syn, O)
        return jnp.sum((pred - obs) ** 2) / jnp.sum(obs**2)

    def evaluate(self, syn_all, idx_all):
        """Filter fitted on the train traces, misfit on test traces behind / before the cube.

        syn_all: synthetics (nt, len(idx_all)) for the receivers idx_all (train + test)."""
        syn_all, pos = np.asarray(syn_all), {k: i for i, k in enumerate(idx_all)}
        tr = [pos[k] for k in self.train]
        S = np.fft.rfft(syn_all, self.nf, axis=0) * np.asarray(self.H2)
        O = np.fft.rfft(self.obs_all[:, self.train], self.nf, axis=0)
        num, den = np.sum(O * np.conj(S[:, tr]), 1), np.sum(np.abs(S[:, tr]) ** 2, 1)
        pred = np.fft.irfft((num / (den + 1e-3 * den.max()))[:, None] * S, self.nf, axis=0)[: len(self.t)]
        out = {}
        for name, sel in (("behind", self.behind_test), ("before", self.before_test)):
            cols = [pos[k] for k in sel]
            o = self.obs_all[:, sel]
            out[name] = float(np.linalg.norm(pred[:, cols] - o) / np.linalg.norm(o))
        return out, pred
