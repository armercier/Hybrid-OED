"""Can the 2-D Mindlin plate model explain the measured PMMA wavefield? (coarse pristine scan)

Data: band-passed (15-90 kHz) LDV vz (and vx, vy) of the coarse scan, dropouts removed, resampled
to dt_obs = 1/(16 f0), window 450-1250 us after the trigger.
Model: homogeneous Mindlin plate (10 mm PMMA, kappa matched to A0 at 40 kHz), 1 mm grid, sponge
edges (plate edges unknown in the scanner frame, >= 150 mm away, PMMA damping), mass-proportional
damping (Q at 40 kHz), vertical point forces at the sources found by locate_sources.py.
Source signatures: unknown, estimated per frequency by least squares (one complex filter per
source, both sources fired together) from vz; vx, vy then use the same filters and one real scale.
Fitted: velocity scale s (vp, vs x s: A0 speed), Q, source positions (+-12 mm), by coordinate search.

Usage:  python src/ldv_pmma/fit_mindlin_coarse.py
Output: ldv_pmma_results/mindlin_fit.{png,json}, mindlin_fit_traces.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "elastic"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from assess_coarse_scan import F_BAND, OUT, T_NOISE, load, to_grid  # noqa: E402
from locate_sources import DEFAULT, dropouts  # noqa: E402
from mindlin import a0_matched_kappa, mindlin_plate  # noqa: E402

F0 = 40e3
VP, VS, RHO, H = 2730.0, 1340.0, 1180.0, 10e-3     # PMMA values of the Salvus specimen
DX = 1e-3
T_WIN = (450e-6, 1250e-6)
SPONGE = 60


class Fit:
    def __init__(self, path=DEFAULT):
        t, xs, ys, ix, iy, comps = load(path)
        fs = 1 / (t[1] - t[0])
        bad = dropouts(comps, t < T_NOISE)
        sos = butter(4, F_BAND, btype="band", fs=fs, output="sos")
        self.dt_obs = 1 / (16 * F0)
        self.t = np.arange(0, T_WIN[1] - T_WIN[0], self.dt_obs)
        keep = ~bad
        self.R = np.c_[xs[ix], ys[iy]][keep]
        self.ix, self.iy, self.nxs, self.nys = ix[keep], iy[keep], len(xs), len(ys)
        self.xs, self.ys = xs, ys
        self.obs, self.noise_ratio = {}, {}
        for c in "xyz":
            v = sosfiltfilt(sos, comps[c][keep], axis=1)
            nrms2 = np.mean(v[:, t < T_NOISE] ** 2)
            self.obs[c] = CubicSpline(t, v, axis=1)(self.t + T_WIN[0]).T.astype(np.float32)   # (nt, nrec)
            self.noise_ratio[c] = float(nrms2 / np.mean(self.obs[c] ** 2))     # misfit floor from noise
        src = json.loads((OUT / "sources.json").read_text())
        self.src0 = np.array([[s["x_mm"], s["y_mm"]] for s in src.values()]) * 1e-3
        # model grid
        lo = np.minimum(self.R.min(0), self.src0.min(0)) - 0.03 - SPONGE * DX
        hi = np.maximum(self.R.max(0), self.src0.max(0)) + 0.03 + SPONGE * DX
        self.origin = lo
        self.nx, self.ny = (np.ceil((hi - lo) / DX).astype(int) + 1)
        self.dt = 0.6 * DX / (VP * 1.07 * np.sqrt(2.0))
        nst = int(np.ceil(self.t[-1] / self.dt)) + 2
        tt = np.arange(nst) * self.dt
        fr, tpk = 45e3, 30e-6
        a = (np.pi * fr * (tt - tpk)) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = self.t / self.dt
        self.i0 = jnp.asarray(np.floor(pos).astype(int))
        self.w = jnp.asarray(pos - np.floor(pos), jnp.float32)[:, None, None]
        self.rec_i = jnp.asarray((self.R - lo) / DX, jnp.float32)
        self.nf = 2 * len(self.t)
        self.freqs = np.fft.rfftfreq(self.nf, self.dt_obs)
        self.band = (self.freqs >= F_BAND[0]) & (self.freqs <= F_BAND[1])
        _, Hf = __import__("scipy.signal", fromlist=["sosfreqz"]).sosfreqz(
            butter(4, F_BAND, btype="band", fs=1 / self.dt_obs, output="sos"), worN=self.freqs, fs=1 / self.dt_obs)
        self.H2 = np.abs(Hf) ** 2
        one = jnp.ones((self.nx, self.ny), jnp.float32)

        def sim(s, damping, kappa, src_i):
            tr = mindlin_plate(VP * s * one, VS * s * one, RHO * one, H, src_i, self.stf, DX, DX, self.dt,
                               self.rec_i, sponge_cells=SPONGE, surface_inplane=True, kappa=kappa,
                               damping=damping)
            return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]      # (nt, nrec, 3): vz, vx, vy

        self._sim = jax.jit(sim)
        print(f"model grid {self.nx} x {self.ny}, {nst} steps, {len(self.R)} receivers, noise floor (misfit "
              f"from noise alone) z {self.noise_ratio['z']:.3f} x {self.noise_ratio['x']:.3f} "
              f"y {self.noise_ratio['y']:.3f}", flush=True)

    def simulate(self, s, Q, src):
        kappa = a0_matched_kappa(H, VP * s, VS * s, RHO, F0)
        damping = 2 * np.pi * F0 / Q
        return [np.asarray(self._sim(jnp.float32(s), jnp.float32(damping), jnp.float32(kappa),
                                     jnp.asarray((p - self.origin) / DX, jnp.float32))) for p in src]

    def predict(self, syns, comp="z", filters=None):
        """Least-squares source filters (one per source) from vz, applied to component comp."""
        k = {"z": 0, "x": 1, "y": 2}[comp]
        G = [np.fft.rfft(g[..., k], self.nf, axis=0) * self.H2[:, None] for g in syns]   # (nf, nrec)
        if filters is None:
            Gz = [np.fft.rfft(g[..., 0], self.nf, axis=0) * self.H2[:, None] for g in syns]
            O = np.fft.rfft(self.obs["z"], self.nf, axis=0)
            A = np.stack(Gz, -1)                                                 # (nf, nrec, nsrc)
            AhA = np.einsum("frs,frt->fst", A.conj(), A)
            Aho = np.einsum("frs,fr->fs", A.conj(), O)
            eps = 1e-3 * np.max(np.abs(AhA)) * np.eye(len(syns))
            filters = np.linalg.solve(AhA + eps, Aho[..., None])[..., 0]          # (nf, nsrc)
        P = sum(filters[:, i, None] * G[i] for i in range(len(syns)))
        return np.fft.irfft(P, self.nf, axis=0)[: len(self.t)], filters

    def misfit(self, pred, comp="z", scale=False):
        o = self.obs[comp]
        if scale:   # one real factor (sign/orientation and gain of the in-plane channels)
            a = np.sum(o * pred) / np.sum(pred**2)
            pred = a * pred
        return float(np.sum((pred - o) ** 2) / np.sum(o**2)), pred


def main():
    fit = Fit()
    hist = []

    def evaluate(s, Q, src, tag=""):
        t0 = time.perf_counter()
        syns = fit.simulate(s, Q, src)
        pred, _ = fit.predict(syns)
        J, _ = fit.misfit(pred)
        hist.append(dict(s=s, Q=Q, src=(src * 1e3).round(1).tolist(), misfit_z=J))
        print(f"{tag:8s} s {s:.3f} Q {Q:5.1f} src {np.round(src * 1e3, 1).tolist()} -> misfit z {J:.4f} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
        return J

    s, Q, src = 1.0, 40.0, fit.src0.copy()
    best = evaluate(s, Q, src, "start")
    for sweep in range(2):
        for cand in (0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05):
            if cand == s:
                continue
            J = evaluate(cand, Q, src, "speed")
            if J < best:
                best, s_new = J, cand
                s = s_new
        for cand in (15, 20, 25, 30, 40, 50, 70, 100):
            if cand == Q:
                continue
            J = evaluate(s, cand, src, "Q")
            if J < best:
                best, Q = J, cand
        for k in range(len(src)):
            step = 8e-3 if sweep == 0 else 3e-3
            improved = True
            while improved:
                improved = False
                for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    trial = src.copy()
                    trial[k] += d
                    J = evaluate(s, Q, trial, f"src{k + 1}")
                    if J < best:
                        best, src, improved = J, trial, True
    # final: all components, figures
    syns = fit.simulate(s, Q, src)
    pred_z, filters = fit.predict(syns)
    res = {"z": fit.misfit(pred_z)}
    for c in "xy":
        p, _ = fit.predict(syns, c, filters)
        res[c] = fit.misfit(p, c, scale=True)
    syn0 = fit.simulate(1.0, 40.0, fit.src0)
    p0, _ = fit.predict(syn0)
    summary = dict(best=dict(velocity_scale=s, Q=Q, sources_mm=(src * 1e3).round(1).tolist(),
                             a0_phase_velocity_40k=None),
                   misfit={c: res[c][0] for c in "xyz"}, misfit_start_z=fit.misfit(p0)[0],
                   noise_floor=fit.noise_ratio, n_receivers=len(fit.R), history=hist)
    # per-point normalised correlation (z)
    o, p = fit.obs["z"], res["z"][1]
    corr = np.sum(o * p, 0) / np.sqrt(np.sum(o**2, 0) * np.sum(p**2, 0))
    summary["corr_z"] = dict(median=float(np.median(corr)), p10=float(np.percentile(corr, 10)))
    (OUT / "mindlin_fit.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1))

    # ---- figure: snapshots obs / pred / residual (z, x, y) + correlation map
    ext = (fit.xs[0] * 1e3, fit.xs[-1] * 1e3, fit.ys[0] * 1e3, fit.ys[-1] * 1e3)
    grid = lambda v: to_grid(v, fit.ix, fit.iy, fit.nxs, fit.nys)
    e = np.sqrt(np.mean(fit.obs["z"] ** 2, 1))
    k1 = int(np.argmax(e))
    ks = [k1, k1 + int(80e-6 / fit.dt_obs)]
    fig, axes = plt.subplots(len(ks) * 3, 3, figsize=(13, 4.0 * 3 * len(ks)), layout="constrained")
    row = 0
    for c in "zxy":
        o, p = fit.obs[c], res[c][1]
        for k in ks:
            vmax = np.percentile(np.abs(o[k]), 99)
            for j, (lab, img) in enumerate((("measured", o[k]), ("Mindlin", p[k]), ("residual", o[k] - p[k]))):
                ax = axes[row, j]
                ax.imshow(grid(img), origin="lower", extent=ext, cmap="seismic", vmin=-vmax, vmax=vmax)
                ax.set_title(f"v{c} {lab}, t = {1e6 * (fit.t[k] + T_WIN[0]):.0f} us"
                             + (f"  (misfit {res[c][0]:.3f})" if j == 2 else ""), fontsize=9)
                ax.set_aspect("equal"), ax.tick_params(labelsize=6)
            row += 1
    fig.suptitle(f"Mindlin fit of the coarse scan: velocity scale {s:.2f}, Q {Q:.0f}; sources (mm) "
                 f"{np.round(src * 1e3).astype(int).tolist()}", fontsize=11)
    fig.savefig(OUT / "mindlin_fit.png", dpi=80)

    fig, axes = plt.subplots(5, 1, figsize=(13, 13), layout="constrained")
    im = None
    order = np.argsort(corr)
    picks = [order[-1], order[len(order) * 3 // 4], order[len(order) // 2], order[len(order) // 10]]
    for a, i in zip(axes[:4], picks):
        a.plot(1e6 * (fit.t + T_WIN[0]), fit.obs["z"][:, i] * 1e3, "k", lw=0.9, label="measured vz")
        a.plot(1e6 * (fit.t + T_WIN[0]), res["z"][1][:, i] * 1e3, "r", lw=0.9, label="Mindlin")
        a.set_title(f"point ({fit.R[i, 0] * 1e3:.0f}, {fit.R[i, 1] * 1e3:.0f}) mm, correlation {corr[i]:.2f}", fontsize=9)
        a.legend(fontsize=7), a.set_ylabel("mm/s")
    im = axes[4].imshow(grid(corr), origin="lower", extent=ext, cmap="viridis", vmin=0.5, vmax=1)
    axes[4].set_title(f"per-point correlation measured / Mindlin vz (median {np.median(corr):.2f})", fontsize=9)
    fig.colorbar(im, ax=axes[4])
    fig.savefig(OUT / "mindlin_fit_traces.png", dpi=85)


if __name__ == "__main__":
    main()
