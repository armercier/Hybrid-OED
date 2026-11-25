"""Step 3: calibrate the Mindlin plate model on the pristine PMMA scan (4 sources, real LDV data).

Data (prep_dataset.py, working copy): band-passed vz, 625 kHz, window T_WIN. Masked: dropouts,
wrong-source points, and the upper half of files 3-4 (wrong-source points there are not all detectable).
Model: homogeneous Mindlin plate, 10 mm PMMA (velocity scale s on vp, vs; kappa matched to A0 at
40 kHz), 1 mm grid, sponge edges (plate edges unknown in the scanner frame and outside the window
for the direct waves), mass-proportional damping (Q at 40 kHz). The four PZT blocks (10 x 10 mm,
7 mm high, back face, 5.3 g) are added-mass patches on their footprint (PZT_MODEL; the bilayer
stiff-layer model of invert.bilayer_ratios fitted slightly worse) and are present in every
simulation; a vertical point force at the centre of the firing block.
Files: file 1 = S1, file k = S1 + Sk. One unknown source signature per transducer (complex filter
per frequency), estimated jointly from all four files by least squares (variable projection).
Fitted by coordinate search: s, Q, the four source positions.

Usage:  python src/ldv_pmma/calibrate_pristine.py [--scan pristine|scatter] [--t1 950]
Output: ldv_pmma_results/calib/calib_{scan}.json, calib_{scan}_snapshots.png, calib_{scan}_traces.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfreqz

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "elastic"))
sys.path.insert(0, str(REPO / "src" / "salvus_specimen"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from invert import bilayer_ratios  # noqa: E402
from mindlin import a0_matched_kappa, mindlin_plate  # noqa: E402

PREP = REPO / "ldv_work" / "prepared"
OUT = REPO / "ldv_pmma_results" / "calib"
F0, F_BAND = 40e3, (15e3, 90e3)
DX, SPONGE, MARGIN = 1e-3, 60, 0.03
KS = (1, 2, 3, 4)
PZT_MODEL = "mass"     # "mass" (block mass on its 10 x 10 mm footprint) or "bilayer"


class Calib:
    def __init__(self, scan="pristine", t_win=(450e-6, 950e-6), pzt=True):
        d = np.load(PREP / f"{scan}.npz")
        geo = json.loads((PREP / "geometry.json").read_text())
        P = geo["plate"]
        self.vp, self.vs, self.rho, self.H = P["vp"], P["vs"], P["rho"], P["thickness_mm"] * 1e-3
        self.pzt = geo["pzt"] if pzt else None
        self.scan = scan
        t = d["t"]
        sel = (t >= t_win[0]) & (t <= t_win[1])
        self.t_abs = t[sel]
        self.t = self.t_abs - t_win[0]
        self.dt_obs = t[1] - t[0]
        R = d["R"] * 1e-3
        O = 0.5 * (R.min(0) + R.max(0))
        self.R, self.O = R, O
        self.mask = {}
        for k in KS:
            m = ~d[f"bad{k}"]
            if scan == "pristine" and k in (3, 4):
                m &= R[:, 1] < O[1]
            self.mask[k] = m
        use = np.any([self.mask[k] for k in KS], 0)
        self.use = use
        self.Ru = R[use]
        self.m = {k: self.mask[k][use].astype(np.float32) for k in KS}
        self.obs = {k: d[f"vz{k}"][use][:, sel].T.astype(np.float64) for k in KS}      # (nt, nrec)
        nz = {k: d[f"noise{k}"][use] for k in KS}
        self.noise_floor = {k: float(np.sum(self.m[k] * nz[k] ** 2) * len(self.t)
                                     / np.sum(self.m[k] * self.obs[k] ** 2)) for k in KS}
        self.src0 = np.array([[geo[scan]["sources"][f"S{k}"]["x"], geo[scan]["sources"][f"S{k}"]["y"]] for k in KS]) * 1e-3
        lo = np.minimum(self.Ru.min(0), self.src0.min(0)) - MARGIN - SPONGE * DX
        hi = np.maximum(self.Ru.max(0), self.src0.max(0)) + MARGIN + SPONGE * DX
        self.origin = lo
        self.nx, self.ny = (np.ceil((hi - lo) / DX).astype(int) + 1)
        self.X, self.Y = np.meshgrid(lo[0] + np.arange(self.nx) * DX, lo[1] + np.arange(self.ny) * DX, indexing="ij")
        self.dt = 0.6 * DX / (self.vp * 1.07 * 1.08 * np.sqrt(2.0))
        nst = int(np.ceil(self.t[-1] / self.dt)) + 2
        tt = np.arange(nst) * self.dt
        a = (np.pi * 45e3 * (tt - 30e-6)) ** 2
        self.stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = self.t / self.dt
        self.i0 = jnp.asarray(np.floor(pos).astype(int))
        self.w = jnp.asarray(pos - np.floor(pos), jnp.float32)[:, None]
        self.rec_i = jnp.asarray((self.Ru - lo) / DX, jnp.float32)
        self.nf = 2 * len(self.t)
        self.freqs = np.fft.rfftfreq(self.nf, self.dt_obs)
        _, Hf = sosfreqz(butter(4, F_BAND, btype="band", fs=1 / self.dt_obs, output="sos"), worN=self.freqs,
                         fs=1 / self.dt_obs)
        self.H2 = (np.abs(Hf) ** 2)[:, None]

        def sim(vp, vs, h, mass, damping, kappa, src_i):
            one = jnp.ones((self.nx, self.ny), jnp.float32)
            tr = mindlin_plate(vp * one, vs * one, self.rho * one, h, src_i, self.stf, DX, DX, self.dt, self.rec_i,
                               sponge_cells=SPONGE, kappa=kappa, mass_scale=mass, damping=damping)[..., 0]
            return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]          # (nt, nrec)

        self._sim = jax.jit(sim)
        print(f"[{scan}] grid {self.nx} x {self.ny} (1 mm), {nst} steps, {len(self.Ru)} receivers, "
              f"window {t_win[0] * 1e6:.0f}-{t_win[1] * 1e6:.0f} us, noise floor "
              + " ".join(f"ex{k} {self.noise_floor[k]:.3f}" for k in KS), flush=True)

    def fields(self, src, s):
        h = np.full((self.nx, self.ny), self.H, np.float32)
        mass = np.ones((self.nx, self.ny), np.float32)
        if self.pzt is not None:
            # default: added mass of the block on its footprint (best and physical: misfit 0.041 vs 0.044 for the
            # bilayer stiff-layer model and 0.042 without blocks); PZT_MODEL = "bilayer" for the composite plate
            half = self.pzt["size_mm"][0] * 5e-4
            hp = self.pzt["height_mm"] * 1e-3
            if PZT_MODEL == "bilayer":
                r_h, r_m = bilayer_ratios(self.vp * s, self.vs * s, self.rho, self.H, hp)
            else:
                r_h, r_m = 1.0, (self.rho * self.H + self.pzt["rho"] * hp) / (self.rho * self.H)
            for p in src:
                inside = (np.abs(self.X - p[0]) <= half) & (np.abs(self.Y - p[1]) <= half)
                h[inside], mass[inside] = self.H * r_h, r_m
        return jnp.asarray(h), jnp.asarray(mass)

    def simulate(self, s, Q, src):
        kappa = a0_matched_kappa(self.H, self.vp * s, self.vs * s, self.rho, F0)
        h, mass = self.fields(src, s)
        out = []
        for p in src:
            out.append(np.asarray(self._sim(jnp.float32(self.vp * s), jnp.float32(self.vs * s), h, mass,
                                            jnp.float32(2 * np.pi * F0 / Q), jnp.float32(kappa),
                                            jnp.asarray((p - self.origin) / DX, jnp.float32)), np.float64))
        return out

    def predict(self, syns):
        """Joint least-squares source signatures: file 1 = f1 G1, file k = f1 G1 + fk Gk."""
        G = [np.fft.rfft(g, self.nf, axis=0) * self.H2 for g in syns]
        Ob = {k: np.fft.rfft(self.obs[k], self.nf, axis=0) for k in KS}
        nfreq = len(self.freqs)
        AhA = np.zeros((nfreq, 4, 4), complex); Aho = np.zeros((nfreq, 4), complex)
        for k in KS:
            cols = [0] if k == 1 else [0, k - 1]
            m = self.m[k][None, :]
            for a in cols:
                Aho[:, a] += np.sum(m * G[a].conj() * Ob[k], 1)
                for b in cols:
                    AhA[:, a, b] += np.sum(m * G[a].conj() * G[b], 1)
        eps = 1e-4 * np.max(np.abs(AhA), axis=(1, 2))[:, None, None] * np.eye(4) + 1e-30 * np.eye(4)
        F = np.linalg.solve(AhA + eps, Aho[..., None])[..., 0]
        pred = {}
        for k in KS:
            P = F[:, 0:1] * G[0] + (0 if k == 1 else F[:, k - 1:k] * G[k - 1])
            pred[k] = np.fft.irfft(P, self.nf, axis=0)[: len(self.t)]
        return pred, F

    def misfit(self, pred, tsel=None):
        tsel = slice(None) if tsel is None else tsel
        per = {}
        num = den = 0.0
        for k in KS:
            r = np.sum(self.m[k] * (pred[k][tsel] - self.obs[k][tsel]) ** 2)
            o = np.sum(self.m[k] * self.obs[k][tsel] ** 2)
            per[k] = float(r / o); num += r; den += o
        return float(num / den), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="pristine")
    ap.add_argument("--t1", type=float, default=950.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cal = Calib(a.scan, (450e-6, a.t1 * 1e-6))
    hist = []

    def evaluate(s, Q, src, tag):
        t0 = time.perf_counter()
        pred, _ = cal.predict(cal.simulate(s, Q, src))
        J, per = cal.misfit(pred)
        hist.append(dict(tag=tag, s=s, Q=Q, src_mm=(src * 1e3).round(1).tolist(), misfit=J, per_file=per))
        print(f"{tag:6s} s {s:.3f} Q {Q:5.1f} src {np.round(src * 1e3, 1).tolist()} -> {J:.4f} "
              f"[{' '.join(f'{v:.3f}' for v in per.values())}] ({time.perf_counter() - t0:.0f} s)", flush=True)
        return J

    s, Q, src = 1.0, 40.0, cal.src0.copy()
    best = evaluate(s, Q, src, "start")
    for sweep in range(2):
        for cand in ((0.97, 0.98, 0.99, 1.01, 1.02, 1.03) if sweep == 0 else (s - 0.005, s + 0.005)):
            J = evaluate(cand, Q, src, "speed")
            if J < best:
                best, s = J, cand
        for cand in ((20, 30, 50, 70) if sweep == 0 else (Q * 0.85, Q * 1.15)):
            J = evaluate(s, cand, src, "Q")
            if J < best:
                best, Q = J, cand
        step = 4e-3 if sweep == 0 else 1.5e-3
        for k in range(4):
            improved = True
            while improved:
                improved = False
                for dd in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    trial = src.copy(); trial[k] += dd
                    J = evaluate(s, Q, trial, f"S{k + 1}")
                    if J < best:
                        best, src, improved = J, trial, True
    # final evaluation, with and without the PZT blocks
    syns = cal.simulate(s, Q, src)
    pred, F = cal.predict(syns)
    J, per = cal.misfit(pred)
    wins = {}
    for lo, hi in ((450, 650), (650, 800), (800, 950), (950, 1300)):
        tsel = (cal.t_abs >= lo * 1e-6) & (cal.t_abs < hi * 1e-6)
        if tsel.any():
            wins[f"{lo}-{hi}us"] = cal.misfit(pred, tsel)[0]
    nopzt = Calib(a.scan, (450e-6, a.t1 * 1e-6), pzt=False)
    J_nopzt = nopzt.misfit(nopzt.predict(nopzt.simulate(s, Q, src))[0])[0]
    corr = {}
    for k in KS:
        o, p = cal.obs[k], pred[k]
        c = np.sum(o * p, 0) / np.sqrt(np.sum(o ** 2, 0) * np.sum(p ** 2, 0) + 1e-30)
        corr[k] = c
    summ = dict(scan=a.scan, window_us=[450, a.t1], velocity_scale=s, Q=Q,
                sources_mm={f"S{k}": (src[k - 1] * 1e3).round(1).tolist() for k in KS},
                sources_start_mm={f"S{k}": (cal.src0[k - 1] * 1e3).round(1).tolist() for k in KS},
                misfit=J, misfit_per_file={f"ex{k}": v for k, v in per.items()}, misfit_per_window=wins,
                misfit_without_pzt_blocks=J_nopzt, noise_floor={f"ex{k}": v for k, v in cal.noise_floor.items()},
                corr_median={f"ex{k}": float(np.median(corr[k][cal.m[k] > 0])) for k in KS},
                receivers={f"ex{k}": int(cal.m[k].sum()) for k in KS}, history=hist)
    (OUT / f"calib_{a.scan}.json").write_text(json.dumps(summ, indent=1))
    np.savez(OUT / f"calib_{a.scan}_filters.npz", F=F, freqs=cal.freqs, s=s, Q=Q, src=src)
    print(json.dumps({k: v for k, v in summ.items() if k != "history"}, indent=1))

    # ---- figures: snapshots (measured / model / residual) per file, and traces
    U, V = (cal.Ru[:, 0] - cal.O[0]) * 1e3, (cal.Ru[:, 1] - cal.O[1]) * 1e3
    fig, ax = plt.subplots(4, 6, figsize=(24, 13))
    for r, k in enumerate(KS):
        e = np.sqrt(np.mean((cal.m[k] * cal.obs[k]) ** 2, 1)); it = [int(np.argmax(e)), int(np.argmax(e)) + int(90e-6 / cal.dt_obs)]
        m = cal.m[k] > 0
        for j, i in enumerate(it):
            v = np.percentile(np.abs(cal.obs[k][i, m]), 98)
            for c, (lab, val) in enumerate((("measured", cal.obs[k][i]), ("Mindlin", pred[k][i]),
                                            ("residual", cal.obs[k][i] - pred[k][i]))):
                axx = ax[r, 3 * j + c]
                axx.scatter(U[m], V[m], c=val[m], s=9 if a.scan == "pristine" else 2, marker="s", cmap="RdBu_r", vmin=-v, vmax=v)
                axx.set_aspect("equal"), axx.set_xticks([]), axx.set_yticks([])
                axx.set_title(f"ex{k} {lab} t={cal.t_abs[i] * 1e6:.0f} us" + (f" (misfit {per[k]:.3f})" if c == 2 else ""), fontsize=8)
    fig.suptitle(f"Mindlin calibration on the {a.scan} scan: s {s:.3f}, Q {Q:.0f}, misfit {J:.3f} "
                 f"(without PZT blocks {J_nopzt:.3f})")
    fig.tight_layout()
    fig.savefig(OUT / f"calib_{a.scan}_snapshots.png", dpi=70)
    fig, ax = plt.subplots(4, 2, figsize=(16, 12))
    for r, k in enumerate(KS):
        m = np.where(cal.m[k] > 0)[0]
        order = m[np.argsort(corr[k][m])]
        for j, i in enumerate((order[len(order) // 2], order[len(order) // 10])):
            ax[r, j].plot(cal.t_abs * 1e6, cal.obs[k][:, i] * 1e3, "k", lw=0.9, label="measured")
            ax[r, j].plot(cal.t_abs * 1e6, pred[k][:, i] * 1e3, "r", lw=0.9, label="Mindlin")
            ax[r, j].set_title(f"ex{k} ({U[i]:.0f}, {V[i]:.0f}) mm, corr {corr[k][i]:.2f} "
                               f"({'median' if j == 0 else '10th pct'})", fontsize=9)
            ax[r, j].set_ylabel("vz (mm/s)")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"calib_{a.scan}_traces.png", dpi=75)


if __name__ == "__main__":
    main()
