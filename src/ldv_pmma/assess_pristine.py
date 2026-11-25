"""Assessment of the four single-source pristine LDV scans of the PMMA plate (Nov 2026).

Per source file: dropouts, band-limited SNR, did the source fire at every point, arrival-time
coherence (point-source moveout fit + per-point timing residual), amplitude drift along the scan
order, Mindlin fit (sponge edges, per-source filter, shared speed scale and Q, refined source
positions). Repeatability: the earlier two-source coarse scan vs the sum of the matching
single-source scans (same grid).

Usage:  python src/ldv_pmma/assess_pristine.py
Output: ldv_pmma_results/pristine/{overview.png, mindlin.png, repeatability.png, summary.json}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import butter, hilbert, sosfiltfilt, sosfreqz

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "elastic"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from assess_coarse_scan import F_BAND, REPO, T_NOISE, load, to_grid  # noqa: E402
from mindlin import a0_matched_kappa, mindlin_plate  # noqa: E402

DATA = REPO / "ldv_data_comprehensive"
FILES = [DATA / "pristine_mat" / f"excitation_{k}_pristine.mat" for k in (1, 2, 3, 4)]
COARSE = DATA / "Soruce_1_40kHz_1-5V_20avg_320nt_3125kHz.mat"
OUT = REPO / "ldv_pmma_results" / "pristine"
F0, VP, VS, RHO, H = 40e3, 2730.0, 1340.0, 1180.0, 10e-3
DX, SPONGE = 1e-3, 60
T_WIN = (450e-6, 1250e-6)


def prep(path):
    t, xs, ys, ix, iy, comps = load(path)
    fs = 1 / (t[1] - t[0])
    noise_i = t < T_NOISE
    raw_noise = {c: np.sqrt(np.mean(v[:, noise_i] ** 2, 1)) for c, v in comps.items()}
    bad = np.zeros(len(ix), bool)
    for c in "xyz":
        bad |= raw_noise[c] > 3 * np.median(raw_noise[c])
    sos = butter(4, F_BAND, btype="band", fs=fs, output="sos")
    vz = sosfiltfilt(sos, comps["z"], axis=1)
    nrms = np.sqrt(np.mean(vz[:, noise_i] ** 2, 1))
    sig = (t > T_WIN[0]) & (t < T_WIN[1])
    snr = np.abs(vz[:, sig]).max(1) / nrms
    return dict(t=t, xs=xs, ys=ys, ix=ix, iy=iy, R=np.c_[xs[ix], ys[iy]], vz=vz, bad=bad, snr=snr, nrms=nrms,
                ref=comps["z"].shape and load_ref(path), fs=fs)


def load_ref(path):
    import scipy.io as sio
    return sio.loadmat(path, variable_names=["ref1"])["ref1"]


def moveout(d):
    """Envelope-peak times, point-source fit t = t0 + |r - s| / vg; returns fit and per-point residual."""
    sel = (d["t"] > T_WIN[0]) & (d["t"] < 1100e-6)
    env = np.abs(hilbert(d["vz"][:, sel], axis=1))
    tp = d["t"][sel][env.argmax(1)]
    use = ~d["bad"] & (d["snr"] > 10)
    R = d["R"]
    # start from each planned source position (drawing, relative to the scan centre O) and keep the best
    O = 0.5 * (R.min(0) + R.max(0))
    planned = O + np.array([[0, -170], [220, 12], [0, 170], [-220, -13]]) * 1e-3
    fun = lambda p, m: (p[2] + np.hypot(R[m, 0] - p[0], R[m, 1] - p[1]) / p[3] - tp[m]) * 1e6
    sols = [least_squares(lambda p: fun(p, use), [*s0, tp[use].min() - 0.1e-3, 1300.0], loss="soft_l1",
                          f_scale=5.0, bounds=([-np.inf, -np.inf, 0, 800], [np.inf, np.inf, 1e-3, 2500]))
            for s0 in planned]
    sol = min(sols, key=lambda r: r.cost)
    res = np.full(len(tp), np.nan)
    res[use] = fun(sol.x, use)
    return sol.x, res, tp


class Model:
    def __init__(self, data, srcs):
        self.data = data
        R = data[0]["R"]
        self.dt_obs = 1 / (16 * F0)
        self.t = np.arange(0, T_WIN[1] - T_WIN[0], self.dt_obs)
        self.keep = [~d["bad"] for d in data]
        self.obs = [CubicSpline(d["t"], d["vz"][k], axis=1)(self.t + T_WIN[0]).T.astype(np.float32)
                    for d, k in zip(data, self.keep)]
        allp = np.vstack([R, srcs])
        lo = allp.min(0) - 0.03 - SPONGE * DX
        hi = allp.max(0) + 0.03 + SPONGE * DX
        self.origin = lo
        self.nx, self.ny = (np.ceil((hi - lo) / DX).astype(int) + 1)
        self.dt = 0.6 * DX / (VP * 1.07 * np.sqrt(2.0))
        nst = int(np.ceil(self.t[-1] / self.dt)) + 2
        tt = np.arange(nst) * self.dt
        a = (np.pi * 45e3 * (tt - 30e-6)) ** 2
        stf = jnp.asarray((1 - 2 * a) * np.exp(-a), jnp.float32)
        pos = self.t / self.dt
        i0 = jnp.asarray(np.floor(pos).astype(int))
        w = jnp.asarray(pos - np.floor(pos), jnp.float32)[:, None]
        rec_i = jnp.asarray((R - lo) / DX, jnp.float32)
        self.nf = 2 * len(self.t)
        f = np.fft.rfftfreq(self.nf, self.dt_obs)
        _, Hf = sosfreqz(butter(4, F_BAND, btype="band", fs=1 / self.dt_obs, output="sos"), worN=f,
                         fs=1 / self.dt_obs)
        self.H2 = (np.abs(Hf) ** 2)[:, None]
        one = jnp.ones((self.nx, self.ny), jnp.float32)

        def sim(s, damping, kappa, src_i):
            tr = mindlin_plate(VP * s * one, VS * s * one, RHO * one, H, src_i, stf, DX, DX, self.dt, rec_i,
                               sponge_cells=SPONGE, kappa=kappa, damping=damping)[..., 0]
            return (1 - w) * tr[i0] + w * tr[i0 + 1]

        self._sim = jax.jit(sim)
        print(f"model grid {self.nx} x {self.ny}, {nst} steps", flush=True)

    def predict(self, k, s, Q, src):
        kappa = a0_matched_kappa(H, VP * s, VS * s, RHO, F0)
        g = np.asarray(self._sim(jnp.float32(s), jnp.float32(2 * np.pi * F0 / Q), jnp.float32(kappa),
                                 jnp.asarray((src - self.origin) / DX, jnp.float32)))[:, self.keep[k]]
        G = np.fft.rfft(g, self.nf, axis=0) * self.H2
        O = np.fft.rfft(self.obs[k], self.nf, axis=0)
        num, den = np.sum(O * G.conj(), 1), np.sum(np.abs(G) ** 2, 1)
        F = num / (den + 1e-3 * den.max())
        p = np.fft.irfft(F[:, None] * G, self.nf, axis=0)[: len(self.t)]
        return p, F

    def misfit(self, k, s, Q, src):
        p, _ = self.predict(k, s, Q, src)
        o = self.obs[k]
        return float(np.sum((p - o) ** 2) / np.sum(o**2)), p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, summary = [], {}
    for k, f in enumerate(FILES, 1):
        d = prep(f)
        (sx, sy, t0, vg), res, tp = moveout(d)
        d.update(src=np.array([sx, sy]), t0=t0, vg=vg, tres=res, tp=tp)
        good = ~d["bad"]
        # firing: points with no coherent arrival = SNR below 5 outside dropouts
        nofire = good & (d["snr"] < 5)
        # drift along acquisition order: SNR-normalised peak / smooth spatial trend
        pk = np.abs(d["vz"][:, (d["t"] > T_WIN[0]) & (d["t"] < T_WIN[1])]).max(1)
        r = np.hypot(*(d["R"] - d["src"]).T)
        A = np.c_[np.ones_like(r), np.log(r), r]
        coef, *_ = np.linalg.lstsq(A[good], np.log(pk[good]), rcond=None)
        d["amp_res"] = np.log(pk) - A @ coef
        ref = d["ref"]
        ref_ok = float(np.min([np.corrcoef(ref[0], ref[i])[0, 1] for i in range(0, len(ref), 25)]))
        summary[f.stem] = dict(
            dropouts=int(d["bad"].sum()), points=len(good),
            snr_median=float(np.median(d["snr"][good])), snr_p10=float(np.percentile(d["snr"][good], 10)),
            snr_min=float(d["snr"][good].min()), points_snr_below_5=int(nofire.sum()),
            noise_band_rms_mm_s=float(np.median(d["nrms"][good]) * 1e3),
            source_mm=[sx * 1e3, sy * 1e3], t0_us=t0 * 1e6, group_velocity=vg,
            arrival_residual_rms_us=float(np.sqrt(np.nanmean(d["tres"] ** 2))),
            arrival_residual_p95_abs_us=float(np.nanpercentile(np.abs(d["tres"]), 95)),
            amplitude_drift_first_vs_last_10pct=float(np.exp(np.mean(d["amp_res"][good][-len(good) // 10:])
                                                             - np.mean(d["amp_res"][good][: len(good) // 10]))),
            trigger_gate_min_corr=ref_ok)
        data.append(d)
        print(f.stem, json.dumps(summary[f.stem]), flush=True)

    # ---- Mindlin: per-source positions (coordinate search), then shared speed / Q
    srcs = np.array([d["src"] for d in data])
    m = Model(data, srcs)
    s, Q = 0.99, 40.0
    J = [m.misfit(k, s, Q, srcs[k])[0] for k in range(4)]
    print("start", np.round(J, 4), flush=True)
    for step in (8e-3, 3e-3, 1e-3):
        for k in range(4):
            improved = True
            while improved:
                improved = False
                for dd in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    tr = srcs[k] + dd
                    Jk = m.misfit(k, s, Q, tr)[0]
                    if Jk < J[k]:
                        J[k], srcs[k], improved = Jk, tr, True
        for sc in (s - 0.01, s + 0.01, s - 0.005, s + 0.005):
            Jt = [m.misfit(k, sc, Q, srcs[k])[0] for k in range(4)]
            if np.mean(Jt) < np.mean(J):
                J, s = Jt, sc
        for qc in (Q * 0.8, Q * 1.25):
            Jt = [m.misfit(k, s, qc, srcs[k])[0] for k in range(4)]
            if np.mean(Jt) < np.mean(J):
                J, Q = Jt, qc
        print(f"step {step * 1e3:.0f} mm: s {s:.3f} Q {Q:.0f} misfit {np.round(J, 4)} src {np.round(srcs * 1e3, 1).tolist()}",
              flush=True)
    preds = [m.misfit(k, s, Q, srcs[k])[1] for k in range(4)]
    corr, jitter = [], []
    for k in range(4):
        o, p = m.obs[k], preds[k]
        corr.append(np.sum(o * p, 0) / np.sqrt(np.sum(o**2, 0) * np.sum(p**2, 0)))
        # per-point time shift (cross-correlation peak, sub-sample by parabola)
        O, P = np.fft.rfft(o, m.nf, axis=0), np.fft.rfft(p, m.nf, axis=0)
        xc = np.fft.irfft(O * P.conj(), m.nf * 8, axis=0)
        lag = np.argmax(np.r_[xc[-40:], xc[:41]], 0) - 40
        jitter.append(lag * m.dt_obs / 8 * 1e6)
    for k, f in enumerate(FILES):
        summary[f.stem].update(mindlin_misfit=J[k], mindlin_source_mm=(srcs[k] * 1e3).round(1).tolist(),
                               corr_median=float(np.median(corr[k])), corr_p10=float(np.percentile(corr[k], 10)),
                               time_shift_vs_model_std_us=float(np.std(jitter[k])),
                               noise_floor_misfit=float(np.mean(data[k]["nrms"][m.keep[k]] ** 2)
                                                        / np.mean(m.obs[k] ** 2)))
    summary["mindlin_shared"] = dict(velocity_scale=s, Q=Q)

    # ---- repeatability: coarse two-source scan vs sum of the matching single-source scans
    dc = prep(COARSE)
    sc_src = json.loads((REPO / "ldv_pmma_results" / "sources.json").read_text())
    sc_src = np.array([[v["x_mm"], v["y_mm"]] for v in sc_src.values()]) * 1e-3
    pair = [int(np.argmin(np.hypot(*(srcs - p).T))) for p in sc_src]
    both = ~dc["bad"] & ~data[pair[0]]["bad"] & ~data[pair[1]]["bad"]
    w = (dc["t"] > T_WIN[0]) & (dc["t"] < T_WIN[1])
    summ = data[pair[0]]["vz"] + data[pair[1]]["vz"]
    a = dc["vz"][both][:, w]
    b = summ[both][:, w]
    gain = float(np.sum(a * b) / np.sum(b**2))
    rep = float(np.sum((a - gain * b) ** 2) / np.sum(a**2))
    # global time stretch (temperature) correction
    best = (rep, 1.0)
    for st in np.linspace(0.99, 1.01, 81):
        tt = T_WIN[0] + (dc["t"][w] - T_WIN[0]) * st
        bs = CubicSpline(dc["t"], summ[both], axis=1)(tt)
        g = np.sum(a * bs) / np.sum(bs**2)
        r = np.sum((a - g * bs) ** 2) / np.sum(a**2)
        if r < best[0]:
            best = (float(r), float(st))
    noise_rep = float((np.mean(dc["nrms"][both] ** 2) + np.mean(data[pair[0]]["nrms"][both] ** 2)
                       + np.mean(data[pair[1]]["nrms"][both] ** 2)) / np.mean(a**2))
    summary["repeatability_coarse_vs_sum"] = dict(
        files=[FILES[pair[0]].stem, FILES[pair[1]].stem], gain=gain, misfit=rep, misfit_after_stretch=best[0],
        best_time_stretch=best[1], noise_only_expectation=noise_rep)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # ---- figures
    xs, ys = data[0]["xs"], data[0]["ys"]
    ext = (xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3)
    g = lambda v, d: to_grid(v, d["ix"], d["iy"], len(xs), len(ys))
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), layout="constrained")
    for k, d in enumerate(data):
        good = ~d["bad"]
        kk = int(np.argmax(np.sqrt(np.mean(d["vz"][good] ** 2, 0))))
        vmax = np.percentile(np.abs(d["vz"][good, kk]), 99)
        panels = [
            (f"vz snapshot t = {d['t'][kk] * 1e6:.0f} us", g(np.where(good, d["vz"][:, kk], np.nan), d),
             dict(cmap="seismic", vmin=-vmax, vmax=vmax)),
            (f"SNR (band-passed), median {np.median(d['snr'][good]):.0f}", g(np.where(good, d["snr"], np.nan), d),
             dict(cmap="viridis", vmin=0, vmax=np.percentile(d["snr"][good], 98))),
            (f"arrival time residual (us), rms {summary[FILES[k].stem]['arrival_residual_rms_us']:.1f}",
             g(d["tres"], d), dict(cmap="RdBu_r", vmin=-15, vmax=15)),
            (f"Mindlin correlation, median {np.median(corr[k]):.2f}",
             g(np.where(good, np.r_[np.zeros(0)], np.nan) if False else _scatter(corr[k], m.keep[k]), d),
             dict(cmap="viridis", vmin=0.6, vmax=1)),
        ]
        for j, (title, img, kw) in enumerate(panels):
            ax = axes[k, j]
            im = ax.imshow(img, origin="lower", extent=ext, **kw)
            ax.plot(*(srcs[k] * 1e3), "*", color="#ff9f1c", ms=12, mec="k", clip_on=False)
            ax.set_xlim(min(ext[0], srcs[k][0] * 1e3) - 10, max(ext[1], srcs[k][0] * 1e3) + 10)
            ax.set_ylim(min(ext[2], srcs[k][1] * 1e3) - 10, max(ext[3], srcs[k][1] * 1e3) + 10)
            ax.set_aspect("equal"), ax.tick_params(labelsize=6)
            ax.set_title(f"{FILES[k].stem}: {title}" if j == 0 else title, fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.7)
        ax = axes[k, 4]
        order = np.arange(len(good))
        ax.plot(order[good], np.exp(d["amp_res"][good]), ".", ms=1.5, color="#1a3fa0")
        sm = np.convolve(np.exp(d["amp_res"][good]), np.ones(100) / 100, mode="same")
        ax.plot(order[good], sm, color="r", lw=1)
        ax.set_ylim(0.5, 1.5), ax.set_xlabel("acquisition order (point index)", fontsize=7)
        ax.set_title("peak amplitude / spreading-law fit vs scan order (drift)", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.savefig(OUT / "overview.png", dpi=75)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), layout="constrained")
    for k in range(4):
        o, p = m.obs[k], preds[k]
        i = int(np.argsort(corr[k])[len(corr[k]) // 2])
        R = data[k]["R"][m.keep[k]]
        axes[k].plot(1e6 * (m.t + T_WIN[0]), o[:, i] * 1e3, "k", lw=0.9, label="measured vz")
        axes[k].plot(1e6 * (m.t + T_WIN[0]), p[:, i] * 1e3, "r", lw=0.9, label="Mindlin")
        axes[k].set_title(f"{FILES[k].stem}: median-correlation point ({R[i, 0] * 1e3:.0f}, {R[i, 1] * 1e3:.0f}) mm, "
                          f"corr {corr[k][i]:.2f}; misfit {J[k]:.3f}", fontsize=9)
        axes[k].legend(fontsize=7)
    fig.savefig(OUT / "mindlin.png", dpi=85)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout="constrained")
    kk = int(np.argmax(np.sqrt(np.mean(a**2, 0))))
    tk = np.flatnonzero(w)[kk]
    vmax = np.percentile(np.abs(dc["vz"][both, tk]), 99)
    for ax, (title, v) in zip(axes, (("two-source scan (13 Nov)", dc["vz"][:, tk]),
                                     (f"{FILES[pair[0]].stem} + {FILES[pair[1]].stem} (x {gain:.2f})", gain * summ[:, tk]),
                                     (f"difference (misfit {rep:.3f}, noise only {noise_rep:.3f})",
                                      dc["vz"][:, tk] - gain * summ[:, tk]))):
        ax.imshow(g(np.where(both, v, np.nan), dc), origin="lower", extent=ext, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}, t = {dc['t'][tk] * 1e6:.0f} us", fontsize=9), ax.set_aspect("equal")
    fig.savefig(OUT / "repeatability.png", dpi=85)


def _scatter(v, keep):
    out = np.full(len(keep), np.nan)
    out[keep] = v
    return out


if __name__ == "__main__":
    main()
