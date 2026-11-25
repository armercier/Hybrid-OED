"""Quality check of the weekend defect (scatter) scan of the PMMA plate, 3 mm grid (Nov 2026).

Checks per file: dropouts, SNR, wrong-source points (file identical to the previous one at the
same point), arrival-time check against the planned source positions; grid match with the
pristine scan; first defect - pristine difference maps on the shared points.
Outputs: ldv_pmma_results/scatter/{quality.png, diff_maps.png, summary.json}

Usage:  python src/ldv_pmma/assess_scatter.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, hilbert, sosfiltfilt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "ldv_data_comprehensive"
OUT = REPO / "ldv_pmma_results" / "scatter"
FS = 3.125e6
SOS = butter(4, (15e3, 90e3), btype="band", fs=FS, output="sos")
T_NOISE = 480e-6
SRC_UV = np.array([[0, -170], [220, 12], [0, 170], [-220, -13]], float)
DEF_UV = {"D1": (-75, 47, 10), "D2": (70, 52, 10), "D3": (0, -13, 10), "D4": (-85, -53, 8), "M1": (80, -48, 10)}
T0, VG = 508e-6, 1336e3   # shot delay (s) and A0 group velocity (mm/s)


def load(path):
    d = sio.loadmat(path, variable_names=["XYZ", "vib_z", "t"])
    R = d["XYZ"][:, :2] * 1e3
    vz = sosfiltfilt(SOS, d["vib_z"], axis=1)
    return R, vz, d["t"][0]


def stats(vz, t):
    n = t < T_NOISE
    nrms = np.sqrt(np.mean(vz[:, n] ** 2, 1))
    snr = np.abs(vz[:, t > 500e-6]).max(1) / nrms
    return nrms, snr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = {k: load(DATA / "scatt_mat" / f"excitation_{k}_scatt.mat") for k in (1, 2, 3, 4)}
    P = {k: load(DATA / "pristine_mat" / f"excitation_{k}_pristine.mat") for k in (1, 2, 3, 4)}
    t = S[1][2]
    summ = {}
    R = S[1][0]
    for k in (2, 3, 4):
        assert np.allclose(S[k][0], R, atol=0.05), "scatter files do not share the same points"
    xs, ys = np.unique(R[:, 0].round(1)), np.unique(R[:, 1].round(1))
    summ["grid"] = dict(n=len(R), nx=len(xs), ny=len(ys), dx=float(np.median(np.diff(xs))), dy=float(np.median(np.diff(ys))),
                        x=[float(xs[0]), float(xs[-1])], y=[float(ys[0]), float(ys[-1])])
    Rp = P[1][0]
    O = 0.5 * (Rp.min(0) + Rp.max(0))
    # the 3 mm grid does not contain the pristine points (offset ~1 mm): interpolate the scatter
    # data (cubic spline in space, 8 points per A0 wavelength) onto the pristine points
    from scipy.spatial import cKDTree
    from scipy.ndimage import map_coordinates
    dist, _ = cKDTree(R).query(Rp)
    x0, y0 = R[:, 0].min(), R[:, 1].min()
    ix = np.round((R[:, 0] - x0) / 3).astype(int); iy = np.round((R[:, 1] - y0) / 3).astype(int)
    nxg, nyg = ix.max() + 1, iy.max() + 1
    have = np.zeros((nyg, nxg), bool); have[iy, ix] = True
    fx, fy = (Rp[:, 0] - x0) / 3, (Rp[:, 1] - y0) / 3
    inside = (fx >= 1) & (fx <= nxg - 2) & (fy >= 1) & (fy <= nyg - 2)
    for j in np.where(inside)[0]:       # all 4x4 neighbours must be measured
        cx, cy = int(np.floor(fx[j])), int(np.floor(fy[j]))
        inside[j] = have[cy - 1:cy + 3, cx - 1:cx + 3].all()
    TW = (t > 440e-6) & (t < 1500e-6)
    def interp(vz):
        g = np.zeros((nyg, nxg, TW.sum())); g[iy, ix] = vz[:, TW]
        gi, gj = np.where(~have)
        for a_, b_ in zip(gi, gj):      # fill the few missing edge points with the nearest row value
            nb = [(a_ + da, b_ + db) for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= a_ + da < nyg and 0 <= b_ + db < nxg and have[a_ + da, b_ + db]]
            if nb: g[a_, b_] = np.mean([g[q] for q in nb], 0)
        nt = TW.sum(); jj = np.where(inside)[0]
        out = np.full((len(Rp), nt), np.nan)
        cy = np.repeat(fy[jj], nt); cx = np.repeat(fx[jj], nt); ct = np.tile(np.arange(nt), len(jj))
        out[jj] = map_coordinates(g, [cy, cx, ct], order=3, mode="nearest").reshape(len(jj), nt)
        return out
    summ["pristine_match"] = dict(note="scatter grid offset from pristine grid; scatter interpolated onto pristine points",
                                  median_offset_mm=float(np.median(dist)), pristine_points_usable=int(inside.sum()))
    SRC = O + SRC_UV
    TA = lambda RR, s: T0 + np.hypot(*(RR - SRC[s]).T) / VG
    nz, snr, bad = {}, {}, {}
    for k in (1, 2, 3, 4):
        nz[k], snr[k] = stats(S[k][1], t)
        bad[k] = nz[k] > 3 * np.median(nz[k])
        pn, ps = stats(P[k][1], t)
        summ[f"ex{k}"] = dict(noise_median_um_s=float(np.median(nz[k]) * 1e6), snr_median=float(np.median(snr[k][~bad[k]])),
                              snr_p5=float(np.percentile(snr[k][~bad[k]], 5)), dropouts=int(bad[k].sum()),
                              n_snr_below_5=int((snr[k] < 5).sum()),
                              pristine_noise_median_um_s=float(np.median(pn) * 1e6),
                              pristine_snr_median=float(np.median(ps[pn < 3 * np.median(pn)])))
    w = t > 500e-6
    stale = {}
    for k in (2, 3, 4):
        a, b = S[k][1], S[k - 1][1]
        r = np.mean((a[:, w] - b[:, w]) ** 2, 1) / (nz[k] ** 2 + nz[k - 1] ** 2)
        ok = (nz[k] < 2 * np.median(nz[k])) & (nz[k - 1] < 2 * np.median(nz[k - 1]))
        stale[k] = (r < 3) & ok
        # onset test: correct source vs previous source
        E = np.abs(hilbert(a, axis=1))
        sel = t > 520e-6
        on = np.array([t[sel][np.argmax(e[sel] > 0.3 * e[sel].max())] for e in E])
        good = np.minimum(TA(R, 0), TA(R, k - 1))
        prev = np.minimum(TA(R, 0), TA(R, k - 2)) if k > 2 else TA(R, 0)
        dec = (np.abs(good - prev) > 30e-6) & ~bad[k]
        off = np.median((on - good)[dec])
        cg = np.abs(on - off - good) < np.abs(on - off - prev)
        summ[f"ex{k}"].update(stale_identical_to_previous=int(stale[k].sum()), onset_decidable=int(dec.sum()),
                              onset_matches_correct_pct=float(cg[dec].mean() * 100))
    tw = t[TW]
    noise_tw = tw < T_NOISE
    sig_tw = (tw > 500e-6) & (tw < 1300e-6)
    I = {k: interp(S[k][1]) for k in (1, 2, 3, 4)}
    Pt = {k: P[k][1][:, TW] for k in (1, 2, 3, 4)}
    pn = {k: np.sqrt(np.mean(P[k][1][:, t < T_NOISE] ** 2, 1)) for k in P}
    pbad = {k: pn[k] > 3 * np.median(pn[k]) for k in P}
    # drift (temperature): global time shift and gain, file 1 (S1 only), early window before any scattering reaches far points
    ok = inside & ~pbad[1]
    a, b = I[1][ok][:, sig_tw], Pt[1][ok][:, sig_tw]
    from scipy.interpolate import CubicSpline
    best = None
    for sh in np.linspace(-3e-6, 3e-6, 61):
        bs = CubicSpline(tw[sig_tw], b, axis=1, extrapolate=True)(tw[sig_tw] + sh)
        g = np.sum(a * bs) / np.sum(bs ** 2); r = np.sum((a - g * bs) ** 2) / np.sum(a ** 2)
        if best is None or r < best[0]: best = (r, g, sh)
    summ["drift_ex1_vs_pristine"] = dict(shift_us=float(best[2] * 1e6), gain=float(best[1]), resid_after=float(best[0]))
    SH, GA = best[2], best[1]
    rel_all = {}
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    U, V = Rp[:, 0] - O[0], Rp[:, 1] - O[1]
    for k in (1, 2, 3, 4):
        okk = inside & ~pbad[k]
        if k > 1:
            pp = pn[k - 1]
            rp = np.mean((P[k][1][:, w] - P[k - 1][1][:, w]) ** 2, 1) / (pn[k] ** 2 + pp ** 2)
            okk &= ~(rp < 3)
        b = CubicSpline(tw, Pt[k], axis=1)(np.clip(tw + SH, tw[0], tw[-1])) * GA
        a = np.nan_to_num(I[k])
        d = a - b
        rel = np.sqrt(np.mean(d[:, sig_tw] ** 2, 1) / np.mean(b[:, sig_tw] ** 2, 1))
        nrel = np.sqrt(np.mean(d[:, noise_tw] ** 2, 1) / np.mean(b[:, sig_tw] ** 2, 1))
        rel_all[k] = (rel, okk, d)
        far = okk & np.all([np.hypot(U - du, V - dv) > 40 for du, dv, _ in DEF_UV.values()], 0)
        near = okk & np.any([np.hypot(U - du, V - dv) < 12 for du, dv, _ in DEF_UV.values()], 0)
        summ[f"ex{k}"].update(diff_points=int(okk.sum()), diff_rel_median=float(np.median(rel[okk])),
                              diff_rel_far_from_defects=float(np.median(rel[far])),
                              diff_rel_near_defects=float(np.median(rel[near])),
                              noise_rel_median=float(np.median(nrel[okk])))
        for row, (val, lab, vmax) in enumerate(((snr[k], "SNR, all 3 mm points", None),
                                                (rel, "|defect - pristine| / |pristine| (500-1300 us)", 0.5))):
            axx = ax[row, k - 1]
            if row == 0:
                sc = axx.scatter(R[:, 0] - O[0], R[:, 1] - O[1], c=val, s=4, marker="s", vmin=0, vmax=np.percentile(val, 95))
            else:
                sc = axx.scatter(U[okk], V[okk], c=val[okk], s=12, marker="s", cmap="magma", vmin=0, vmax=vmax)
                axx.scatter(U[~okk], V[~okk], c="#bbb", s=4, marker="x")
            for nm, (du, dv, dd) in DEF_UV.items():
                axx.add_patch(plt.Circle((du, dv), dd / 2 + 3, fill=False, ec="c", lw=1))
                axx.text(du + 7, dv + 7, nm, color="c", fontsize=8)
            axx.set_xlim(-160, 160), axx.set_ylim(-110, 110), axx.set_aspect("equal")
            axx.set_title(f"ex{k}: {lab}", fontsize=9)
            plt.colorbar(sc, ax=axx, shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "diff_maps.png", dpi=90)
    # snapshots of the difference field, file 1 (S1 only)
    rel, okk, d = rel_all[1]
    fig, ax = plt.subplots(2, 4, figsize=(21, 8))
    for i, tt in enumerate((640e-6, 720e-6, 800e-6, 900e-6)):
        it = np.argmin(np.abs(tw - tt))
        b = Pt[1][:, it]; v = np.percentile(np.abs(b[okk]), 98)
        for row, (val, lab) in enumerate(((b, "pristine"), (d[:, it], "defect - pristine (same colour scale)"))):
            ax[row, i].scatter(U[okk], V[okk], c=val[okk], s=12, marker="s", cmap="RdBu_r", vmin=-v, vmax=v)
            for nm, (du, dv, dd) in DEF_UV.items():
                ax[row, i].add_patch(plt.Circle((du, dv), dd / 2 + 3, fill=False, ec="k", lw=0.8))
                ax[row, i].text(du + 7, dv + 7, nm, fontsize=8)
            ax[row, i].set_aspect("equal"), ax[row, i].set_title(f"S1, t = {tt*1e6:.0f} us: {lab}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "diff_snapshots_ex1.png", dpi=90)
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
