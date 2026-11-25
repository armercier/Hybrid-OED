"""Registered defect - pristine difference of the PMMA 3-D LDV scans (Nov 2026).

The plate moved between the pristine (6 mm grid) and defect (3 mm grid) scans. The pristine point p
corresponds to the defect-scan location  R(theta) (p - O) + O + T  (O = pristine grid centre),
with a small per-file delay tau_k. theta, T, tau_k are fitted on the direct waves (520-820 us) of
all four files (wavefield registration), then the defect scan is spline-interpolated at those
locations (3 mm grid = 8 points per A0 wavelength) and subtracted from the pristine (gain per file).
Pristine points with the wrong source (identical to the previous file) and dropouts are masked.

Outputs (ldv_pmma_results/scatter/):
  registration.json      theta, T, tau_k, misfits
  diff_maps_registered.png   per file: relative difference energy map + windows
  diff_snapshots_registered.png  per file: pristine and difference snapshots
  anim_diff_ex{k}.gif    pristine | defect | difference animation per file

Usage:  python src/ldv_pmma/scatter_maps.py [--no-anim]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import minimize

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from assess_scatter import DATA, DEF_UV, FS, T_NOISE, load

OUT = Path(__file__).resolve().parents[2] / "ldv_pmma_results" / "scatter"
KS = (1, 2, 3, 4)


def main():
    S = {k: load(DATA / f"scatt_mat/excitation_{k}_scatt.mat") for k in KS}
    P = {k: load(DATA / f"pristine_mat/excitation_{k}_pristine.mat") for k in KS}
    t = S[1][2]; R = S[1][0]; Rp = P[1][0]; O = 0.5 * (Rp.min(0) + Rp.max(0))
    x0, y0 = R[:, 0].min(), R[:, 1].min()
    ix = np.round((R[:, 0] - x0) / 3).astype(int); iy = np.round((R[:, 1] - y0) / 3).astype(int)
    nyg, nxg = iy.max() + 1, ix.max() + 1
    have = np.zeros((nyg, nxg), bool); have[iy, ix] = True
    TW = (t > 440e-6) & (t < 1500e-6); tw = t[TW]; nt = TW.sum()
    G = {}
    for k in KS:
        g = np.zeros((nyg, nxg, nt)); g[iy, ix] = S[k][1][:, TW]
        for a_, b_ in zip(*np.where(~have)):       # fill the missing edge points from their neighbours
            nb = [(a_ + da, b_ + db) for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= a_ + da < nyg and 0 <= b_ + db < nxg and have[a_ + da, b_ + db]]
            if nb:
                g[a_, b_] = np.mean([g[q] for q in nb], 0)
        G[k] = spline_filter(g, order=3)
    # pristine masks
    pn = {k: np.sqrt(np.mean(P[k][1][:, t < T_NOISE] ** 2, 1)) for k in KS}
    w = t > 500e-6
    okp = {}
    for k in KS:
        ok = pn[k] < 2 * np.median(pn[k])
        if k > 1:
            rp = np.mean((P[k][1][:, w] - P[k - 1][1][:, w]) ** 2, 1) / (pn[k] ** 2 + pn[k - 1] ** 2)
            ok &= rp >= 3
        okp[k] = ok
    sn = {k: np.sqrt(np.mean(S[k][1][:, t < T_NOISE] ** 2, 1)) for k in KS}
    sbad = np.zeros((nyg, nxg), bool)
    for k in KS:
        sbad[iy, ix] |= sn[k] > 3 * np.median(sn[k])

    def locate(q, k, idx):
        th, tx, ty = q[:3]
        d = Rp[idx] - O; c, s = np.cos(th), np.sin(th)
        X = O[0] + c * d[:, 0] - s * d[:, 1] + tx; Y = O[1] + s * d[:, 0] + c * d[:, 1] + ty
        fx, fy = (X - x0) / 3, (Y - y0) / 3
        inside = (fx >= 1) & (fx <= nxg - 2) & (fy >= 1) & (fy <= nyg - 2)
        return fx, fy, inside

    def sample(q, k, idx, win):
        fx, fy, inside = locate(q, k, idx)
        idx, fx, fy = idx[inside], fx[inside], fy[inside]
        it = np.where(win)[0] + q[2 + k] * 1e-6 * FS
        n, m = len(idx), len(it)
        a = map_coordinates(G[k], [np.repeat(fy, m), np.repeat(fx, m), np.tile(it, n)], order=3, prefilter=False,
                            mode="nearest").reshape(n, m)
        return idx, a

    WD = (tw > 520e-6) & (tw < 820e-6)
    sub = {k: np.where(okp[k])[0][::2] for k in KS}

    def misfit(q, ret=False):
        tot = den = 0; out = {}
        for k in KS:
            idx, a = sample(q, k, sub[k], WD)
            b = P[k][1][idx][:, TW][:, WD]
            g = np.sum(a * b) / np.sum(b * b); r = np.sum((a - g * b) ** 2)
            tot += r; den += np.sum(a * a); out[k] = (float(r / np.sum(a * a)), float(g))
        return (tot / den, out) if ret else tot / den

    best = None
    for th in np.radians(np.arange(-1.5, 1.51, 0.5)):
        for tx in (-4, 0, 4):
            for ty in (-4, 0, 4):
                q = np.array([th, tx, ty, 0.4, 0.4, 0.4, 0.4]); m = misfit(q)
                if best is None or m < best[0]:
                    best = (m, q)
    r = minimize(misfit, best[1], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=2500))
    q = r.x; m, per = misfit(q, True)
    reg = dict(rotation_deg=float(np.degrees(q[0])), shift_mm=[float(q[1]), float(q[2])],
               delay_us={f"ex{k}": float(q[2 + k]) for k in KS}, misfit_direct=float(m),
               per_file={f"ex{k}": dict(resid=per[k][0], gain=per[k][1]) for k in KS},
               convention="pristine point p -> defect-scan location R(theta)(p-O)+O+T, O = pristine grid centre",
               O=[float(O[0]), float(O[1])])
    print(json.dumps(reg, indent=1))
    json.dump(reg, open(OUT / "registration.json", "w"), indent=1)

    # differences on all usable pristine points, full window
    U, V = Rp[:, 0] - O[0], Rp[:, 1] - O[1]
    allw = np.ones(nt, bool)
    sig = (tw > 500e-6) & (tw < 1300e-6)
    D, B, A, OK = {}, {}, {}, {}
    summ = {}
    for k in KS:
        idx, a = sample(q, k, np.arange(len(Rp)), allw)
        fx, fy, _ = locate(q, k, idx)
        near_bad = np.array([sbad[int(round(y)), int(round(x))] for x, y in zip(fx, fy)])
        b = P[k][1][idx][:, TW]
        g = per[k][1]
        ok = okp[k][idx] & ~near_bad
        d = a - g * b
        Bk = np.full((len(Rp), nt), np.nan); Ak = Bk.copy(); Dk = Bk.copy()
        Bk[idx], Ak[idx], Dk[idx] = g * b, a, d
        okk = np.zeros(len(Rp), bool); okk[idx] = ok
        D[k], B[k], A[k], OK[k] = Dk, Bk, Ak, okk
        far = okk & np.all([np.hypot(U - du, V - dv) > 40 for du, dv, _ in DEF_UV.values()], 0)
        e_b = np.nanmean(Bk[far][:, sig] ** 2)
        noise = np.nanmean(Dk[okk][:, tw < T_NOISE] ** 2)
        wins = [(500, 650), (650, 800), (800, 1000), (1000, 1300)]
        summ[f"ex{k}"] = dict(points=int(okk.sum()),
                              far_rel_diff={f"{lo}-{hi}us": float(np.sqrt(np.nanmean(Dk[far][:, (tw > lo * 1e-6) & (tw < hi * 1e-6)] ** 2)
                                                                       / np.nanmean(Bk[far][:, (tw > lo * 1e-6) & (tw < hi * 1e-6)] ** 2)))
                                            for lo, hi in wins},
                              noise_rel=float(np.sqrt(noise / e_b)))
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(OUT / "diff_registered_summary.json", "w"), indent=1)

    # maps: RMS of the difference over 500-1300 us, normalised by the far-field pristine RMS of that file
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    for k, axx in zip(KS, ax.ravel()):
        okk = OK[k]
        ref = np.sqrt(np.nanmean(B[k][okk][:, sig] ** 2))
        val = np.sqrt(np.nanmean(D[k][:, sig] ** 2, 1)) / ref
        sc = axx.scatter(U[okk], V[okk], c=val[okk], s=30, marker="s", cmap="magma", vmin=0, vmax=np.nanpercentile(val[okk], 98))
        axx.scatter(U[~okk], V[~okk], c="#bbb", s=5, marker="x")
        for nm, (du, dv, dd) in DEF_UV.items():
            axx.add_patch(plt.Circle((du, dv), dd / 2 + 2, fill=False, ec="c", lw=1.2))
            axx.text(du + 7, dv + 7, nm, color="c", fontsize=9, fontweight="bold")
        axx.set_aspect("equal"), axx.set_xlim(-155, 155), axx.set_ylim(-105, 105)
        axx.set_title(f"ex{k}: RMS(defect - pristine), 500-1300 us, / RMS(pristine)", fontsize=10)
        plt.colorbar(sc, ax=axx, shrink=0.8)
    fig.suptitle("Registered difference (plate moved: rotation %.2f deg, shift (%.1f, %.1f) mm). Grey x = masked"
                 % (reg["rotation_deg"], *reg["shift_mm"]))
    fig.tight_layout()
    fig.savefig(OUT / "diff_maps_registered.png", dpi=85)

    # snapshots
    times = (620e-6, 700e-6, 780e-6, 880e-6)
    fig, ax = plt.subplots(4, 4, figsize=(20, 14))
    for r_, k in enumerate(KS):
        okk = OK[k]
        for c_, tt in enumerate(times):
            it = np.argmin(np.abs(tw - tt))
            v = np.nanpercentile(np.abs(B[k][okk][:, it]), 98)
            axx = ax[r_, c_]
            axx.scatter(U[okk], V[okk], c=D[k][okk, it], s=26, marker="s", cmap="RdBu_r", vmin=-v / 2, vmax=v / 2)
            for nm, (du, dv, dd) in DEF_UV.items():
                axx.add_patch(plt.Circle((du, dv), dd / 2 + 2, fill=False, ec="k", lw=0.8))
                axx.text(du + 7, dv + 7, nm, fontsize=8)
            axx.set_aspect("equal"), axx.set_xlim(-155, 155), axx.set_ylim(-105, 105)
            axx.set_title(f"ex{k}  t = {tt * 1e6:.0f} us: defect - pristine (scale = half pristine)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "diff_snapshots_registered.png", dpi=75)

    if "--no-anim" in sys.argv:
        return
    # animations pristine | defect | difference on the pristine 6 mm grid
    xs, ys = np.unique(Rp[:, 0].round(1)), np.unique(Rp[:, 1].round(1))
    jx = np.searchsorted(xs, Rp[:, 0].round(1)); jy = np.searchsorted(ys, Rp[:, 1].round(1))
    ext = (xs[0] - 3 - O[0], xs[-1] + 3 - O[0], ys[0] - 3 - O[1], ys[-1] + 3 - O[1])
    fr = np.where((tw > 480e-6) & (tw < 1300e-6))[0][::13]          # ~4 us
    for k in KS:
        okk = OK[k]
        def grid(M, i):
            g = np.full((len(ys), len(xs)), np.nan); g[jy[okk], jx[okk]] = M[okk, i] * 1e3; return g
        v = np.nanpercentile(np.abs(B[k][okk][:, fr]), 98) * 1e3
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), dpi=70)
        ims = []
        for axx, lab, vm in zip(axs, ("pristine", "defect scan (registered)", "difference x2"), (v, v, v / 2)):
            ims.append(axx.imshow(np.zeros((len(ys), len(xs))), origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm,
                                  interpolation="bilinear"))
            for nm, (du, dv, dd) in DEF_UV.items():
                axx.add_patch(plt.Circle((du, dv), dd / 2 + 2, fill=False, ec="k", lw=0.6))
            axx.set_title(lab, fontsize=9), axx.set_xticks([]), axx.set_yticks([])
        ttl = fig.suptitle("")
        fig.tight_layout()

        def upd(i):
            for im, M in zip(ims, (B[k], A[k], D[k])):
                im.set_data(grid(M, fr[i]))
            ttl.set_text(f"ex{k}   t = {tw[fr[i]] * 1e6:.0f} us")
            return ims

        FuncAnimation(fig, upd, frames=len(fr)).save(OUT / f"anim_diff_ex{k}.gif", writer=PillowWriter(fps=12))
        plt.close(fig)
        print(OUT / f"anim_diff_ex{k}.gif")


if __name__ == "__main__":
    main()
