"""Locate the PZT sources in the pristine and defect (scatter) 3-D LDV scans of the PMMA plate.

Each file k > 1 holds S1 + Sk (S1 is never switched off), so Sk alone = file k - file 1 within
the same scan. Arrival time = envelope peak of the band-passed vz in the direct-wave window;
fit t = t0 + |r - s| / v for the source position s (t0, v free).
Prints the positions (scanner frame, mm) and saves ldv_pmma_results/scatter/sources_scans.json.

Usage:  python src/ldv_pmma/locate_sources_scans.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import hilbert

from assess_scatter import DATA, SRC_UV, T_NOISE, load

OUT = Path(__file__).resolve().parents[2] / "ldv_pmma_results" / "scatter"


def fit_source(R, v, t, guess):
    nrms = np.sqrt(np.mean(v[:, t < T_NOISE] ** 2, 1))
    sel = (t > 520e-6) & (t < 1000e-6)
    env = np.abs(hilbert(v[:, sel], axis=1))
    snr = env.max(1) / nrms
    use = (snr > 20) & (nrms < 2 * np.median(nrms))
    # first strong arrival: first envelope local max above 50 % of the trace max
    tp = np.full(len(R), np.nan)
    ts = t[sel]
    for n in np.where(use)[0]:
        e = env[n]; th = 0.5 * e.max()
        i = np.argmax(e > th)
        while i + 1 < len(e) and e[i + 1] >= e[i]:
            i += 1
        tp[n] = ts[i]
    ok = np.isfinite(tp)

    def res(p):
        return p[0] + np.hypot(R[ok, 0] - p[2], R[ok, 1] - p[3]) / p[1] - tp[ok]

    best = None
    for gx, gy in [guess] + [(guess[0] + dx, guess[1] + dy) for dx in (-40, 40) for dy in (-40, 40)]:
        r = least_squares(res, [508e-6, 1336e3, gx, gy], x_scale=[1e-5, 1e5, 10, 10], loss="soft_l1", f_scale=3e-6)
        if best is None or r.cost < best.cost:
            best = r
    rr = res(best.x)
    return dict(x=float(best.x[2]), y=float(best.x[3]), t0_us=float(best.x[0] * 1e6), v=float(best.x[1] / 1e3),
                resid_us=float(np.median(np.abs(rr)) * 1e6), n=int(ok.sum()))


def main():
    out = {}
    for scan, fmt in (("pristine", "pristine_mat/excitation_{}_pristine.mat"), ("scatter", "scatt_mat/excitation_{}_scatt.mat")):
        F = {k: load(DATA / fmt.format(k)) for k in (1, 2, 3, 4)}
        R, t = F[1][0], F[1][2]
        O = 0.5 * (R.min(0) + R.max(0))
        out[scan] = {}
        for k in (1, 2, 3, 4):
            v = F[1][1] if k == 1 else F[k][1] - F[1][1]
            # start from the planned position and from its mirror in u (labels may be swapped)
            cands = [fit_source(R, v, t, tuple(O + SRC_UV[k - 1] * m)) for m in (np.array([1, 1]), np.array([-1, 1]))]
            c = min(cands, key=lambda d: d["resid_us"])
            out[scan][f"file{k}"] = c
            print(f"{scan:8s} file {k}: source at ({c['x']:7.1f}, {c['y']:6.1f}) mm, t0 {c['t0_us']:.1f} us, "
                  f"v {c['v']:.0f} m/s, median |resid| {c['resid_us']:.2f} us, {c['n']} points")
    json.dump(out, open(OUT / "sources_scans.json", "w"), indent=1)


if __name__ == "__main__":
    main()
