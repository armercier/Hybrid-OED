"""Locate the PZT sources of a (possibly multi-source) LDV scan from arrival times.

The wavefield vz is band-passed, split by propagation direction in the (kx, ky, f) domain into
one field per source (sectors given on the command line, default: +x and +y travel = one source
on the left and one below), the envelope peak time is picked per point, and a point source
t = t0 + |r - s| / v_g is fitted (robust least squares) for each field.

Usage:  python src/ldv_pmma/locate_sources.py [scan.mat]
Output: ldv_pmma_results/sources.json, sources.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, hilbert, sosfiltfilt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess_coarse_scan import F_BAND, OUT, REPO, T_NOISE, load, to_grid  # noqa: E402

DEFAULT = REPO / "ldv_data_comprehensive" / "Soruce_1_40kHz_1-5V_20avg_320nt_3125kHz.mat"


def dropouts(comps, noise_i):
    bad = np.zeros(len(comps["z"]), bool)
    for v in comps.values():
        n = np.sqrt(np.mean(v[:, noise_i] ** 2, 1))
        bad |= n > 3 * np.median(n)
    return bad


def split_by_direction(G, dx, dt, sectors):
    """G (ny, nx, nt) -> list of fields keeping waves whose propagation azimuth is in each sector."""
    ny, nx, nt = G.shape
    S = np.fft.fftn(G)
    ky = np.fft.fftfreq(ny, dx)[:, None, None]
    kx = np.fft.fftfreq(nx, dx)[None, :, None]
    f = np.fft.fftfreq(nt, dt)[None, None, :]
    # a wave exp(i(k.r - w t)) travels along +k for f > 0; with numpy's fft sign convention the
    # propagation direction of a component at (kx, ky, f) is -sign(f) * (kx, ky)
    px, py = -np.sign(f) * kx, -np.sign(f) * ky
    az = np.degrees(np.arctan2(py, px))
    out = []
    for lo, hi in sectors:
        m = ((az - lo) % 360) <= ((hi - lo) % 360)
        out.append(np.real(np.fft.ifftn(S * m)))
    return out


def main(path=DEFAULT, sectors=((-60, 60), (30, 150))):
    t, xs, ys, ix, iy, comps = load(path)
    fs = 1 / (t[1] - t[0])
    dx = float(np.median(np.diff(xs)))
    noise_i = t < T_NOISE
    bad = dropouts(comps, noise_i)
    sos = butter(4, F_BAND, btype="band", fs=fs, output="sos")
    sel = (t > 450e-6) & (t < 1100e-6)
    ts = t[sel]
    vz = sosfiltfilt(sos, comps["z"], axis=1)[:, sel]
    G = to_grid(np.where(bad[:, None], 0.0, vz), ix, iy, len(xs), len(ys))
    G = np.nan_to_num(G)
    fields = split_by_direction(G, dx, ts[1] - ts[0], sectors)
    X, Y = np.meshgrid(xs, ys)
    good = to_grid((~bad).astype(float), ix, iy, len(xs), len(ys)) > 0.5
    res = {}
    fig, axes = plt.subplots(1, len(fields), figsize=(7 * len(fields), 5), layout="constrained")
    for n, (Fd, sec) in enumerate(zip(fields, sectors)):
        env = np.abs(hilbert(Fd, axis=2))
        tp = ts[env.argmax(2)]
        amp = env.max(2)
        use = good & (amp > 0.3 * np.percentile(amp[good], 90))
        x, y, tt = X[use], Y[use], tp[use]
        # initial guess: the source lies against the propagation direction, ~200 mm upstream
        a = np.radians(np.mean(sec))
        x0 = [X.mean() - 0.2 * np.cos(a), Y.mean() - 0.2 * np.sin(a), tt.min() - 0.1e-3, 1200.0]
        fun = lambda p: (p[2] + np.hypot(x - p[0], y - p[1]) / p[3] - tt) * 1e6
        sol = least_squares(fun, x0, loss="soft_l1", f_scale=5.0)
        sx, sy, t0, vg = sol.x
        r = fun(sol.x)
        res[f"source_{n + 1}"] = dict(sector_deg=list(sec), x_mm=sx * 1e3, y_mm=sy * 1e3, t0_us=t0 * 1e6,
                                      group_velocity=vg, residual_rms_us=float(np.sqrt(np.mean(r**2))),
                                      n_points=int(use.sum()))
        ax = axes[n]
        im = ax.imshow(np.where(use, tp * 1e6, np.nan), origin="lower", cmap="viridis",
                       extent=(xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3))
        fig.colorbar(im, ax=ax).set_label("envelope peak time (us)")
        ax.plot(sx * 1e3, sy * 1e3, "r*", ms=14, clip_on=False)
        ax.set_xlim(min(xs[0], sx) * 1e3 - 20, max(xs[-1], sx) * 1e3 + 20)
        ax.set_ylim(min(ys[0], sy) * 1e3 - 20, max(ys[-1], sy) * 1e3 + 20)
        ax.set_aspect("equal")
        ax.set_title(f"field {n + 1} (azimuth {sec[0]}..{sec[1]} deg): source ({sx * 1e3:.0f}, {sy * 1e3:.0f}) mm,\n"
                     f"t0 {t0 * 1e6:.0f} us, v_g {vg:.0f} m/s, residual {res[f'source_{n + 1}']['residual_rms_us']:.1f} us",
                     fontsize=9)
    fig.savefig(OUT / "sources.png", dpi=90)
    (OUT / "sources.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
