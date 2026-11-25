"""Quality and mode content of the coarse pristine 3-D LDV scan of the PMMA plate (Nov 2026).

Data: Polytec export, vib_x / vib_y / vib_z (m/s) on a regular 6 mm grid, ref1 = trigger gate.
The wave arrives ~550 us after the trigger; the first 500 us are used as the noise window.
Outputs (ldv_pmma_results/):
  quality.png   noise and band-limited SNR maps per component, dropout points
  snapshots.png band-passed wavefield snapshots (x, y, z)
  fk.png        wavenumber spectra |V(kx, ky, f)| at 30/40/50 kHz per component + dispersion picks
  summary.json  numbers

Usage:  python src/ldv_pmma/assess_coarse_scan.py [path/to/scan.mat]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "ldv_pmma_results"
F_BAND = (15e3, 90e3)
T_NOISE = 500e-6          # noise window: [0, T_NOISE]
T_SIG = (520e-6, 950e-6)  # signal window


def load(path):
    d = sio.loadmat(path)
    t = d["t"][0]
    xyz = d["XYZ"][:, :2]
    xs, ys = np.unique(np.round(xyz[:, 0], 6)), np.unique(np.round(xyz[:, 1], 6))
    ix = np.searchsorted(xs, np.round(xyz[:, 0], 6))
    iy = np.searchsorted(ys, np.round(xyz[:, 1], 6))
    comps = {c: d[f"vib_{c}"] for c in "xyz"}
    return t, xs, ys, ix, iy, comps


def to_grid(v, ix, iy, nx, ny):
    g = np.full((ny, nx) + v.shape[1:], np.nan)
    g[iy, ix] = v
    return g


def main(path):
    OUT.mkdir(exist_ok=True)
    t, xs, ys, ix, iy, comps = load(path)
    fs = 1 / (t[1] - t[0])
    nx, ny = len(xs), len(ys)
    dx = float(np.median(np.diff(xs)))
    ext = (xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3)
    sos = butter(4, F_BAND, btype="band", fs=fs, output="sos")
    noise_i, sig_i = t < T_NOISE, (t >= T_SIG[0]) & (t <= T_SIG[1])
    summary = dict(file=str(path), n_points=len(ix), grid=[nx, ny], dx_mm=dx * 1e3, fs_hz=fs,
                   record_us=float(t[-1] * 1e6), band_hz=F_BAND)
    filt, snr, noise_raw = {}, {}, {}
    for c, v in comps.items():
        noise_raw[c] = np.sqrt(np.mean(v[:, noise_i] ** 2, 1))
        vf = sosfiltfilt(sos, v, axis=1)
        filt[c] = vf
        nrms = np.sqrt(np.mean(vf[:, noise_i] ** 2, 1))
        snr[c] = np.abs(vf[:, sig_i]).max(1) / nrms
    # dropouts: raw noise far above the typical level of that component
    bad = np.zeros(len(ix), bool)
    for c in "xyz":
        med = np.median(noise_raw[c])
        bad |= noise_raw[c] > 3 * med
    summary["dropouts"] = int(bad.sum())
    for c in "xyz":
        s = snr[c][~bad]
        summary[f"snr_{c}"] = dict(median=float(np.median(s)), p10=float(np.percentile(s, 10)),
                                   p90=float(np.percentile(s, 90)), frac_above_10=float(np.mean(s > 10)),
                                   noise_raw_median_mm_s=float(np.median(noise_raw[c]) * 1e3),
                                   peak_band_median_mm_s=float(np.median(np.abs(filt[c][~bad][:, sig_i]).max(1)) * 1e3))

    # ---- quality figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), layout="constrained")
    for j, c in enumerate("xyz"):
        im = axes[0, j].imshow(to_grid(noise_raw[c] * 1e3, ix, iy, nx, ny), origin="lower", extent=ext, cmap="magma")
        fig.colorbar(im, ax=axes[0, j], shrink=0.8).set_label("mm/s")
        axes[0, j].set_title(f"vib_{c}: raw noise rms (0-{T_NOISE * 1e6:.0f} us, full band)", fontsize=9)
        g = to_grid(np.where(bad, np.nan, snr[c]), ix, iy, nx, ny)
        im = axes[1, j].imshow(g, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=np.nanpercentile(g, 98))
        fig.colorbar(im, ax=axes[1, j], shrink=0.8).set_label("peak / noise rms")
        axes[1, j].set_title(f"vib_{c}: SNR after {F_BAND[0] / 1e3:.0f}-{F_BAND[1] / 1e3:.0f} kHz band-pass "
                             f"(median {summary[f'snr_{c}']['median']:.0f})", fontsize=9)
        by, bx = iy[bad], ix[bad]
        axes[1, j].plot(xs[bx] * 1e3, ys[by] * 1e3, "x", color="r", ms=4)
    for a in axes.flat:
        a.set_aspect("equal"), a.tick_params(labelsize=7)
    fig.suptitle(f"Coarse pristine scan: {Path(path).name}; red x: dropouts ({bad.sum()} points, raw noise > 3x median)")
    fig.savefig(OUT / "quality.png", dpi=90)

    # ---- snapshots
    env = np.sqrt(np.nanmean(np.where(bad[:, None], np.nan, filt["z"]) ** 2, 0))
    k0 = int(np.argmax(env))
    times = [k0 - int(60e-6 * fs), k0, k0 + int(60e-6 * fs), k0 + int(140e-6 * fs)]
    fig, axes = plt.subplots(3, 4, figsize=(17, 8.5), layout="constrained")
    for j, c in enumerate("xyz"):
        G = to_grid(np.where(bad[:, None], np.nan, filt[c]), ix, iy, nx, ny)
        vmax = np.nanpercentile(np.abs(G[..., times]), 99)
        for i, k in enumerate(times):
            axes[j, i].imshow(G[..., k], origin="lower", extent=ext, cmap="seismic", vmin=-vmax, vmax=vmax)
            axes[j, i].set_title(f"vib_{c}, t = {t[k] * 1e6:.0f} us", fontsize=9)
            axes[j, i].set_aspect("equal"), axes[j, i].tick_params(labelsize=6)
    fig.suptitle("Band-passed wavefield snapshots (same colour scale per row)")
    fig.savefig(OUT / "snapshots.png", dpi=90)

    # ---- f-k: 3-D FFT over (y, x, t) in the signal window, dropouts set to 0
    win = np.hanning(sig_i.sum())
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dx))
    f = np.fft.rfftfreq(sig_i.sum(), 1 / fs)
    taper = np.outer(np.hanning(ny), np.hanning(nx))[..., None]
    freqs = (30e3, 40e3, 50e3)
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), layout="constrained")
    picks = {}
    KX, KY = np.meshgrid(kx, ky)
    KR = np.hypot(KX, KY)
    kbins = np.linspace(0, min(kx.max(), ky.max()), 40)
    for j, c in enumerate("xyz"):
        G = to_grid(np.where(bad[:, None], 0.0, filt[c][:, sig_i]), ix, iy, nx, ny) * win * taper
        S = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.rfft(G, axis=2), axes=(0, 1)), axes=(0, 1)))
        picks[c] = {}
        for i, fr in enumerate(freqs):
            kf = int(np.argmin(np.abs(f - fr)))
            P = S[..., kf] ** 2
            axes[j, i].imshow(P, origin="lower", extent=(kx[0], kx[-1], ky[0], ky[-1]), cmap="magma")
            axes[j, i].set_title(f"vib_{c}: |V(kx, ky)| at {fr / 1e3:.0f} kHz", fontsize=9)
            axes[j, i].set_xlabel("kx (1/m)", fontsize=7), axes[j, i].tick_params(labelsize=6)
            # radial average -> dominant wavenumber
            prof = np.array([P[(KR >= a) & (KR < b)].mean() if ((KR >= a) & (KR < b)).any() else 0
                             for a, b in zip(kbins[:-1], kbins[1:])])
            kc = 0.5 * (kbins[:-1] + kbins[1:])
            kpk = float(kc[np.argmax(prof[1:]) + 1])
            picks[c][f"{fr / 1e3:.0f}kHz"] = dict(k_cyc_per_m=kpk, wavelength_mm=1e3 / kpk, phase_velocity=fr / kpk)
            for lam, lab, col in ((25e-3, "A0 25 mm", "c"), (58e-3, "S0 58 mm", "lime"), (34e-3, "SH0 34 mm", "w")):
                kk = (1 / lam) * fr / 40e3 if lab.startswith(("S0", "SH0")) else (1 / lam) * np.sqrt(fr / 40e3)
                axes[j, i].add_patch(plt.Circle((0, 0), kk, fill=False, ec=col, lw=0.6, ls="--"))
            axes[j, 3].plot(kc, prof / prof.max(), label=f"{fr / 1e3:.0f} kHz")
        axes[j, 3].set_xlabel("|k| (1/m)"), axes[j, 3].legend(fontsize=7)
        axes[j, 3].set_title(f"vib_{c}: radial wavenumber spectrum", fontsize=9)
        for kk, lab in ((40, "A0"), (17, "S0"), (29, "SH0")):
            axes[j, 3].axvline(kk, color="k", lw=0.5, ls=":")
            axes[j, 3].text(kk, 1.02, lab, fontsize=7, ha="center")
    fig.suptitle("Wavenumber content (dashed circles at 40 kHz-scaled expectations: cyan A0, green S0, white SH0)")
    fig.savefig(OUT / "fk.png", dpi=85)
    summary["dominant_wavenumber"] = picks
    summary["energy_ratio_in_band"] = {c: float(np.mean(filt[c][~bad][:, sig_i] ** 2) /
                                                np.mean(filt["z"][~bad][:, sig_i] ** 2)) for c in "xyz"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         REPO / "ldv_data_comprehensive" / "Soruce_1_40kHz_1-5V_20avg_320nt_3125kHz.mat")
