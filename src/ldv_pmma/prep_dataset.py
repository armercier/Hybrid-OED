"""Step 1: inversion dataset from the PMMA 3-D LDV scans (working copy in ldv_work/, never the originals).

For each scan (pristine 6 mm grid, defect 3 mm grid) and file k = 1..4 (file 1 = S1, file k = S1 + Sk,
S1 is never switched off):
  - vz band-passed 15-90 kHz (zero phase), resampled to 625 kHz, window 400-1500 us after the trigger
  - noise RMS (0-480 us), dropouts (noise > 3 x median)
  - pristine only: wrong-source points (file k identical to file k-1 within noise)
Saves ldv_work/prepared/{pristine,scatter}.npz with R (scanner frame, mm), t (s), vz[k] (nrec, nt, float32,
m/s), noise[k], bad[k] (dropout or wrong source), and the grid step.

Usage:  python src/ldv_pmma/prep_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "ldv_work"
OUT = WORK / "prepared"
F_BAND = (15e3, 90e3)
FS_OUT = 625e3
T_OUT = (400e-6, 1500e-6)
T_NOISE = 480e-6
FILES = {"pristine": ("pristine_mat/excitation_{}_pristine.mat", 6.0), "scatter": ("scatt_mat/excitation_{}_scatt.mat", 3.0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_out = np.arange(T_OUT[0], T_OUT[1], 1 / FS_OUT)
    info = {}
    for scan, (fmt, step) in FILES.items():
        vz, noise, raw = {}, {}, {}
        R = None
        for k in (1, 2, 3, 4):
            d = sio.loadmat(WORK / fmt.format(k), variable_names=["XYZ", "vib_z", "t"])
            Rk = d["XYZ"][:, :2] * 1e3
            if R is None:
                R = Rk
            assert np.allclose(R, Rk, atol=0.05), f"{scan}: files do not share the points"
            t = d["t"][0]
            fs = 1 / (t[1] - t[0])
            v = sosfiltfilt(butter(4, F_BAND, btype="band", fs=fs, output="sos"), d["vib_z"], axis=1)
            noise[k] = np.sqrt(np.mean(v[:, t < T_NOISE] ** 2, 1))
            raw[k] = v[:, t > 500e-6]
            vz[k] = CubicSpline(t, v, axis=1)(t_out).astype(np.float32)
        bad = {k: noise[k] > 3 * np.median(noise[k]) for k in vz}
        wrong = {k: np.zeros(len(R), bool) for k in vz}
        for k in (2, 3, 4):
            r = np.mean((raw[k] - raw[k - 1]) ** 2, 1) / (noise[k] ** 2 + noise[k - 1] ** 2)
            clean = (noise[k] < 2 * np.median(noise[k])) & (noise[k - 1] < 2 * np.median(noise[k - 1]))
            wrong[k] = (r < 3) & clean
        np.savez(OUT / f"{scan}.npz", R=R, t=t_out, step=step,
                 **{f"vz{k}": vz[k] for k in vz}, **{f"noise{k}": noise[k] for k in vz},
                 **{f"dropout{k}": bad[k] for k in vz}, **{f"wrong{k}": wrong[k] for k in vz},
                 **{f"bad{k}": bad[k] | wrong[k] for k in vz})
        info[scan] = dict(points=len(R), fs_hz=FS_OUT, window_us=[T_OUT[0] * 1e6, T_OUT[1] * 1e6],
                          **{f"ex{k}": dict(dropouts=int(bad[k].sum()), wrong_source=int(wrong[k].sum()),
                                            usable=int((~(bad[k] | wrong[k])).sum()),
                                            noise_median_um_s=float(np.median(noise[k]) * 1e6)) for k in vz})
        print(scan, json.dumps(info[scan], indent=1), flush=True)
    (OUT / "prepared_info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
