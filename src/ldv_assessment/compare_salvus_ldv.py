"""Score the Salvus 3-D LDV models (no cube / steel cube positions) against the LDV data.

Same pipeline as fit_scatterer_ldv.py: 8-70 kHz band-pass, window t < 650 us, one
deconvolution filter (wavelet + trigger delay) per model estimated on training points
outside the near-source zone and the bad scan row, misfit on held-out points before and
behind the cube, plus the on-axis amplitude ratio (observed / model RMS).

Usage:  python src/ldv_assessment/compare_salvus_ldv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_mindlin_ldv import OUT, load_ldv  # noqa: E402


def main():
    P, t, obs, _, sos = load_ldv(650e-6)
    sal = np.load(OUT / "salvus_ldv_cube.npz")
    meta = json.loads((OUT / "salvus_ldv_cube.json").read_text())
    idx = sal["rec_idx"]
    P, obs = P[idx], obs[:, idx]
    src = np.asarray(meta["source"])
    r = np.hypot(*(P - src).T)
    usable = (r > 0.04) & (np.abs(P[:, 1] - src[1]) > 0.004)
    train = (np.random.default_rng(0).random(len(P)) < 0.5) & usable
    behind = ~train & usable & (P[:, 0] > -0.035)
    before = ~train & usable & (P[:, 0] <= -0.035)
    axis = usable & (np.degrees(np.abs(np.arctan2(P[:, 1] - src[1], P[:, 0] - src[0]))) < 15)
    nf = 2 * len(t)
    bins = np.arange(0.04, 0.26, 0.02)
    results, preds = {}, {}
    names = [k for k in sal.files if k != "rec_idx" and not k.startswith("t__")]
    for name in names:
        ts, v = sal[f"t__{name}"], sal[name]
        syn = np.stack([np.interp(t, ts, v[:, k], left=0.0) for k in range(len(idx))], 1)
        syn = sosfiltfilt(sos, syn, axis=0)
        S, O = np.fft.rfft(syn, nf, axis=0), np.fft.rfft(obs, nf, axis=0)
        num, den = np.sum(O[:, train] * np.conj(S[:, train]), 1), np.sum(np.abs(S[:, train]) ** 2, 1)
        pred = np.fft.irfft((num / (den + 1e-3 * den.max()))[:, None] * S, nf, axis=0)[: len(t)]
        mis = lambda sel: float(np.linalg.norm(pred[:, sel] - obs[:, sel]) / np.linalg.norm(obs[:, sel]))
        ratio = np.sqrt(np.mean(obs**2, 0)) / np.maximum(np.sqrt(np.mean(pred**2, 0)), 1e-12)
        prof = [float(np.median(ratio[axis & (r >= a) & (r < a + 0.02)])) for a in bins]
        results[name] = dict(behind=mis(behind), before=mis(before), axis_ratio=prof)
        preds[name] = pred.astype(np.float32)
        print(f"{name:14s} misfit behind cube {results[name]['behind']:.3f}  before {results[name]['before']:.3f} | "
              "on-axis obs/model RMS: " + " ".join(f"{v:.2f}" for v in prof), flush=True)
    print("distance bins (cm):", " ".join(f"{100 * (a + 0.01):.0f}" for a in bins))
    np.savez_compressed(OUT / "salvus_ldv_compare.npz", P=P, t=t, obs=obs, src=src, train=train, **preds)
    (OUT / "salvus_ldv_compare.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
