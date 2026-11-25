"""Waveform check for the paper: LDV data vs Mindlin plate vs Salvus 3-D, on the pristine and the defect scan.

Every model gives one Green's-function set per source (Ricker 45 kHz at 30 us, the Mindlin convention; Salvus traces,
Ricker centred at t = 0, are read at t - 30 us), and goes through the same joint least-squares source-signature fit of
its scan (file 1 = S1, file k = S1 + Sk, 15-90 kHz), so the misfits are directly comparable.
  pristine scan (Calib, window 450-800 us): Mindlin calibrated on it (calib_pristine.json), Salvus 'pristine'
  defect scan (ThicknessInversion, blind background calib_blind.json, no-surface points excluded):
      Mindlin pristine plate, Mindlin as built (holes as thinning on the 3 mm grid, D3 at h_min, cube as rigid body),
      Mindlin blind full-scan inversion (v3), Salvus as built ('asbuilt')
Misfit = sum of squared residuals / sum of squared data over all files and points (as in the inversion).
GPU (JAX); run after salvus_asbuilt.py.

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/waveform_check.py
Output: ldv_pmma_results/waveform_check/{waveform_check.npz, misfits.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
import jax.numpy as jnp  # noqa: E402

from calibrate_pristine import KS, Calib  # noqa: E402
from invert_thickness import ThicknessInversion  # noqa: E402

OUT = REPO / "ldv_pmma_results" / "waveform_check"
SAL = REPO / "ldv_pmma_results" / "salvus_asbuilt"
T0 = 30e-6


def salvus_syns(cfg, t_rel, Ru_mm):
    d = np.load(SAL / f"traces_{cfg}.npz")
    R = d["R_scanner"]
    idx = {tuple(np.round(p, 3)): i for i, p in enumerate(R)}
    j = np.array([idx.get(tuple(np.round(p, 3)), -1) for p in Ru_mm])
    out = []
    for k in KS:
        v = d[f"vz_S{k}"]
        tr = np.stack([np.interp(t_rel - T0, d["t"], v[:, i]) if i >= 0 else np.zeros(len(t_rel)) for i in j], 1)
        out.append(tr)
    return out, j >= 0


def misfit(pred, obs, m):
    r = sum(float(np.sum(m[k] * (pred[k] - obs[k]) ** 2)) for k in KS)
    o = sum(float(np.sum(m[k] * obs[k] ** 2)) for k in KS)
    per = {f"ex{k}": float(np.sum(m[k] * (pred[k] - obs[k]) ** 2) / np.sum(m[k] * obs[k] ** 2)) for k in KS}
    return r / o, per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, mis = {}, {}

    # pristine scan
    c = Calib("pristine", (450e-6, 800e-6))
    cal = json.loads((REPO / "ldv_pmma_results" / "calib" / "calib_pristine.json").read_text())
    src = np.array([cal["sources_mm"][f"S{k}"] for k in KS]) * 1e-3
    G = {"mindlin": c.simulate(cal["velocity_scale"], cal["Q"], src)}
    G["salvus"], ok = salvus_syns("pristine", c.t, c.Ru * 1e3)
    m = {k: c.m[k] * ok for k in KS}
    for name, g in G.items():
        pred, _ = c.predict(g)
        mis[f"pristine/{name}"] = misfit(pred, c.obs, m)
        for k in KS:
            res[f"pristine_{name}_ex{k}"] = pred[k].astype(np.float32)
    for k in KS:
        res[f"pristine_data_ex{k}"] = c.obs[k].astype(np.float32); res[f"pristine_mask_ex{k}"] = m[k]
    res["pristine_R"], res["pristine_t"] = c.Ru * 1e3, c.t_abs
    print("pristine", {k: round(v[0], 4) for k, v in mis.items()}, flush=True)

    # defect scan
    bg = REPO / "ldv_pmma_results" / "calib" / "calib_blind.json"
    inv = ThicknessInversion(background=bg, cube=False, nosurface=True)
    ic = c2 = inv.c
    obs = {k: np.asarray(inv.obs[k]) for k in KS}
    sal, ok = salvus_syns("asbuilt", c2.t, c2.Ru * 1e3)
    mm = {k: np.asarray(inv.m[k])[0] * ok for k in KS}
    nm = inv.Xc.shape
    m0 = jnp.zeros((1, *nm), jnp.float32)
    hmin, H = inv.hmin * 1e3, 10.0
    hb = np.full(nm, H)
    X, Y = inv.Xc * 1e3, inv.Yc * 1e3
    for n, d in inv.geo["scatter"]["defects"].items():
        if n.startswith("D"):
            r = np.hypot(X - d["x"], Y - d["y"]) <= d["diameter_mm"] / 2 + 1.5      # control cells touching the hole
            hb[r] = max(H - d["depth_mm"], hmin + 1e-3)
    m_built = jnp.asarray(np.log((hb - hmin) / (H - hmin)), jnp.float32)[None]
    v3 = jnp.asarray(np.load(REPO / "ldv_pmma_results" / "inversion" / "v3" / "result.npz")["m"], jnp.float32)
    inv_cube = ThicknessInversion(background=bg, cube=True, nosurface=True)
    preds = {"mindlin_pristine": inv.predict(inv.sims(m0)), "mindlin_asbuilt": inv_cube.predict(inv_cube.sims(m_built)),
             "mindlin_v3": inv.predict(inv.sims(v3)), "salvus": inv.predict([jnp.asarray(s, jnp.float32) for s in sal])}     # predict applies the taper
    for name, p in preds.items():
        p = {k: np.asarray(p[k]) for k in KS}
        mis[f"defect/{name}"] = misfit(p, obs, mm)
        for k in KS:
            res[f"defect_{name}_ex{k}"] = p[k].astype(np.float32)
    for k in KS:
        res[f"defect_data_ex{k}"] = obs[k].astype(np.float32); res[f"defect_mask_ex{k}"] = mm[k]
    res["defect_R"], res["defect_t"] = ic.Ru * 1e3, ic.t_abs
    print("defect", {k: round(v[0], 4) for k, v in mis.items() if k.startswith("defect")}, flush=True)
    np.savez_compressed(OUT / "waveform_check.npz", **res)
    (OUT / "misfits.json").write_text(json.dumps({k: dict(total=v[0], per_file=v[1]) for k, v in mis.items()}, indent=1))


if __name__ == "__main__":
    main()
