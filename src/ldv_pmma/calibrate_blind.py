"""Blind background calibration on the defect scan (no knowledge of the defects).

1. "No surface" points: LDV points whose vz RMS (500-800 us) is more than NOSURF_RATIO x the median of their
   neighbours within 12 mm (median over the four files); e.g. the beam inside a through-hole.
   Saved to ldv_work/prepared/scatter_nosurface.npz (scatter.npz is not modified).
2. Robust self-calibration of velocity scale s, Q and the four source positions on the defect scan:
   same model and search as calibrate_pristine.py (PZT blocks as added mass, joint source signatures),
   window 450-800 us, no defect and no cube in the model, start from the pristine values (s, Q) and the
   arrival-time source positions of the defect scan. Robust misfit by iteratively reweighted least squares:
   per-point weight w = min(1, C * median(e) / e), e = residual energy of the point (all files), updated
   at the start of each sweep, so high-residual zones (scatterers) are down-weighted from the data alone.
3. Also writes the "pristine-only" background (pristine s, Q, sources mapped with the registration) for
   comparison: ldv_pmma_results/calib/background_pristine_mapped.json.

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/calibrate_blind.py
Output: ldv_pmma_results/calib/calib_blind.{json,log}, calib_blind_weights.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
from calibrate_pristine import KS, OUT, Calib  # noqa: E402

PREP = REPO / "ldv_work" / "prepared"
T_WIN = (450e-6, 800e-6)
NOSURF_RATIO, NOSURF_RADIUS = 1.5, 12.0
C_ROBUST = 2.0


def no_surface(d):
    R, t = d["R"], d["t"]
    w = (t > 500e-6) & (t < 800e-6)
    bad = np.any([d[f"bad{k}"] for k in KS], 0)
    rms = np.stack([np.sqrt(np.mean(d[f"vz{k}"][:, w] ** 2, 1)) for k in KS])
    q = np.full(len(R), np.nan)
    for i, n in enumerate(cKDTree(R).query_ball_point(R, NOSURF_RADIUS)):
        n = [j for j in n if j != i and not bad[j]]
        if len(n) >= 4:
            q[i] = np.median(rms[:, i] / np.median(rms[:, n], 1))
    return (q > NOSURF_RATIO) & ~bad, q


def pristine_mapped():
    cal = json.loads((OUT / "calib_pristine.json").read_text())
    reg = json.loads((PREP / "geometry.json").read_text())["registration"]
    O = np.array(reg["O"]); th = np.radians(reg["rotation_deg"]); T = np.array(reg["shift_mm"])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = {k: ((np.array(v) - O) @ Rm.T + O + T).round(2).tolist() for k, v in cal["sources_mm"].items()}
    out = dict(velocity_scale=cal["velocity_scale"], Q=cal["Q"], sources_mm=src,
               note="pristine calibration, sources mapped to the defect-scan frame with the registration")
    (OUT / "background_pristine_mapped.json").write_text(json.dumps(out, indent=1))


def main():
    pristine_mapped()
    d = np.load(PREP / "scatter.npz")
    ns, q = no_surface(d)
    np.savez(PREP / "scatter_nosurface.npz", nosurface=ns, ratio=q, rule=f"rms > {NOSURF_RATIO} x median of neighbours "
             f"within {NOSURF_RADIUS} mm (median over files)")
    print(f"no-surface points: {ns.sum()} (rule: amplitude > {NOSURF_RATIO} x local median)", flush=True)
    cal = Calib("scatter", T_WIN)
    base = {k: cal.m[k] * (~ns[cal.use]).astype(np.float32) for k in KS}
    cal.m = {k: base[k].copy() for k in KS}
    pc = json.loads((OUT / "calib_pristine.json").read_text())
    s, Q, src = pc["velocity_scale"], pc["Q"], cal.src0.copy()
    hist = []

    def run(s, Q, src):
        pred, _ = cal.predict(cal.simulate(s, Q, src))
        return pred

    def evaluate(s, Q, src, tag):
        t0 = time.perf_counter()
        J, per = cal.misfit(run(s, Q, src))
        hist.append(dict(tag=tag, s=s, Q=Q, src_mm=(src * 1e3).round(1).tolist(), misfit=J))
        print(f"{tag:6s} s {s:.4f} Q {Q:5.1f} src {np.round(src * 1e3, 1).tolist()} -> {J:.4f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        return J

    def reweight(pred):
        e = sum(base[k] * np.sum((pred[k] - cal.obs[k]) ** 2, 0) for k in KS)
        n = sum(base[k] * np.sum(cal.obs[k] ** 2, 0) for k in KS)
        e = e / np.maximum(n, 1e-30)
        med = np.median(e[sum(base[k] for k in KS) > 0])
        wgt = np.minimum(1.0, C_ROBUST * med / np.maximum(e, 1e-30))
        for k in KS:
            cal.m[k] = (base[k] * wgt).astype(np.float32)
        return wgt

    wgt = reweight(run(s, Q, src))
    for sweep in range(3):
        if sweep > 0:
            wgt = reweight(run(s, Q, src))
        print(f"sweep {sweep}: weights < 0.5 on {(wgt < 0.5).sum()} points, mean weight {wgt.mean():.3f}", flush=True)
        best = evaluate(s, Q, src, "start")
        for cand in ((0.985, 0.99, 1.0, 1.005) if sweep == 0 else (s - 0.0025, s + 0.0025)):
            J = evaluate(cand, Q, src, "speed")
            if J < best:
                best, s = J, cand
        for cand in ((25, 30, 40) if sweep == 0 else (Q * 0.9, Q * 1.1)):
            J = evaluate(s, cand, src, "Q")
            if J < best:
                best, Q = J, cand
        step = (4e-3, 1.5e-3, 0.75e-3)[sweep]
        for k in range(4):
            improved = True
            while improved:
                improved = False
                for dd in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    trial = src.copy(); trial[k] += dd
                    J = evaluate(s, Q, trial, f"S{k + 1}")
                    if J < best:
                        best, src, improved = J, trial, True
    pred = run(s, Q, src)
    wgt = reweight(pred)
    J_w, _ = cal.misfit(pred)
    cal.m = {k: base[k].copy() for k in KS}
    J_all, per_all = cal.misfit(pred)
    geo = json.loads((PREP / "geometry.json").read_text())
    D3 = geo["scatter"]["defects"]["D3"]
    summ = dict(window_us=[450, 800], no_surface_points=int(ns.sum()), robust=f"IRLS, w = min(1, {C_ROBUST} median(e)/e)",
                velocity_scale=s, Q=Q, sources_mm={f"S{k}": (src[k - 1] * 1e3).round(1).tolist() for k in KS},
                sources_from_D3_mm={f"S{k}": [round(src[k - 1][0] * 1e3 - D3["x"], 1), round(src[k - 1][1] * 1e3 - D3["y"], 1)]
                                    for k in KS},
                misfit_weighted=J_w, misfit_all_points=J_all, per_file_all={f"ex{k}": v for k, v in per_all.items()},
                downweighted_points=int((wgt < 0.5).sum()), history=hist)
    (OUT / "calib_blind.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "history"}, indent=1), flush=True)
    U, V = (cal.Ru[:, 0] - cal.O[0]) * 1e3, (cal.Ru[:, 1] - cal.O[1]) * 1e3
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(U, V, c=wgt, s=6, marker="s", cmap="viridis", vmin=0, vmax=1)
    nsu = ns[cal.use]
    ax.scatter(U[nsu], V[nsu], s=14, marker="x", c="r", label="no-surface points (excluded)")
    for n, dd in geo["scatter"]["defects"].items():
        r = dd.get("diameter_mm", dd.get("size_mm")) / 2
        ax.add_patch(plt.Circle((dd["x"] - cal.O[0] * 1e3, dd["y"] - cal.O[1] * 1e3), r + 2, fill=False, ec="w", lw=1))
        ax.text(dd["x"] - cal.O[0] * 1e3 + 7, dd["y"] - cal.O[1] * 1e3 + 7, n, color="w", fontsize=9, fontweight="bold")
    ax.set_aspect("equal"), ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Blind self-calibration: robust weights (found from the data; outlines only for reading)\n"
                 f"s {s:.4f}, Q {Q:.1f}, misfit {J_all:.3f} (all points)", fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    fig.tight_layout()
    fig.savefig(OUT / "calib_blind_weights.png", dpi=90)


if __name__ == "__main__":
    main()
