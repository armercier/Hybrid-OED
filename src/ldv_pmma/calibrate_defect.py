"""Step A: recalibrate the background model directly on the defect scan.

Same model and search as calibrate_pristine.py (velocity scale s, Q, four source positions; PZT blocks as
added mass; joint source signatures), but
  - data: defect scan, window 450-800 us (edge echoes after ~800 us), points within EXCLUDE mm of any
    defect (D1-D4, M1) removed, so the background cannot absorb the defects;
  - the steel cube M1 is in the model as the rigid bonded body calibrated in October 2026
    (reports/2026-10_plate_models: 6 mm effective footprint, 7.85 g, full rotary inertia M(a^2/12 + d^2),
    d = H/2 + a/2), at its planned position;
  - start: source positions from the arrival-time fit in the defect-scan frame (geometry.json).

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/calibrate_defect.py
Output: ldv_pmma_results/calib/calib_defect.json, calib_defect_residual.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from calibrate_pristine import DX, KS, OUT, SPONGE, Calib  # noqa: E402
from mindlin import mindlin_plate  # noqa: E402

T_WIN = (450e-6, 800e-6)
EXCLUDE = 25e-3
CUBE_FOOTPRINT = 6e-3


def cube_rigid(c, geo):
    d = geo["scatter"]["defects"]["M1"]
    a = d["size_mm"] * 1e-3
    M_c = d["mass_g"] * 1e-3
    J_c = M_c * (a ** 2 / 12 + (c.H / 2 + a / 2) ** 2)
    mask = (np.abs(c.X - d["x"] * 1e-3) <= CUBE_FOOTPRINT / 2) & (np.abs(c.Y - d["y"] * 1e-3) <= CUBE_FOOTPRINT / 2)
    return mask, M_c, J_c


class CalibDefect(Calib):
    def __init__(self, exclude=EXCLUDE, cube=True):
        super().__init__("scatter", T_WIN)
        geo = json.loads((REPO / "ldv_work" / "prepared" / "geometry.json").read_text())
        self.geo = geo
        near = np.zeros(len(self.Ru), bool)
        for d in geo["scatter"]["defects"].values():
            near |= np.hypot(self.Ru[:, 0] - d["x"] * 1e-3, self.Ru[:, 1] - d["y"] * 1e-3) < exclude
        self.near = near
        for k in KS:
            self.m[k] = self.m[k] * (~near).astype(np.float32)
        rigid = dict(rigid_mask=None)
        if cube:
            mask, M_c, J_c = cube_rigid(self, geo)
            rigid = dict(rigid_mask=mask, rigid_added_mass=M_c, rigid_added_inertia=J_c)
        self.rigid = rigid

        def sim(vp, vs, h, mass, damping, kappa, src_i):
            one = jnp.ones((self.nx, self.ny), jnp.float32)
            tr = mindlin_plate(vp * one, vs * one, self.rho * one, h, src_i, self.stf, DX, DX, self.dt, self.rec_i,
                               sponge_cells=SPONGE, kappa=kappa, mass_scale=mass, damping=damping, **rigid)[..., 0]
            return (1 - self.w) * tr[self.i0] + self.w * tr[self.i0 + 1]

        self._sim = jax.jit(sim)
        print(f"excluded {near.sum()} points within {exclude * 1e3:.0f} mm of a defect; cube "
              f"{'rigid ' + str(int(CUBE_FOOTPRINT * 1e3)) + ' mm' if cube else 'not modelled'}", flush=True)


def main():
    cal = CalibDefect()
    hist = []

    def evaluate(s, Q, src, tag):
        t0 = time.perf_counter()
        pred, _ = cal.predict(cal.simulate(s, Q, src))
        J, per = cal.misfit(pred)
        hist.append(dict(tag=tag, s=s, Q=Q, src_mm=(src * 1e3).round(1).tolist(), misfit=J, per_file=per))
        print(f"{tag:6s} s {s:.4f} Q {Q:5.1f} src {np.round(src * 1e3, 1).tolist()} -> {J:.4f} "
              f"[{' '.join(f'{v:.3f}' for v in per.values())}] ({time.perf_counter() - t0:.0f} s)", flush=True)
        return J

    pc = json.loads((OUT / "calib_pristine.json").read_text())
    s, Q, src = pc["velocity_scale"], pc["Q"], cal.src0.copy()
    best = evaluate(s, Q, src, "start")
    for sweep in range(3):
        steps_s = (0.985, 0.99, 0.995, 1.0, 1.005, 1.01) if sweep == 0 else (s - 0.0025, s + 0.0025)
        for cand in steps_s:
            if abs(cand - s) < 1e-9:
                continue
            J = evaluate(cand, Q, src, "speed")
            if J < best:
                best, s = J, cand
        for cand in ((25, 30, 40, 50) if sweep == 0 else (Q * 0.9, Q * 1.1)):
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
    pred, F = cal.predict(cal.simulate(s, Q, src))
    J, per = cal.misfit(pred)
    # misfit on all points (defect neighbourhoods included) with this background
    m_excl = {k: cal.m[k].copy() for k in KS}
    d = np.load(REPO / "ldv_work" / "prepared" / "scatter.npz")
    for k in KS:
        cal.m[k] = (~d[f"bad{k}"][cal.use]).astype(np.float32)
    J_all, per_all = cal.misfit(pred)
    cal.m = m_excl
    D3 = cal.geo["scatter"]["defects"]["D3"]
    rel = {f"S{k}": [round(src[k - 1][0] * 1e3 - D3["x"], 1), round(src[k - 1][1] * 1e3 - D3["y"], 1)] for k in KS}
    summ = dict(window_us=[T_WIN[0] * 1e6, T_WIN[1] * 1e6], exclude_mm=EXCLUDE * 1e3, cube=f"rigid {CUBE_FOOTPRINT * 1e3:.0f} mm",
                velocity_scale=s, Q=Q, sources_mm={f"S{k}": (src[k - 1] * 1e3).round(1).tolist() for k in KS},
                sources_from_D3_mm=rel, ruler_from_D3_mm=dict(S1=[0, -160], S2=[223, 24], S3=[0, 180], S4=[-216, 0]),
                misfit_background_points=J, per_file={f"ex{k}": v for k, v in per.items()},
                misfit_all_points=J_all, per_file_all={f"ex{k}": v for k, v in per_all.items()},
                noise_floor={f"ex{k}": v for k, v in cal.noise_floor.items()}, history=hist)
    (OUT / "calib_defect.json").write_text(json.dumps(summ, indent=1))
    np.savez(OUT / "calib_defect_filters.npz", F=F, s=s, Q=Q, src=src)
    print(json.dumps({k: v for k, v in summ.items() if k != "history"}, indent=1), flush=True)
    # residual energy map (all points)
    U, V = (cal.Ru[:, 0] - cal.O[0]) * 1e3, (cal.Ru[:, 1] - cal.O[1]) * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    for k, axx in zip(KS, ax.ravel()):
        ok = ~d[f"bad{k}"][cal.use]
        r = np.sqrt(np.sum((cal.obs[k] - pred[k]) ** 2, 0) / np.mean(np.sum(cal.obs[k][:, ok] ** 2, 0)))
        sc = axx.scatter(U[ok], V[ok], c=r[ok], s=5, marker="s", cmap="magma", vmin=0, vmax=np.percentile(r[ok], 99))
        for n, dd in cal.geo["scatter"]["defects"].items():
            rr = dd.get("diameter_mm", dd.get("size_mm")) / 2
            axx.add_patch(plt.Circle((dd["x"] - cal.O[0] * 1e3, dd["y"] - cal.O[1] * 1e3), rr + 2, fill=False, ec="c", lw=1.2))
            axx.add_patch(plt.Circle((dd["x"] - cal.O[0] * 1e3, dd["y"] - cal.O[1] * 1e3), EXCLUDE * 1e3, fill=False,
                                     ec="w", lw=0.6, ls=":"))
            axx.text(dd["x"] - cal.O[0] * 1e3 + 7, dd["y"] - cal.O[1] * 1e3 + 7, n, color="c", fontsize=9, fontweight="bold")
        axx.set_aspect("equal")
        axx.set_title(f"ex{k}: |measured - background model| 450-800 us (misfit all points {per_all[k]:.3f})", fontsize=9)
        plt.colorbar(sc, ax=axx, shrink=0.8)
    fig.suptitle(f"Background recalibrated on the defect scan (dotted: excluded {EXCLUDE * 1e3:.0f} mm zones): s {s:.4f}, "
                 f"Q {Q:.1f}, misfit {J:.3f} (background points), {J_all:.3f} (all points)")
    fig.tight_layout()
    fig.savefig(OUT / "calib_defect_residual.png", dpi=85)


if __name__ == "__main__":
    main()
