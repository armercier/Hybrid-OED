"""Step B: thickness-only inversion of the defect scan on the background recalibrated on that scan.

Differences from invert_defect_scan.py (v1):
  - background (velocity scale, Q, source positions) from calibrate_defect.py (step A), not mapped from
    the pristine scan through the registration;
  - the steel cube M1 is a known rigid bonded body (6 mm footprint, 7.85 g, full rotary inertia) at its
    planned position, not something to image;
  - a single unknown map, the thickness, with relative steps and a smooth lower limit:
        h = h_min + (H - h_min) * exp(m),   m = 0 -> 10 mm,  m -> -inf -> h_min = 1 mm
    the mass follows the thickness (rho h), as for a real hole.
Same data, window (450-800 us, 30 us taper), source-signature estimation, TV and Adam as v1.

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/invert_thickness.py [--iters 150] [--tag v2]
Output: ldv_pmma_results/inversion/<tag>/{result.npz, settings.json, run.log}; figure with plot_inversion.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax.scipy.ndimage import map_coordinates  # noqa: E402

from calibrate_defect import cube_rigid  # noqa: E402
from calibrate_pristine import DX, KS, SPONGE  # noqa: E402
from invert_defect_scan import T_WIN, TAPER, SPACING, Inversion  # noqa: E402
from mindlin import a0_matched_kappa, mindlin_plate  # noqa: E402

CALD = REPO / "ldv_pmma_results" / "calib" / "calib_defect.json"


class ThicknessInversion(Inversion):
    def __init__(self, hmin=1e-3, checkpoint_every=100, background=CALD, cube=True, nosurface=False, subset=None):
        super().__init__(hmin=hmin, checkpoint_every=checkpoint_every)
        c = self.c
        cal = json.loads(Path(background).read_text())
        if nosurface:       # exclude the "no surface" points found blind by calibrate_blind.py
            ns = np.load(REPO / "ldv_work" / "prepared" / "scatter_nosurface.npz")["nosurface"][c.use]
            self.m = {k: self.m[k] * jnp.asarray(~ns, jnp.float32)[None, :] for k in KS}
            self.norm = {k: float(jnp.sum(self.m[k] * self.obs[k] ** 2)) for k in KS}
        if subset is not None:  # reduced scan (make_subsets.py): keep only the subset, minus its own no-surface points
            sb = np.load(subset)
            keep = (sb["keep"] & ~sb["nosurface"])[c.use]
            self.m = {k: self.m[k] * jnp.asarray(keep, jnp.float32)[None, :] for k in KS}
            self.norm = {k: float(jnp.sum(self.m[k] * self.obs[k] ** 2)) for k in KS}
            self.n_points = int(keep.sum())
        self.s, self.Q = cal["velocity_scale"], cal["Q"]
        self.src = np.array([cal["sources_mm"][f"S{k}"] for k in KS]) * 1e-3
        self.kappa = float(a0_matched_kappa(c.H, c.vp * self.s, c.vs * self.s, c.rho, 40e3))
        self.damping = 2 * np.pi * 40e3 / self.Q
        self.src_i = [jnp.asarray((p - c.origin) / DX, jnp.float32) for p in self.src]
        _, self.mass0 = c.fields(self.src, self.s)
        if cube:
            mask, M_c, J_c = cube_rigid(c, self.geo)
            self.rigid = dict(rigid_mask=mask, rigid_added_mass=M_c, rigid_added_inertia=J_c)
        else:
            self.rigid = dict(rigid_mask=None)
        # LDV_FILES selects the scan files used (default all). File 1 = S1, file k = S1 + Sk. Every PZT stays in the model as
        # added mass; unused files have an empty mask and transducers driven in no selected file are not simulated.
        self.active = tuple(int(k) for k in os.environ.get("LDV_FILES", ",".join(map(str, KS))).split(","))
        self.active_src = sorted({0} | {k - 1 for k in self.active if k > 1})
        for k in KS:
            if k not in self.active:
                self.m[k] = jnp.zeros_like(self.m[k])

    def misfits(self, m):
        pred = self.predict(self.sims(m))
        return jnp.stack([jnp.sum(self.m[k] * (pred[k] - self.obs[k]) ** 2) / self.norm[k] for k in self.active])

    def fields(self, m):
        up = map_coordinates(m[0], self.coords, order=1, mode="constant", cval=0.0)
        return self.hmin + (self.H - self.hmin) * jnp.exp(up), self.mass0

    def sims(self, m):
        c = self.c
        h, mass = self.fields(m)
        one = jnp.ones((c.nx, c.ny), jnp.float32)
        out = []
        for i, si in enumerate(self.src_i):
            if i not in self.active_src:
                out.append(jnp.zeros_like(out[0]))
                continue
            tr = mindlin_plate(c.vp * self.s * one, c.vs * self.s * one, c.rho * one, h, si, c.stf, DX, DX, c.dt,
                               c.rec_i, sponge_cells=SPONGE, kappa=self.kappa, mass_scale=mass,
                               damping=self.damping, checkpoint_every=self.ckpt, **self.rigid)[..., 0]
            out.append((1 - c.w) * tr[c.i0] + c.w * tr[c.i0 + 1])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--background", default=str(CALD), help="json with velocity_scale, Q, sources_mm (defect-scan frame)")
    ap.add_argument("--no-cube", action="store_true", help="cube not in the model (unknown target)")
    ap.add_argument("--nosurface", action="store_true", help="exclude the blind 'no surface' points")
    ap.add_argument("--subset", default=None, help="subset npz from make_subsets.py (reduced scan); "
                    "uses the subset's own no-surface points, so do not combine with --nosurface")
    a = ap.parse_args()
    out = REPO / "ldv_pmma_results" / "inversion" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    inv = ThicknessInversion(hmin=a.hmin * 1e-3, background=a.background, cube=not a.no_cube, nosurface=a.nosurface,
                             subset=a.subset)
    vg = jax.jit(jax.value_and_grad(inv.loss, has_aux=True))
    m = jnp.zeros((1, *inv.nc), jnp.float32)
    t0 = time.perf_counter()
    (L, (per, tv)), g = vg(m, a.tv)
    print(f"start (background {Path(a.background).name}, {'no cube' if a.no_cube else 'rigid cube'}, "
          f"{'no-surface points excluded, ' if a.nosurface else ''}no holes): misfit per file {np.round(np.asarray(per), 4)}, "
          f"mean {float(jnp.mean(per)):.4f} ({time.perf_counter() - t0:.0f} s incl. compile)", flush=True)
    opt = optax.adam(a.lr)
    state = opt.init(m)
    hist, m_prev = [], m
    for it in range(a.iters):
        t0 = time.perf_counter()
        (L, (per, tv)), g = vg(m, a.tv)
        J = float(jnp.mean(per))
        if not np.isfinite(J):
            print(f"it {it} misfit not finite: stop, keep previous model", flush=True)
            m = m_prev
            break
        hist.append((J, float(tv), np.asarray(per).tolist()))
        m_prev = m
        upd, state = opt.update(g, state)
        m = optax.apply_updates(m, upd)
        h, _ = inv.fields(m)
        print(f"it {it:3d} misfit {J:.4f} [{' '.join(f'{v:.3f}' for v in np.asarray(per))}] tv {float(tv):.0f} "
              f"min h {float(h.min()) * 1e3:.2f} mm ({time.perf_counter() - t0:.1f} s)", flush=True)
        if it % 10 == 9:
            np.savez(out / "result.npz", m=np.asarray(m), Xc=inv.Xc, Yc=inv.Yc, iters_done=it + 1,
                     hist=np.array([x[0] for x in hist]))
    np.savez(out / "result.npz", m=np.asarray(m), Xc=inv.Xc, Yc=inv.Yc, iters_done=len(hist),
             hist=np.array([x[0] for x in hist]))
    (out / "settings.json").write_text(json.dumps(dict(
        vars(a), param="log_thickness", hmax_mm=None, window_us=[T_WIN[0] * 1e6, T_WIN[1] * 1e6], taper_us=TAPER * 1e6,
        spacing_mm=SPACING * 1e3, background_file=a.background,
        cube_model="none (unknown target)" if a.no_cube else "rigid 6 mm, 7.85 g", s=inv.s, Q=inv.Q,
        sources_m=inv.src.tolist(), active_files=list(inv.active), n_points=getattr(inv, "n_points", None), misfit_start=hist[0][0], misfit_end=hist[-1][0], per_file_end=hist[-1][2]), indent=1))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
