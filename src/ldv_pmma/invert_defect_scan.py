"""Step 4: full-waveform inversion of the PMMA defect scan (real 3-D LDV data, vz) with the Mindlin model.

Data: defect scan (prep_dataset.py, working copy), all usable points, window 450-800 us with a 30 us
cosine taper at the end. The plate edges are not modelled (sponge); their echoes arrive after ~800 us
(calibration misfit 0.028 / 0.027 in 450-650 / 650-800 us, ~1 after 800 us), so the window removes them.
Model: calibrated pristine model (calibrate_pristine.py: velocity scale s, Q, source positions mapped
to the defect-scan frame with the registration, PZT blocks as added mass) + two unknown maps on a
3 mm grid over the scan area (bilinear onto the 1 mm FD grid, pristine outside):
  thickness  h = h_min + (h_max - h_min) * sigmoid(m0 + b)   (b: m0 = 0 gives the nominal 10 mm)
             h_min = 1 mm: the conservative way to allow holes of unknown depth, through-holes included
  mass scale = exp(m1) on rho h (cube; thinning already removes mass through h)
Files: file 1 = S1, file k = S1 + Sk; one source signature per transducer re-estimated jointly at every
iteration by least squares (variable projection, no gradient through it). Loss: sum over files of the
normalised residual energy + TV on (m0, m1). Adam.

Usage:  python src/ldv_pmma/invert_defect_scan.py [--iters 150] [--lr 0.03] [--tv 1e-4] [--hmin 1.0] [--bench]
Output: ldv_pmma_results/inversion/<tag>/{result.npz, settings.json, maps.png, history.png}
"""
from __future__ import annotations

import argparse
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
import optax  # noqa: E402
from jax.scipy.ndimage import map_coordinates  # noqa: E402

from calibrate_pristine import DX, KS, SPONGE, Calib  # noqa: E402
from mindlin import a0_matched_kappa, mindlin_plate  # noqa: E402

CAL = REPO / "ldv_pmma_results" / "calib" / "calib_pristine.json"
PREP = REPO / "ldv_work" / "prepared"
T_WIN, TAPER = (450e-6, 800e-6), 30e-6
SPACING = 3e-3


class Inversion:
    def __init__(self, hmin=1e-3, hmax=12e-3, checkpoint_every=100):
        cal = json.loads(CAL.read_text())
        geo = json.loads((PREP / "geometry.json").read_text())
        self.c = c = Calib("scatter", T_WIN)
        reg = geo["registration"]
        O = np.array(reg["O"]); th = np.radians(reg["rotation_deg"]); T = np.array(reg["shift_mm"])
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        srcP = np.array([cal["sources_mm"][f"S{k}"] for k in KS])
        self.src = ((srcP - O) @ Rm.T + O + T) * 1e-3
        self.s, self.Q = cal["velocity_scale"], cal["Q"]
        self.geo, self.cal = geo, cal
        self.hmin, self.hmax, self.H = hmin, hmax, c.H
        self.b = float(np.log((c.H - hmin) / (hmax - c.H)))
        # fixed: PZT block masses
        _, mass0 = c.fields(self.src, self.s)
        self.mass0 = mass0
        # time taper (applied to data and prediction)
        w = np.clip((T_WIN[1] - c.t_abs) / TAPER, 0, 1)
        self.taper = jnp.asarray(0.5 - 0.5 * np.cos(np.pi * w), jnp.float32)[:, None]
        self.obs = {k: jnp.asarray(c.obs[k], jnp.float32) * self.taper for k in KS}
        self.m = {k: jnp.asarray(c.m[k], jnp.float32)[None, :] for k in KS}
        self.norm = {k: float(jnp.sum(self.m[k] * self.obs[k] ** 2)) for k in KS}
        # model grid over the scan area
        lo, hi = c.Ru.min(0), c.Ru.max(0)
        nc = np.round((hi - lo) / SPACING).astype(int) + 1
        self.lo, self.nc = lo, nc
        self.Xc, self.Yc = np.meshgrid(lo[0] + np.arange(nc[0]) * SPACING, lo[1] + np.arange(nc[1]) * SPACING, indexing="ij")
        self.coords = [jnp.asarray((c.X - lo[0]) / SPACING, jnp.float32), jnp.asarray((c.Y - lo[1]) / SPACING, jnp.float32)]
        self.kappa = float(a0_matched_kappa(c.H, c.vp * self.s, c.vs * self.s, c.rho, 40e3))
        self.damping = 2 * np.pi * 40e3 / self.Q
        self.src_i = [jnp.asarray((p - c.origin) / DX, jnp.float32) for p in self.src]
        self.H2 = jnp.asarray(c.H2, jnp.float32)
        self.ckpt = checkpoint_every

    def fields(self, m):
        up = lambda f: map_coordinates(f, self.coords, order=1, mode="constant", cval=0.0)
        h = self.hmin + (self.hmax - self.hmin) * jax.nn.sigmoid(up(m[0]) + self.b)
        return h, jnp.exp(up(m[1])) * self.mass0

    def sims(self, m):
        c = self.c
        h, mass = self.fields(m)
        one = jnp.ones((c.nx, c.ny), jnp.float32)
        out = []
        for si in self.src_i:
            tr = mindlin_plate(c.vp * self.s * one, c.vs * self.s * one, c.rho * one, h, si, c.stf, DX, DX, c.dt,
                               c.rec_i, sponge_cells=SPONGE, kappa=self.kappa, mass_scale=mass,
                               damping=self.damping, checkpoint_every=self.ckpt)[..., 0]
            out.append((1 - c.w) * tr[c.i0] + c.w * tr[c.i0 + 1])
        return out

    def predict(self, syns):
        nf = self.c.nf
        G = [jnp.fft.rfft(self.taper * g, nf, axis=0) * self.H2 for g in syns]
        Ob = {k: jnp.fft.rfft(self.obs[k], nf, axis=0) for k in KS}
        nfr = G[0].shape[0]
        AhA = jnp.zeros((nfr, 4, 4), jnp.complex64); Aho = jnp.zeros((nfr, 4), jnp.complex64)
        for k in KS:
            cols = [0] if k == 1 else [0, k - 1]
            for a in cols:
                Aho = Aho.at[:, a].add(jnp.sum(self.m[k] * jnp.conj(G[a]) * Ob[k], 1))
                for b in cols:
                    AhA = AhA.at[:, a, b].add(jnp.sum(self.m[k] * jnp.conj(G[a]) * G[b], 1))
        eps = 1e-4 * jnp.max(jnp.abs(AhA), axis=(1, 2))[:, None, None] * jnp.eye(4) + 1e-30 * jnp.eye(4)
        F = jax.lax.stop_gradient(jnp.linalg.solve(AhA + eps, Aho[..., None])[..., 0])
        pred = {}
        for k in KS:
            P = F[:, 0:1] * G[0] + (0 if k == 1 else F[:, k - 1:k] * G[k - 1])
            pred[k] = jnp.fft.irfft(P, nf, axis=0)[: len(self.c.t)]
        return pred

    def misfits(self, m):
        pred = self.predict(self.sims(m))
        return jnp.stack([jnp.sum(self.m[k] * (pred[k] - self.obs[k]) ** 2) / self.norm[k] for k in KS])

    def loss(self, m, tv_w):
        per = self.misfits(m)
        gx, gy = m[:, 1:, :-1] - m[:, :-1, :-1], m[:, :-1, 1:] - m[:, :-1, :-1]
        tv = jnp.sum(jnp.sqrt(gx ** 2 + gy ** 2 + 1e-6) - 1e-3)
        return jnp.mean(per) + tv_w * tv, (per, tv)

    def truth(self):
        """Thickness and mass scale of the as-built plate on the model grid (planned positions, registered)."""
        X, Y = self.Xc * 1e3, self.Yc * 1e3
        thick, mass = np.full_like(X, self.H * 1e3), np.ones_like(X)
        for n, d in self.geo["scatter"]["defects"].items():
            if n.startswith("D"):
                thick[np.hypot(X - d["x"], Y - d["y"]) <= d["diameter_mm"] / 2] = self.H * 1e3 - d["depth_mm"]
            else:
                a = d["size_mm"] / 2
                mass[(np.abs(X - d["x"]) <= a) & (np.abs(Y - d["y"]) <= a)] = 1 + 7850 * d["size_mm"] / (self.H * 1e3 * self.c.rho)
        return thick, mass


def plot(inv, m, hist, out):
    h, mass = (np.asarray(f) for f in inv.fields(jnp.asarray(m)))
    # sample the FD-grid fields at the model grid nodes
    ix = np.round((inv.Xc - inv.c.origin[0]) / DX).astype(int); iy = np.round((inv.Yc - inv.c.origin[1]) / DX).astype(int)
    hm, mm = h[ix, iy] * 1e3, mass[ix, iy] / np.asarray(inv.mass0)[ix, iy]
    tt, tm = inv.truth()
    O = inv.c.O * 1e3
    ext = (inv.Xc[0, 0] * 1e3 - O[0], inv.Xc[-1, 0] * 1e3 - O[0], inv.Yc[0, 0] * 1e3 - O[1], inv.Yc[0, -1] * 1e3 - O[1])
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    for axx, img, lab, cm, vr in ((ax[0, 0], tt, "as built: thickness (mm)", "viridis", (0, 11)),
                                  (ax[0, 1], hm, "inverted: thickness (mm)", "viridis", (0, 11)),
                                  (ax[1, 0], tm, "as built: mass scale", "magma", (0.8, max(2, tm.max()))),
                                  (ax[1, 1], mm, "inverted: mass scale", "magma", (0.8, max(2, tm.max())))):
        im = axx.imshow(img.T, origin="lower", extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1])
        for n, d in inv.geo["scatter"]["defects"].items():
            r = d.get("diameter_mm", d.get("size_mm")) / 2
            axx.add_patch(plt.Circle((d["x"] - O[0], d["y"] - O[1]), r + 2, fill=False, ec="w", lw=0.8, ls="--"))
            axx.text(d["x"] - O[0] + r + 3, d["y"] - O[1] + r + 3, n, color="w", fontsize=8)
        axx.set_title(lab), axx.set_aspect("equal")
        plt.colorbar(im, ax=axx, shrink=0.8)
    fig.suptitle(f"Defect-scan inversion, 450-800 us, h_min {inv.hmin * 1e3:.1f} mm; misfit {hist[0][0]:.3f} -> {hist[-1][0]:.3f}")
    fig.tight_layout()
    fig.savefig(out / "maps.png", dpi=80)
    plt.close(fig)
    fig, axx = plt.subplots(figsize=(7, 4))
    hh = np.array([x[0] for x in hist]); per = np.array([x[2] for x in hist])
    axx.semilogy(hh, "k", label="mean misfit")
    for k in KS:
        axx.semilogy(per[:, k - 1], lw=0.8, label=f"ex{k}")
    axx.legend(), axx.set_xlabel("iteration"), axx.set_ylabel("misfit")
    fig.tight_layout(); fig.savefig(out / "history.png", dpi=80); plt.close(fig)
    # per-defect summary
    rows = {}
    for n, d in inv.geo["scatter"]["defects"].items():
        near = np.hypot(inv.Xc * 1e3 - d["x"], inv.Yc * 1e3 - d["y"]) <= d.get("diameter_mm", d.get("size_mm")) / 2 + 3
        rows[n] = dict(true_min_thickness_mm=float(tt[near].min()), inverted_min_thickness_mm=float(hm[near].min()),
                       true_max_mass=float(tm[near].max()), inverted_max_mass=float(mm[near].max()))
    far = np.ones_like(hm, bool)
    for d in inv.geo["scatter"]["defects"].values():
        far &= np.hypot(inv.Xc * 1e3 - d["x"], inv.Yc * 1e3 - d["y"]) > 25
    rows["background"] = dict(thickness_mm_p5_p95=[float(np.percentile(hm[far], 5)), float(np.percentile(hm[far], 95))],
                              mass_p5_p95=[float(np.percentile(mm[far], 5)), float(np.percentile(mm[far], 95))])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--hmin", type=float, default=1.0, help="minimum thickness (mm)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    out = REPO / "ldv_pmma_results" / "inversion" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    inv = Inversion(hmin=a.hmin * 1e-3)
    vg = jax.jit(jax.value_and_grad(inv.loss, has_aux=True))
    m = jnp.zeros((2, *inv.nc), jnp.float32)
    t0 = time.perf_counter()
    (L, (per, tv)), g = vg(m, a.tv)
    print(f"start: misfit per file {np.round(np.asarray(per), 4)}, model grid {tuple(inv.nc)}, "
          f"{time.perf_counter() - t0:.0f} s incl. compile", flush=True)
    if a.bench:
        t0 = time.perf_counter(); jax.block_until_ready(vg(m, a.tv)); print(f"one iteration {time.perf_counter() - t0:.1f} s")
        return
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
              f"min h {float(h.min()) * 1e3:.2f} mm max mass {float(jnp.exp(m[1].max())):.2f} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
        if it % 10 == 9 or it == a.iters - 1:
            np.savez(out / "result.npz", m=np.asarray(m), Xc=inv.Xc, Yc=inv.Yc, iters_done=it + 1,
                     hist=np.array([x[0] for x in hist]))
    np.savez(out / "result.npz", m=np.asarray(m), Xc=inv.Xc, Yc=inv.Yc, iters_done=len(hist),
             hist=np.array([x[0] for x in hist]))
    rows = plot(inv, np.asarray(m), hist, out)
    (out / "settings.json").write_text(json.dumps(dict(vars(a), window_us=[T_WIN[0] * 1e6, T_WIN[1] * 1e6],
                                                       taper_us=TAPER * 1e6, spacing_mm=SPACING * 1e3,
                                                       hmax_mm=inv.hmax * 1e3, sources_m=inv.src.tolist(), s=inv.s,
                                                       Q=inv.Q, misfit_start=hist[0][0], misfit_end=hist[-1][0],
                                                       per_file_end=hist[-1][2], defects=rows), indent=1))
    print(json.dumps(rows, indent=1))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
