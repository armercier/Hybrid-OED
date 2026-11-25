"""Truth test: does the as-built plate (holes as thinning, cube as added mass) explain the defect scan
better than the v1 inversion? Forward runs only, same data, window, sources and source-signature
estimation as invert_defect_scan.py.

Variants: pristine; v1 inversion result; as built; holes only; cube only; through-hole D3 at 0.5 / 1 / 2 mm;
blind holes 3 / 4 / 5 mm left; cube mass x0.5 / x1 / x2.

Usage:  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/truth_test.py
Output: ldv_pmma_results/inversion/truth_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from invert_defect_scan import Inversion  # noqa: E402

OUT = REPO / "ldv_pmma_results" / "inversion"


def main():
    inv = Inversion()
    c = inv.c
    X, Y = c.X * 1e3, c.Y * 1e3                       # FD grid, mm, defect-scan frame
    D = inv.geo["scatter"]["defects"]
    H = inv.H * 1e3
    rho_h = c.rho * inv.H

    def model(blind_left=4.0, through=1.0, cube_scale=1.0, holes=True, cube=True, shift=(0.0, 0.0)):
        h = np.full(X.shape, H)
        mass = np.asarray(inv.mass0).copy()
        for n, d in D.items():
            x, y = d["x"] + shift[0], d["y"] + shift[1]
            if n.startswith("D") and holes:
                inside = np.hypot(X - x, Y - y) <= d["diameter_mm"] / 2
                h[inside] = through if d["depth_mm"] >= H else blind_left
            if n == "M1" and cube:
                a = d["size_mm"] / 2
                inside = (np.abs(X - x) <= a) & (np.abs(Y - y) <= a)
                mass[inside] *= 1 + cube_scale * (d["mass_g"] * 1e-3 / (d["size_mm"] * 1e-3) ** 2) / rho_h
        return jnp.asarray(h * 1e-3, jnp.float32), jnp.asarray(mass, jnp.float32)

    fields = {}
    inv.fields = lambda m: fields["cur"]
    run = jax.jit(inv.misfits)
    dummy = jnp.zeros((2, *inv.nc), jnp.float32)
    res = {}

    def evaluate(name, f):
        fields["cur"] = f
        per = np.asarray(jax.jit(inv.misfits)(dummy)) if False else np.asarray(inv.misfits(dummy))
        res[name] = dict(mean=float(per.mean()), per_file=[round(float(v), 4) for v in per])
        print(f"{name:42s} misfit {per.mean():.4f}  [{' '.join(f'{v:.3f}' for v in per)}]", flush=True)

    one = np.ones(X.shape)
    evaluate("pristine (no defects)", (jnp.asarray(one * inv.H, jnp.float32), inv.mass0))
    r = np.load(OUT / "v1" / "result.npz")
    inv_v1 = Inversion.__new__(Inversion); inv_v1.__dict__.update(inv.__dict__)
    fields_v1 = Inversion.fields(inv_v1, jnp.asarray(r["m"]))
    evaluate("v1 inversion result", fields_v1)
    evaluate("AS BUILT (holes 4 mm left, D3 1 mm, cube)", model())
    evaluate("holes only", model(cube=False))
    evaluate("cube only", model(holes=False))
    for th in (0.5, 2.0):
        evaluate(f"as built, D3 {th} mm", model(through=th))
    for bl in (3.0, 5.0, 7.0):
        evaluate(f"as built, blind holes {bl} mm left", model(blind_left=bl))
    for cs in (0.5, 2.0):
        evaluate(f"as built, cube mass x{cs}", model(cube_scale=cs))
    for sh in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        evaluate(f"as built, all defects shifted {sh} mm", model(shift=sh))
    (OUT / "truth_test.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
