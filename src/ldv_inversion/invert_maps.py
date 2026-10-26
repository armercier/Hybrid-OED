"""Stage B: blind FWI of mass and rotary-inertia maps from the LDV scan (no cube prior).

Unknowns: a = log(mass scale), b = log(rotary-inertia scale) of the Mindlin plate on a
coarse grid (--spacing) over the scan area, outside --source-mask of the transducer;
bilinear interpolation to the FD grid; start a = b = 0 (plain plate). Gradients by
JAX through the remat'd solver; misfit = closed-form-source-filter misfit on the
train points (ldv_problem.LDVProblem) + TV on a and b. The held-out misfit (test
points behind / before the cube) is evaluated every --eval-every iterations.

Usage:  python src/ldv_inversion/invert_maps.py --iterations 40
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ldv_problem import CUBE_SALVUS, OUT, LDVProblem

import jax
import jax.numpy as jnp
import optax
from jax.scipy.ndimage import map_coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--spacing", type=float, default=2.5e-3)
    parser.add_argument("--source-mask", type=float, default=0.03)
    parser.add_argument("--tv", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--tag", default="maps_v1")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    prob = LDVProblem()
    lo_s, hi_s = prob.P.min(0), prob.P.max(0)
    nc = np.round((hi_s - lo_s) / args.spacing).astype(int) + 1
    Xc, Yc = np.meshgrid(lo_s[0] + np.arange(nc[0]) * args.spacing, lo_s[1] + np.arange(nc[1]) * args.spacing,
                         indexing="ij")
    mask = (np.hypot(Xc - prob.src[0], Yc - prob.src[1]) > args.source_mask).astype(np.float32)
    mask_j = jnp.asarray(mask)
    coords = [jnp.asarray((prob.X - lo_s[0]) / args.spacing, jnp.float32),
              jnp.asarray((prob.Y - lo_s[1]) / args.spacing, jnp.float32)]

    def fine(c):  # coarse log-field -> FD grid (zero outside the scan area)
        return map_coordinates(mask_j * c, coords, order=1, mode="constant", cval=0.0)

    def tv(c):
        m = mask_j * c
        gx, gy = m[1:, :-1] - m[:-1, :-1], m[:-1, 1:] - m[:-1, :-1]
        return jnp.sum(jnp.sqrt(gx**2 + gy**2 + 1e-6) - 1e-3)

    rec_train = prob.receivers(prob.train)
    obs_train = jnp.asarray(prob.obs_all[:, prob.train], jnp.float32)
    idx_all = np.concatenate([prob.train, prob.test])
    rec_all = prob.receivers(idx_all)

    def loss(params):
        a, b = params
        syn = prob.simulate(rec_train, mass_scale=jnp.exp(fine(a)), inertia_scale=jnp.exp(fine(b)))
        data = prob.misfit(syn, obs_train)
        reg = args.tv * (tv(a) + tv(b))
        return data + reg, (data, reg)

    grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True))
    full = jax.jit(lambda p: prob.simulate(rec_all, mass_scale=jnp.exp(fine(p[0])), inertia_scale=jnp.exp(fine(p[1]))))
    params = (jnp.zeros(tuple(nc), jnp.float32), jnp.zeros(tuple(nc), jnp.float32))
    opt = optax.adam(args.lr)
    state = opt.init(params)

    history = []
    started = time.perf_counter()
    for it in range(args.iterations + 1):
        (total, (data, reg)), g = grad_fn(params)
        rec = dict(iteration=it, loss=float(total), data=float(data), tv=float(reg), t=time.perf_counter() - started)
        if it % args.eval_every == 0 or it == args.iterations:
            m, _ = prob.evaluate(full(params), idx_all)
            rec.update(test_behind=m["behind"], test_before=m["before"])
            a = np.asarray(mask * params[0]), np.asarray(mask * params[1])
            k = np.unravel_index(np.argmax(a[1]), a[1].shape)
            rec.update(peak_inertia=float(np.exp(a[1].max())), peak_mass=float(np.exp(a[0].max())),
                       peak_at=[float(Xc[k]), float(Yc[k])])
            print(f"{it:3d} data {data:.4f} tv {reg:.4f} | test behind {m['behind']:.3f} before {m['before']:.3f} | "
                  f"max inertia x{rec['peak_inertia']:.1f} mass x{rec['peak_mass']:.2f} at "
                  f"({rec['peak_at'][0]*1e3:+.0f},{rec['peak_at'][1]*1e3:+.0f}) mm  t={rec['t']:.0f}s", flush=True)
        history.append(rec)
        if it == args.iterations:
            break
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)

    np.savez_compressed(OUT / f"{args.tag}.npz", log_mass=np.asarray(mask * params[0]),
                        log_inertia=np.asarray(mask * params[1]), Xc=Xc, Yc=Yc, P=prob.P, src=prob.src)
    (OUT / f"{args.tag}.json").write_text(json.dumps(dict(args=vars(args), history=history,
                                                          salvus_cube=CUBE_SALVUS.tolist()), indent=1))


if __name__ == "__main__":
    main()
