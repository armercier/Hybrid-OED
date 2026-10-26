"""Scattered-field (FB)PINN for the 2-D acoustic wave equation (feasibility, step 1).

Normalised coordinates: X = (x - x_c) / L, Y likewise, tau = t / T with T = L / c0,
so the target box is [-1, 1]^2 and c0 = 1.  With u = u0 + du and constant density,

    (1/c^2) du_tt - lap(du) = -(1/c^2 - 1) u0_tt =: -G(x, t),

whose source G is confined to the inclusion; u0 is the (converged, r = 4) FD
incident field.  du = A(tau) * U(x, tau) with A = 0 before the first incident
arrival at the inclusion (tau_a), which hard-enforces zero initial conditions.
An optional first-order absorbing condition du_tau + n.grad(du) = 0 is imposed
softly on the box faces.

Models:
  * fbpinn: overlapping subdomains on a regular (X, Y, tau) vertex grid; every
    cell is covered by its 8 corner subnetworks, blended with a C2 (quintic
    smoothstep) partition of unity;
  * pinn: one MLP, optionally with random Fourier features.

Usage:  python src/pinn_feasibility/scattered_pinn.py --model fbpinn --steps 20000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass

import numpy as np

from common import BOX_CENTER, BOX_HALF, F0, REFERENCE_DIR, RESULTS, T_SCALE

import jax
import jax.numpy as jnp
import optax
from jax.scipy.ndimage import map_coordinates

OMEGA = 2 * np.pi * F0 * T_SCALE  # normalised angular frequency at f0 (~18.7)


@dataclass
class Config:
    source: int = 0
    model: str = "fbpinn"
    h_space: float = 0.25
    h_time: float = 0.25
    hidden: int = 32
    layers: int = 2
    pinn_hidden: int = 256
    pinn_layers: int = 4
    fourier_features: int = 0
    fourier_scale: float = 10.0
    points: int = 64
    source_points: int = 64
    boundary_points: int = 32
    bc_weight: float = 1.0
    ramp_sigma: float = 0.3
    steps: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5
    eval_every: int = 1000
    # "physics": PDE residual loss (the PINN).  "data": physics-free regression on
    # the r4 reference field at the same collocation points, to test whether the
    # architecture can represent the solution at all.
    fit: str = "physics"
    # Causal time-marching (0 = train all time slabs jointly).  A window of
    # ``window`` time slabs slides from early to late, ``march_steps`` per slab,
    # followed by ``steps`` global fine-tuning steps.
    march_steps: int = 0
    window: int = 4
    march_points: int = 256
    march_source_points: int = 128
    march_boundary_points: int = 64
    seed: int = 0
    tag: str = ""


# --------------------------------------------------------------------------- data


class Problem:
    """Reference fields, source term and evaluation sets for one source."""

    def __init__(self, source: int):
        d = np.load(REFERENCE_DIR / f"source_{source}.npz")
        self.t = d["t"]
        self.dt = float(d["dt"])
        bx, by = d["box_x"], d["box_y"]
        self.box_x0, self.box_y0, self.dxb = bx[0], by[0], bx[1] - bx[0]
        u0, u = d["field_homo"], d["field_true"]
        inv_c2 = (float(d["c0"]) / d["c_true"]) ** 2

        dtau = self.dt / T_SCALE
        u0_tt = np.zeros_like(u0)
        u0_tt[1:-1] = (u0[2:] - 2 * u0[1:-1] + u0[:-2]) / dtau**2
        mask = np.abs(inv_c2 - 1) > 1e-6
        ii, jj = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        i0, i1, j0, j1 = ii[0] - 2, ii[-1] + 3, jj[0] - 2, jj[-1] + 3
        self.S = float(np.max(np.abs(u0[:, i0:i1, j0:j1])))
        G = (inv_c2 - 1)[None, i0:i1, j0:j1] * u0_tt[:, i0:i1, j0:j1] / self.S
        self.G = jnp.asarray(G, jnp.float32)
        self.G_x0, self.G_y0 = bx[i0], by[j0]
        self.inv_c2 = jnp.asarray(inv_c2, jnp.float32)

        g = np.max(np.abs(G), axis=(1, 2))
        self.tau_a = float(self.t[np.argmax(g > 1e-3 * g.max())] / T_SCALE) - 0.05
        self.tau_max = float(self.t[-1] / T_SCALE)
        self.incl_lo = (np.array([bx[i0], by[j0]]) - BOX_CENTER) / BOX_HALF
        self.incl_hi = (np.array([bx[i1 - 1], by[j1 - 1]]) - BOX_CENTER) / BOX_HALF

        # Evaluation: receiver traces (FWI-normalised) and a subsampled box field.
        self.rec_m, self.n_ring = d["rec_m"], int(d["n_ring"])
        # Each FD level is normalised by its own total-trace maximum (the FWI
        # convention); the source amplitude differs between grid resolutions.
        self.norms = {r: np.max(np.abs(d[f"tr_true_r{r}"]), axis=0) for r in (1, 2, 4)}
        self.du_traces = {r: d[f"tr_true_r{r}"] - d[f"tr_homo_r{r}"] for r in (1, 2, 4)}
        Xg = (bx - BOX_CENTER[0]) / BOX_HALF
        Yg = (by - BOX_CENTER[1]) / BOX_HALF
        si = np.where(np.abs(Xg) <= 1)[0][::2]
        sj = np.where(np.abs(Yg) <= 1)[0][::2]
        sk = np.arange(0, len(self.t), 8)
        self.field_ref = ((u - u0) / self.S)[np.ix_(sk, si, sj)]
        self.du_full = jnp.asarray((u - u0) / self.S, jnp.float32)
        K, I, J = np.meshgrid(sk, si, sj, indexing="ij")
        self.field_pts = np.stack([Xg[I], Yg[J], self.t[K] / T_SCALE], -1).reshape(-1, 3)
        R, Tt = np.meshgrid(np.arange(len(self.rec_m)), self.t, indexing="ij")
        xy = (self.rec_m - BOX_CENTER) / BOX_HALF
        self.trace_pts = np.concatenate(
            [xy[R.ravel()], (Tt.ravel() / T_SCALE)[:, None]], 1
        )  # (nrec * nt, 3), receiver-major

    def source_term(self, x):
        xm, ym = x[0] * BOX_HALF + BOX_CENTER[0], x[1] * BOX_HALF + BOX_CENTER[1]
        coords = [x[2] * T_SCALE / self.dt, (xm - self.G_x0) / self.dxb, (ym - self.G_y0) / self.dxb]
        return map_coordinates(self.G, coords, order=1, mode="constant", cval=0.0)

    def du_ref_at(self, x):
        xm, ym = x[0] * BOX_HALF + BOX_CENTER[0], x[1] * BOX_HALF + BOX_CENTER[1]
        coords = [x[2] * T_SCALE / self.dt, (xm - self.box_x0) / self.dxb, (ym - self.box_y0) / self.dxb]
        return map_coordinates(self.du_full, coords, order=1, mode="nearest")

    def inv_c2_at(self, x):
        xm, ym = x[0] * BOX_HALF + BOX_CENTER[0], x[1] * BOX_HALF + BOX_CENTER[1]
        coords = [(xm - self.box_x0) / self.dxb, (ym - self.box_y0) / self.dxb]
        return map_coordinates(self.inv_c2, coords, order=1, mode="nearest")

    def trace_error(self, du_pred_traces: np.ndarray, level: int = 4) -> dict[str, float]:
        """Relative L2 error vs the r4 scattered traces, in the FWI trace normalisation.

        ``level`` selects the normalisation of the prediction (4 for the PINN,
        whose amplitude comes from the r4 incident field).
        """
        ref = self.du_traces[4] / self.norms[4]
        pred = du_pred_traces / self.norms[level]
        out = {}
        for name, sl in (("ring", slice(0, self.n_ring)), ("diag", slice(self.n_ring, None)),
                         ("all", slice(None))):
            out[name] = float(np.linalg.norm(pred[:, sl] - ref[:, sl]) / np.linalg.norm(ref[:, sl]))
        return out


# -------------------------------------------------------------------------- model


def init_mlp(key, sizes, n=None):
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        shape = (din, dout) if n is None else (n, din, dout)
        W = jax.random.normal(k, shape) * np.sqrt(2.0 / (din + dout))
        b = jnp.zeros(shape[:-2] + (dout,))
        params.append((W, b))
    return params


def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return (x @ W + b)[0]


def smoothstep(s):
    s = jnp.clip(s, 0.0, 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


OFFSETS = jnp.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], jnp.float32)


class Model:
    def __init__(self, cfg: Config, prob: Problem):
        self.cfg, self.prob = cfg, prob
        self.lo = jnp.array([-1.0, -1.0, prob.tau_a])
        span = np.array([2.0, 2.0, prob.tau_max - prob.tau_a])
        hs = np.array([cfg.h_space, cfg.h_space, cfg.h_time])
        self.ncell = np.maximum(np.round(span / hs).astype(int), 1)
        self.h = jnp.asarray(span / self.ncell, jnp.float32)
        self.nvert = self.ncell + 1
        key = jax.random.PRNGKey(cfg.seed)
        if cfg.model == "fbpinn":
            n = int(np.prod(self.nvert))
            self.params = init_mlp(key, [3] + [cfg.hidden] * cfg.layers + [1], n)
        else:
            k1, k2 = jax.random.split(key)
            din = 3 + 2 * cfg.fourier_features
            self.B = cfg.fourier_scale * jax.random.normal(k2, (3, cfg.fourier_features))
            self.params = init_mlp(k1, [din] + [cfg.pinn_hidden] * cfg.pinn_layers + [1])
        self.n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(self.params)))

    def ramp(self, tau):
        s = (tau - self.prob.tau_a) / self.cfg.ramp_sigma
        return jnp.where(s > 0, -jnp.expm1(-s * s), 0.0)

    def corner_ids(self, cell):
        v = cell[None, :] + OFFSETS.astype(jnp.int32)
        return (v[:, 0] * self.nvert[1] + v[:, 1]) * self.nvert[2] + v[:, 2]

    def cell_of(self, x):
        c = jnp.floor((x - self.lo) / self.h).astype(jnp.int32)
        return jnp.clip(c, 0, jnp.asarray(self.ncell) - 1)

    def fb_value(self, g, cell, x):
        lo = self.lo + cell * self.h
        S = smoothstep((x - lo) / self.h)
        w = jnp.prod(jnp.where(OFFSETS == 1, S, 1.0 - S), axis=1)
        outs = jax.vmap(lambda p, off: mlp(p, (x - lo) / self.h - off))(g, OFFSETS)
        return jnp.sum(w * outs)

    def field_fn(self, params, cell):
        """du(x) restricted to one cell (fbpinn) or global (pinn)."""
        if self.cfg.model == "fbpinn":
            g = jax.tree_util.tree_map(lambda l: l[self.corner_ids(cell)], params)
            return lambda x: self.ramp(x[2]) * self.fb_value(g, cell, x)

        def f(x):
            z = x - jnp.array([0.0, 0.0, 0.5 * (self.prob.tau_a + self.prob.tau_max)])
            z = z / jnp.array([1.0, 1.0, 0.5 * (self.prob.tau_max - self.prob.tau_a)])
            if self.cfg.fourier_features:
                proj = (x * jnp.array([1.0, 1.0, 1.0])) @ self.B
                z = jnp.concatenate([z, jnp.sin(proj), jnp.cos(proj)])
            return self.ramp(x[2]) * mlp(params, z)
        return f

    def evaluate(self, params, pts: np.ndarray, chunk: int = 8192) -> np.ndarray:
        @jax.jit
        def run(p, X):
            return jax.vmap(lambda x: self.field_fn(p, self.cell_of(x))(x))(X)
        n = len(pts)
        pad = (-n) % chunk
        P = np.concatenate([pts, np.repeat(pts[-1:], pad, 0)]).astype(np.float32)
        out = [np.asarray(run(params, jnp.asarray(P[i:i + chunk]))) for i in range(0, len(P), chunk)]
        return np.concatenate(out)[:n]


# ----------------------------------------------------------------------- training


def build_regions(model: Model, prob: Problem):
    """Sampling regions (cell, lo, hi): interior cells, inclusion refinement, faces."""
    nc, h, lo0 = model.ncell, np.asarray(model.h), np.asarray(model.lo)
    cells = np.stack(np.meshgrid(*[np.arange(n) for n in nc], indexing="ij"), -1).reshape(-1, 3)
    clo, chi = lo0 + cells * h, lo0 + (cells + 1) * h
    interior = (cells, clo, chi)

    ilo = np.array([*prob.incl_lo, -np.inf])
    ihi = np.array([*prob.incl_hi, np.inf])
    slo, shi = np.maximum(clo, ilo), np.minimum(chi, ihi)
    keep = np.all(shi > slo, axis=1)
    source = (cells[keep], slo[keep], shi[keep])

    faces = []
    for dim in (0, 1):
        for side, value in ((0, -1.0), (nc[dim] - 1, 1.0)):
            sel = cells[:, dim] == side
            flo, fhi = clo[sel].copy(), chi[sel].copy()
            flo[:, dim] = fhi[:, dim] = value
            normal = np.zeros((sel.sum(), 3))
            normal[:, dim] = value
            faces.append((cells[sel], flo, fhi, normal))
    boundary = tuple(np.concatenate([f[i] for f in faces]) for i in range(4))

    def time_major(group):
        """Reshape a region group to (n_time_slabs, regions_per_slab, ...)."""
        order = np.argsort(group[0][:, 2], kind="stable")
        return tuple(
            jnp.asarray(a[order].reshape(nc[2], -1, *a.shape[1:]),
                        jnp.int32 if a.dtype.kind == "i" else jnp.float32)
            for a in group
        )
    return time_major(interior), time_major(source), time_major(boundary)


def sample(key, lo, hi, n):
    u = jax.random.uniform(key, (lo.shape[0], n, 3))
    return lo[:, None, :] + u * (hi - lo)[:, None, :]


def second_derivs(f, x):
    def along(e):
        return jax.jvp(lambda y: jax.jvp(f, (y,), (e,))[1], (x,), (e,))
    return jax.vmap(along)(jnp.eye(3))  # (first (3,), second (3,))


def first_derivs(f, x):
    return jax.vmap(lambda e: jax.jvp(f, (x,), (e,))[1])(jnp.eye(3))


def make_loss(model: Model, prob: Problem, regions, window: int, points: int,
              source_points: int, boundary_points: int):
    """Loss over ``window`` consecutive time slabs starting at a traced index."""
    cfg = model.cfg

    def pde_residuals(params, cells, pts):
        def one(cell, P):
            f = model.field_fn(params, cell)

            def r(x):
                _, d2 = second_derivs(f, x)
                return (prob.inv_c2_at(x) * d2[2] - d2[0] - d2[1] + prob.source_term(x)) / OMEGA**2
            return jax.vmap(r)(P)
        return jax.vmap(one)(cells, pts)

    def bc_residuals(params, cells, pts, normals):
        def one(cell, P, n):
            f = model.field_fn(params, cell)
            return jax.vmap(lambda x: (lambda d: d[2] + n[0] * d[0] + n[1] * d[1])(first_derivs(f, x)))(P) / OMEGA
        return jax.vmap(one)(cells, pts, normals)

    def take(group, start):
        return tuple(
            jax.lax.dynamic_slice_in_dim(a, start, window, axis=0).reshape(-1, *a.shape[2:])
            for a in group
        )

    def loss(params, key, start):
        (ic, ilo, ihi), (sc, slo, shi), (bc, blo, bhi, bn) = (take(g, start) for g in regions)
        k1, k2, k3 = jax.random.split(key, 3)
        r_int = pde_residuals(params, ic, sample(k1, ilo, ihi, points))
        r_src = pde_residuals(params, sc, sample(k2, slo, shi, source_points))
        terms = {"pde": jnp.mean(r_int**2), "src": jnp.mean(r_src**2)}
        total = terms["pde"] + terms["src"]
        if cfg.bc_weight > 0:
            r_bc = bc_residuals(params, bc, sample(k3, blo, bhi, boundary_points), bn)
            terms["bc"] = jnp.mean(r_bc**2)
            total = total + cfg.bc_weight * terms["bc"]
        return total, terms

    return loss


def make_data_loss(model: Model, prob: Problem, regions, points: int):
    """Physics-free regression of du on the reference field (all time slabs)."""
    (ic, ilo, ihi), _, _ = (tuple(a.reshape(-1, *a.shape[2:]) for a in g) for g in regions)

    def loss(params, key, start):
        del start
        pts = sample(key, ilo, ihi, points)

        def one(cell, P):
            f = model.field_fn(params, cell)
            return jax.vmap(lambda x: f(x) - prob.du_ref_at(x))(P)
        err = jax.vmap(one)(ic, pts)
        return jnp.mean(err**2), {"data": jnp.mean(err**2)}
    return loss


def metrics(model: Model, prob: Problem, params) -> dict[str, float]:
    du = model.evaluate(params, prob.trace_pts).reshape(len(prob.rec_m), -1).T * prob.S
    field = model.evaluate(params, prob.field_pts).reshape(prob.field_ref.shape)
    m = {f"trace_{k}": v for k, v in prob.trace_error(du).items()}
    m["field"] = float(np.linalg.norm(field - prob.field_ref) / np.linalg.norm(prob.field_ref))
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Config(**vars(parser.parse_args()))
    prob = Problem(cfg.source)
    model = Model(cfg, prob)
    regions = build_regions(model, prob)
    name = cfg.tag or f"{cfg.model}_s{cfg.source}"
    out_dir = RESULTS / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{name}: params={model.n_params:,}  cells={tuple(model.ncell)}  "
          f"tau=[{prob.tau_a:.2f},{prob.tau_max:.2f}]  regions int/src/bc="
          f"{regions[0][0][0].size // 3}/{regions[1][0][0].size // 3}/{regions[2][0][0].size // 3} per slab")
    fd_ctx = {f"fd_r{r}": prob.trace_error(prob.du_traces[r], level=r)["all"] for r in (1, 2)}
    print("reference context (scattered trace rel. err vs r4):", fd_ctx)

    n_slabs = int(model.ncell[2])
    window = min(cfg.window, n_slabs)
    march_total = cfg.march_steps * n_slabs
    sched = optax.join_schedules(
        [optax.constant_schedule(cfg.lr),
         optax.exponential_decay(cfg.lr, max(cfg.steps, 1), cfg.lr_final / cfg.lr)],
        [march_total],
    )
    opt = optax.adam(sched)

    def make_step(loss_fn):
        @jax.jit
        def step(params, state, key, start):
            (loss, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, start)
            updates, state = opt.update(grads, state, params)
            return optax.apply_updates(params, updates), state, loss, terms
        return step

    physics_loss = make_loss(model, prob, regions, n_slabs, cfg.points,
                             cfg.source_points, cfg.boundary_points)
    physics_eval = jax.jit(lambda p, k: physics_loss(p, k, 0)[0])
    if cfg.fit == "data":
        global_step = make_step(make_data_loss(model, prob, regions, cfg.points))
    else:
        global_step = make_step(physics_loss)
    march_step = make_step(make_loss(model, prob, regions, window, cfg.march_points,
                                     cfg.march_source_points, cfg.march_boundary_points))

    def schedule(it):
        """(step function, first slab of the window) for iteration ``it``."""
        if it < march_total:
            slab = it // cfg.march_steps
            return march_step, int(np.clip(slab - window + 1, 0, n_slabs - window))
        return global_step, 0

    params, state = model.params, opt.init(model.params)
    key = jax.random.PRNGKey(cfg.seed + 1)
    history = []
    train_time = 0.0
    total_steps = march_total + cfg.steps
    compiled = set()
    for it in range(total_steps + 1):
        key, sub = jax.random.split(key)
        step, start = schedule(it)
        t0 = time.perf_counter()
        params, state, loss, terms = step(params, state, sub, start)
        loss.block_until_ready()
        if id(step) in compiled:  # exclude compilation
            train_time += time.perf_counter() - t0
        compiled.add(id(step))
        if it % cfg.eval_every == 0 or it == total_steps:
            m = metrics(model, prob, params)
            m["physics_loss"] = float(np.mean([physics_eval(params, jax.random.PRNGKey(1000 + i))
                                               for i in range(4)]))
            rec = {"step": it, "window_start": start, "train_s": train_time, "loss": float(loss),
                   **{k: float(v) for k, v in terms.items()}, **m}
            history.append(rec)
            print(f"  {it:6d}  w={start:2d}  t={train_time:7.1f}s  loss={float(loss):.3e}  "
                  + "  ".join(f"{k}={float(v):.2e}" for k, v in terms.items())
                  + f"  | trace err all={m['trace_all']:.3f} ring={m['trace_ring']:.3f} "
                    f"diag={m['trace_diag']:.3f}  field={m['field']:.3f}  physics={m['physics_loss']:.2e}", flush=True)
            if not np.isfinite(float(loss)):
                break

    du = model.evaluate(params, prob.trace_pts).reshape(len(prob.rec_m), -1).T * prob.S
    np.savez_compressed(out_dir / "prediction.npz", du_traces=du,
                        field=model.evaluate(params, prob.field_pts).reshape(prob.field_ref.shape))
    (out_dir / "summary.json").write_text(json.dumps(
        {"config": dataclasses.asdict(cfg), "n_params": model.n_params,
         "fd_context": fd_ctx, "history": history,
         "sec_per_step": train_time / max(total_steps, 1), "train_s": train_time}, indent=2))


if __name__ == "__main__":
    main()
