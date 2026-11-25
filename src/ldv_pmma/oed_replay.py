"""Adaptive OED replay on the PMMA defect scan: design -> acquire (nearest scanned point) -> FWI, with accumulation.

Same illumination criterion as the workshop paper and src/oed_ldv (problem.py:136, oed_ldv_summary.png):
for each source, Rademacher probes q on the (windowed) traces at the LDV points, sensitivity s = J^T q by one VJP,
h = mean over probes and sources of s^2, 3x3 smoothing, score = var(h / mean h) - beta log(mean h) over the model
grid; Adam on the LDV point coordinates. Nothing about the defects is used: the design sees only the current model.

Counting rule (b): budget N measured points, R rounds, N/R new points per round, every measured point is kept.
Round k: (1) the k-th of R interleaved groups of the regular layout of N points is the starting position of the new
points (with no design the replay measures exactly the regular layout); (2) design: a few Adam steps on the new
points, illumination of new + already measured points, current model; (3) acquire: each new point snapped to the
nearest scanned point not yet measured (the replay "lab": the dense 3 mm scan); (4) FWI updates on all points
measured so far (same model, background, window, signatures, TV and Adam as invert_thickness.py; the Adam state is
reset each round as in the paper, the model is carried). "No surface" points are found among the measured points
with the subset rule of make_subsets.py and excluded.

Background: calib_blind.json, no cube, as all reduced-scan runs (ldv_pmma_results/reduced_scan_runs.md).

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python src/ldv_pmma/oed_replay.py --tag o9 --spacing 9
  python src/ldv_pmma/oed_replay.py --tag test_cpu --test       # small CPU check (4 mm grid, few points)
Output: ldv_pmma_results/inversion/<tag>/{result.npz, settings.json, run.log}; subset ldv_work/prepared/subsets/<tag>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "ldv_pmma"))
PREP = REPO / "ldv_work" / "prepared"
CALB = REPO / "ldv_pmma_results" / "calib" / "calib_blind.json"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--spacing", type=int, default=9, help="regular layout the budget is taken from (6, 9, 12 mm)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--fwi-per-round", type=int, default=20, help="5 x 20 = 100, as the regular/random runs")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--design-on-background", action="store_true",
                    help="design every round on the initial (defect-free background) model instead of the current inversion: "
                         "a pre-planned, non-adaptive control with the same acquisition and inversion (23 Nov)")
    ap.add_argument("--criterion", choices=("var", "logmean"), default="var",
                    help="var: var(h / mean h) - beta log(mean h) (paper); logmean: -mean(log h) = -mean(log(h / mean h)) - log(mean h), "
                         "a peak-robust uniformity term (dark cells weigh, hot spots saturate) plus the overall illumination at weight 1 "
                         "(diagonal D-optimality); beta is ignored")
    ap.add_argument("--design-steps", type=int, default=5)
    ap.add_argument("--design-lr", type=float, default=3e-3, help="Adam step on point coordinates (m)")
    ap.add_argument("--design-opt", choices=("adam", "gdnorm"), default="adam",
                    help="adam: Adam (every coordinate moves ~lr whatever its gradient size); gdnorm: gradient descent "
                         "with one scale per step, the point with the largest gradient moves --design-move, the others "
                         "in proportion to their gradient")
    ap.add_argument("--design-move", type=float, default=8e-3, help="gdnorm: largest move per design step (m)")
    ap.add_argument("--move-all", action="store_true",
                    help="paper protocol (run_oed_multiseed_full_autodiff.py): all points start on the regular layout "
                         "and ALL move every round (design Adam state carried across rounds); each round's FWI uses only "
                         "the points measured in that round; the model is carried")
    ap.add_argument("--fwi-first", action="store_true",
                    help="no design in round 0: invert first on the starting layout, then design -> acquire -> FWI in the "
                         "next rounds (every design step uses a model that has seen data; R FWI blocks, R-1 moves)")
    ap.add_argument("--grad-check", default=None, metavar="TAG",
                    help="full-size check of the design gradient on the model of run TAG (fast vs remat, finite "
                         "differences at the points with the largest and median gradient), then exit")
    ap.add_argument("--random-moves-from", default=None, metavar="TAG",
                    help="move-all control: each point moves as far as in run TAG, in a uniformly random direction")
    ap.add_argument("--select", choices=("coverage", "targeted"), default=None,
                    help="adaptive selection among all unmeasured scan points instead of moving coordinates: greedy "
                         "information gain on linearised data changes. coverage: random smooth model perturbations; "
                         "targeted: the same, weighted towards where the current model departs from the background")
    ap.add_argument("--budget", type=int, default=31, help="--select: distinct points in total (6 start points on a 3 x 2 grid)")
    ap.add_argument("--score-layouts", nargs="+", default=None, metavar="ID",
                    help="evaluate the design criteria on existing subsets (background and v3 models), write "
                         "ldv_pmma_results/criterion_check/<tag>.json, then exit; no inversion")
    ap.add_argument("--offline-steps", type=int, default=0,
                    help="design all points in simulation only, on the background model (no data, no acquisition): Adam "
                         "with --design-lr and fresh probes each step, from subset --offline-start; the layout with the "
                         "best score on separate validation probes is snapped to scan points and saved as subset <tag>")
    ap.add_argument("--offline-start", default="rc30", help="starting subset for --offline-steps")
    ap.add_argument("--layout-groups", default=None, metavar="ID",
                    help="start positions from a 6 x 5 cell-centred subset (rc30), split into --rounds interleaved groups")
    ap.add_argument("--fwi-total", type=int, default=0, help="add the missing FWI updates to the last round to reach this total")
    ap.add_argument("--design-fresh-probes", action="store_true", help="new probes at every design step (default: fixed within a round)")
    ap.add_argument("--design-valid-every", type=int, default=0,
                    help="every N design steps, snap the new points to unmeasured scan points and score them on 4 separate "
                         "probe sets (with the measured points, current model); the best snapped candidate is measured")
    ap.add_argument("--score-terms", nargs="+", default=None, metavar="ID",
                    help="background model only: the two terms of the illumination score (variance of h/mean, mean) for the "
                         "given subsets and --term-seeds probe seeds, so any beta can be evaluated afterwards; writes "
                         "ldv_pmma_results/criterion_check/<tag>.json, then exit")
    ap.add_argument("--term-seeds", type=int, default=4)
    ap.add_argument("--probe-chunk", type=int, default=0,
                    help="fast design: process the probes in groups of this size (same result, memory of one group); 0 = all at once")
    ap.add_argument("--start-subset", default=None, metavar="ID", help="--move-all: start from this subset instead of the regular groups")
    ap.add_argument("--no-design", action="store_true", help="control: measure the interleaved regular groups as they are")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="CPU check: 4 mm grid, 2 rounds, few updates and probes")
    ap.add_argument("--fd-check", action="store_true", help="compare the design gradient with finite differences, then exit")
    ap.add_argument("--bench", action="store_true", help="one design step + one FWI update at full size, time and memory, then exit")
    ap.add_argument("--design-impl", choices=("remat", "fast"), default="remat",
                    help="remat: one (source, probe) term at a time, each rematerialised (17 Nov reference); "
                         "fast: one forward run per source, the probe VJPs batched (vmap) on it")
    ap.add_argument("--src-batch", action="store_true", help="fast: batch all sources (vmap); same as --src-chunk 4")
    ap.add_argument("--src-chunk", type=int, default=1,
                    help="fast: sources run together (vmap) per chunk, chunks looped (lax.map); 1 = one at a time, "
                         "2 = two at a time (~2x the design memory), must divide the number of sources")
    ap.add_argument("--design-ckpt", type=int, default=None,
                    help="time-step checkpoint interval of the solver inside the design (default: as the FWI, 100)")
    ap.add_argument("--bench-ref", action="store_true", help="bench: also run the remat reference and compare value/grad")
    ap.add_argument("--bench-skip-fwi", action="store_true", help="bench: skip the FWI-update timing")
    return ap.parse_args()


a = parse()
if a.test:  # coarse grid for a quick CPU check; must be set before the modules below build their grids
    import calibrate_pristine
    calibrate_pristine.DX, calibrate_pristine.SPONGE = 4e-3, 15
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import calibrate_pristine as cp  # noqa: E402
import invert_defect_scan as ids  # noqa: E402
import invert_thickness as it  # noqa: E402
import make_subsets  # noqa: E402
from make_subsets import nosurface_subset  # noqa: E402
from mindlin import mindlin_plate  # noqa: E402

if a.test:
    ids.DX = it.DX = cp.DX
    it.SPONGE = cp.SPONGE
KS = cp.KS


def interleaved_groups(spacing, rounds, R, usable):
    """Regular layout (every spacing/3-th scan point) split into `rounds` interleaved lattices: (i + 2 j) mod rounds."""
    ij = np.round((R - R.min(0)) / 3.0).astype(int)
    q = spacing // 3
    reg = (ij[:, 0] % q == 0) & (ij[:, 1] % q == 0) & usable
    I, J = ij[reg, 0] // q, ij[reg, 1] // q
    g = (I + 2 * J) % rounds
    idx = np.flatnonzero(reg)
    return [idx[g == k] for k in range(rounds)]


def main():
    out = REPO / "ldv_pmma_results" / "inversion" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    d = np.load(PREP / "scatter.npz")
    inv = it.ThicknessInversion(hmin=a.hmin * 1e-3, background=CALB, cube=False, nosurface=False)
    c = inv.c
    base_m = {k: np.asarray(inv.m[k]) for k in KS}               # usable-point masks per file (on c.use)
    use_idx = np.flatnonzero(c.use)                               # c.use position -> index in scatter.npz
    use_idx = use_idx[np.any([base_m[k][0] > 0 for k in inv.active], 0)]   # usable in the selected scan files
    make_subsets.KS = inv.active                                  # "no surface" check from the selected files only
    usable_all = np.zeros(len(d["R"]), bool); usable_all[use_idx] = True
    R_mm = d["R"]
    rounds = a.rounds if not a.test else 2
    fwi_per_round = a.fwi_per_round if not a.test else 2
    probes_n = a.probes if (not a.test or a.score_layouts or a.offline_steps or a.score_terms or a.probe_chunk) else 2
    dsteps = a.design_steps if not a.test else 2
    groups = interleaved_groups(a.spacing, a.rounds, R_mm, usable_all)
    if a.test:
        groups = [g[::12] for g in groups[:rounds]]
    if a.select:   # round 0: 3 x 2 cell-centred grid snapped to usable points; later rounds: only the counts are used
        lo_, hi_ = R_mm[use_idx].min(0), R_mm[use_idx].max(0)
        tr_ = cKDTree(R_mm[use_idx])
        cen = [lo_ + (np.array([i, j]) + 0.5) * (hi_ - lo_) / [3, 2] for i in range(3) for j in range(2)]
        groups = [use_idx[tr_.query(cen)[1]]] + [np.zeros(len(g), int) for g in np.array_split(np.arange(a.budget - 6), rounds - 1)]
    if a.layout_groups:
        idx_ = np.flatnonzero(np.load(PREP / "subsets" / f"{a.layout_groups}.npz")["keep"] & usable_all)
        assert len(idx_) == 30, "--layout-groups expects a 6 x 5 layout"
        lo_, hi_ = R_mm[use_idx].min(0), R_mm[use_idx].max(0)
        c_ = np.minimum(((R_mm[idx_] - lo_) / (hi_ - lo_) * [6, 5]).astype(int), [5, 4])
        g_ = (c_[:, 0] + 2 * c_[:, 1]) % a.rounds
        groups = [idx_[g_ == r_] for r_ in range(a.rounds)]
    budget = sum(len(g) for g in groups[:rounds])
    log(f"[{a.tag}] budget {budget} points in {rounds} rounds ({[len(g) for g in groups[:rounds]]}), "
        f"{'move-all (paper protocol), ' if a.move_all else ''}{'FWI first, ' if a.fwi_first else ''}{fwi_per_round} FWI updates per round, design {'none' if a.no_design else (f'{dsteps} Adam steps, lr {a.design_lr * 1e3:.1f} mm' if a.design_opt == 'adam' else f'{dsteps} normalised gradient steps, largest move {a.design_move * 1e3:.1f} mm')}, "
        f"{probes_n} probes, grid {c.nx} x {c.ny} ({cp.DX * 1e3:.0f} mm)")

    # ---- design: illumination at continuous LDV positions (m), current model m ----
    lo_scan, hi_scan = c.Ru.min(0), c.Ru.max(0)
    one = jnp.ones((c.nx, c.ny), jnp.float32)
    vp, vs, rho = c.vp * inv.s * one, c.vs * inv.s * one, c.rho * one

    ckpt = inv.ckpt if a.design_ckpt is None else a.design_ckpt

    def traces(m, rec_m, si):
        h, mass = inv.fields(m)
        rec_i = (rec_m - jnp.asarray(c.origin, jnp.float32)) / cp.DX
        tr = mindlin_plate(vp, vs, rho, h, si, c.stf, cp.DX, cp.DX, c.dt, rec_i, sponge_cells=cp.SPONGE,
                           kappa=inv.kappa, mass_scale=mass, damping=inv.damping, checkpoint_every=ckpt)[..., 0]
        return inv.taper * ((1 - c.w) * tr[c.i0] + c.w * tr[c.i0 + 1])

    SI = jnp.stack([inv.src_i[i] for i in inv.active_src])        # transducers driven in the selected scan files
    NS = len(inv.active_src)
    n_terms = NS * probes_n                    # (source, probe) pairs

    def sens_one(rec, m, i, key):
        """Sensitivity J^T q of the traces of source i // P to the log-thickness, probe i % P (Rademacher, fixed by key)."""
        j, pr = i // probes_n, i % probes_n
        out_, pull = jax.vjp(lambda mm: traces(mm, rec, SI[j]), m)
        q = jax.random.rademacher(jax.random.fold_in(jax.random.fold_in(key, j), pr), out_.shape).astype(out_.dtype)
        return pull(q / jnp.sqrt(out_.size))[0][0]

    def smooth(h):
        n0, n1 = h.shape
        hp = jnp.pad(h, 1, mode="edge")
        return sum(hp[i:i + n0, j:j + n1] for i in range(3) for j in range(3)) / 9.0

    def score(h):
        h = smooth(h)
        mean = jnp.mean(h)
        if a.criterion == "logmean":
            return -jnp.mean(jnp.log(h + 1e-3 * jax.lax.stop_gradient(mean)))
        return jnp.mean((h / mean - 1.0) ** 2) - a.beta * jnp.log(mean)

    def illumination(new, old, m, key):
        """Reference (one graph, all terms): only for the CPU check, needs ~n_terms x the memory."""
        rec = jnp.concatenate([new, old])
        sens = jax.lax.map(lambda i: sens_one(rec, m, i, key), jnp.arange(n_terms))
        return score(jnp.mean(sens ** 2, 0))

    def illumination_remat(new, old, m, key):
        """Same objective, rematerialised as in src/remat/remat_hypergrad.py (Oct 2026 report, Sec. 2): every
        (source, probe) term is wrapped in jax.checkpoint and mapped one at a time (lax.map), so the backward
        pass stores only the inputs of a term and recomputes it (memory ~ one term instead of all of them)."""
        rec = jnp.concatenate([new, old])
        term = jax.checkpoint(lambda i: sens_one(rec, m, i, key) ** 2)
        return score(jnp.mean(jax.lax.map(term, jnp.arange(n_terms)), 0))

    def sens_sq_source(rec, m, j, key):
        """Sum over the probes of (J^T q)^2 for source j: ONE forward run (vjp), the probe pullbacks batched with vmap.
        Same probes as sens_one (key folded with j, then with the probe index), so the objective is identical."""
        out_, pull = jax.vjp(lambda mm: traces(mm, rec, SI[j]), m)
        kj = jax.random.fold_in(key, j)
        Q = jax.vmap(lambda pr: jax.random.rademacher(jax.random.fold_in(kj, pr), out_.shape).astype(out_.dtype))(
            jnp.arange(probes_n)) / jnp.sqrt(out_.size)
        if a.probe_chunk and a.probe_chunk < probes_n:
            Qc = Q.reshape(probes_n // a.probe_chunk, a.probe_chunk, *Q.shape[1:])
            grp = jax.checkpoint(lambda q: jnp.sum(jax.vmap(lambda qq: pull(qq)[0][0])(q) ** 2, 0))   # store only each group's input
            return jnp.sum(jax.lax.map(grp, Qc), 0)
        S = jax.vmap(lambda q: pull(q)[0][0])(Q)
        return jnp.sum(S ** 2, 0)

    def illumination_fast(new, old, m, key):
        """Same objective as illumination_remat; memory traded for compute: per source, the forward run is shared by
        all probes and the probe adjoints run side by side. Each source is one rematerialised block (jax.checkpoint);
        sources in chunks of --src-chunk run together (vmap), chunks looped (lax.map); --src-batch = all at once."""
        rec = jnp.concatenate([new, old])
        blk = jax.checkpoint(lambda j: sens_sq_source(rec, m, j, key))
        chunk = NS if a.src_batch else a.src_chunk
        js = jnp.arange(NS).reshape(-1, chunk)                  # (n_chunks, chunk)
        tot = jnp.sum(jax.lax.map(jax.vmap(blk), js), (0, 1))
        return score(tot / n_terms)

    illum_vg_remat = jax.jit(jax.value_and_grad(illumination_remat))
    illum_vg_fast = jax.jit(jax.value_and_grad(illumination_fast))
    illum_vg = illum_vg_fast if a.design_impl == "fast" else illum_vg_remat
    illum_vg_ref = jax.jit(jax.value_and_grad(illumination))
    if a.select or a.score_layouts:
        rec_all = jnp.asarray(R_mm[use_idx] * 1e-3, jnp.float32)
        n_time = int(traces(jnp.zeros((1, *inv.nc), jnp.float32), rec_all[:1], SI[0]).shape[0])
        Om = jnp.asarray(np.random.default_rng(a.seed).normal(size=(16, n_time)) / np.sqrt(n_time), jnp.float32)

        @jax.jit
        def lin_features(m_, dm, si):
            """Change of every candidate's traces for a model change dm (one forward + tangent run), relative to the
            candidate's own trace level, compressed in time by a fixed random projection: (n_candidates, 16)."""
            y, t = jax.jvp(lambda mm: traces(mm, rec_all, si), (m_,), (dm,))
            rms = jnp.sqrt(jnp.mean(y ** 2, 0))
            return (Om @ (t / (rms + 1e-3 * jnp.max(rms)))).T

        def select_points(m_, measured_, n_new, k_):
            m0 = np.asarray(m_[0])
            rng = np.random.default_rng([a.seed, k_])
            w = 1.0
            if a.select == "targeted":
                dev = uniform_filter(np.abs(m0), 3, mode="nearest")
                w = 0.1 + dev / max(dev.max(), 1e-9)
            A = []
            for _ in range(probes_n):
                z = uniform_filter(rng.choice([-1.0, 1.0], m0.shape), 5, mode="nearest") * w
                dm = jnp.asarray((z / np.sqrt(np.mean(z ** 2)))[None], jnp.float32)
                A += [np.asarray(lin_features(m_, dm, SI[j])) for j in range(NS)]
            A = np.concatenate(A, 1)
            s2 = 0.01 * np.median(np.sum(A ** 2, 1))
            def deflate(i):
                v = A[i].copy()
                A[:] -= np.outer(A @ v, v) / (s2 + v @ v)
            free = ~measured_[use_idx]
            for i in np.flatnonzero(~free):
                deflate(i)
            out = []
            for _ in range(n_new):
                i = int(np.argmax(np.where(free, np.sum(A ** 2, 1), -1.0)))
                out.append(i); free[i] = False; deflate(i)
            return use_idx[out]

    clip = lambda r: jnp.clip(r, jnp.asarray(lo_scan, jnp.float32), jnp.asarray(hi_scan, jnp.float32))

    # ---- FWI on the measured points ----
    def set_mask(measured):
        ns = nosurface_subset(d, measured) if measured.sum() >= 30 else np.zeros_like(measured)
        keep = (measured & ~ns)[c.use].astype(np.float32)
        inv.m = {k: jnp.asarray(base_m[k] * keep[None, :]) for k in KS}
        inv.norm = {k: float(jnp.sum(inv.m[k] * inv.obs[k] ** 2)) for k in KS}
        return ns

    m = jnp.zeros((1, *inv.nc), jnp.float32)
    measured = np.zeros(len(R_mm), bool)
    tree_all = cKDTree(R_mm[use_idx])
    hist, rounds_log, traj = [], [], []
    key = jax.random.PRNGKey(a.seed)
    t_start = time.perf_counter()
    if a.offline_steps:
        keep0 = np.load(PREP / "subsets" / f"{a.offline_start}.npz")["keep"] & usable_all
        new = jnp.asarray(R_mm[keep0] * 1e-3, jnp.float32); old = jnp.zeros((0, 2), jnp.float32)
        f = jax.jit(illumination_fast); opt_r = optax.adam(a.design_lr); st_r = opt_r.init(new)
        vkeys = [jax.random.PRNGKey(100000 + i) for i in range(4)]             # never used for the gradient
        valid = lambda r: float(np.mean([float(f(r, old, m, kv)) for kv in vkeys]))
        tree = cKDTree(R_mm[use_idx])
        def snap(r):
            taken = []
            for p_ in np.asarray(r) * 1e3:
                taken.append(next(use_idx[j] for j in tree.query(p_, 16)[1] if use_idx[j] not in taken))
            k_ = np.zeros(len(R_mm), bool); k_[taken] = True
            return k_
        best_v, keep = valid(new), keep0.copy(); path = [dict(step=0, valid_snapped=best_v)]
        log(f"offline design from {a.offline_start}: {len(new)} points, validation score {best_v:.4f}")
        every = max(1, a.offline_steps // 12)
        for s_ in range(1, a.offline_steps + 1):
            t0 = time.perf_counter(); val, g = illum_vg(new, old, m, jax.random.fold_in(key, s_))
            upd, st_r = opt_r.update(g, st_r); new = clip(optax.apply_updates(new, upd))
            if s_ % every == 0 or s_ == a.offline_steps:
                k_ = snap(new); v = valid(jnp.asarray(R_mm[k_] * 1e-3, jnp.float32))
                path.append(dict(step=s_, train=float(val), valid_snapped=v))
                if v < best_v: best_v, keep = v, k_
                log(f"step {s_}: train {float(val):.4f}, validation after snapping {v:.4f}, best {best_v:.4f}, "
                    f"move from start median {np.median(np.linalg.norm(np.asarray(new) - R_mm[keep0] * 1e-3, axis=1)) * 1e3:.1f} mm ({time.perf_counter() - t0:.0f} s)")
        final_v = best_v
        np.savez(PREP / "subsets" / f"{a.tag}.npz", keep=keep, nosurface=nosurface_subset(d, keep),
                 rule=f"offline illumination design from {a.offline_start}; no surface: subset rule of make_subsets.py")
        (out / "offline.json").write_text(json.dumps(dict(vars(a), path=path, snapped_valid=final_v, points_mm=R_mm[keep].tolist()), indent=1))
        log(f"offline design done: best validation after snapping {final_v:.4f}; {int((keep & ~keep0).sum())} of {int(keep.sum())} points differ from the start")
        return
    if a.score_terms:
        @jax.jit
        def terms(new, m_, key_):
            blk = jax.checkpoint(lambda j: sens_sq_source(new, m_, j, key_))
            h = smooth(jnp.sum(jax.lax.map(jax.vmap(blk), jnp.arange(NS).reshape(-1, a.src_chunk)), (0, 1)) / n_terms)
            mean = jnp.mean(h)
            return jnp.mean((h / mean - 1.0) ** 2), mean
        res = {}
        for t in a.score_terms:
            keep = np.load(PREP / "subsets" / f"{t}.npz")["keep"] & usable_all
            pts = jnp.asarray(R_mm[keep] * 1e-3, jnp.float32)
            res[t] = dict(points=int(keep.sum()), var=[], mean=[])
            for s_ in range(a.term_seeds):
                v_, mu_ = terms(pts, m, jax.random.PRNGKey(5000 + s_))
                res[t]["var"].append(float(v_)); res[t]["mean"].append(float(mu_))
            log(f"{t}: var {np.round(res[t]['var'], 4).tolist()} mean {np.round(res[t]['mean'], 6).tolist()}")
        (REPO / "ldv_pmma_results" / "criterion_check").mkdir(exist_ok=True)
        (REPO / "ldv_pmma_results" / "criterion_check" / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
        log("score-terms done")
        return
    if a.score_layouts:
        INV = REPO / "ldv_pmma_results" / "inversion"
        f = jax.jit(illumination_fast)
        pos = -np.ones(len(R_mm), int); pos[use_idx] = np.arange(len(use_idx))
        out_c = {}
        for name, m_ in (("background", m), ("v3", jnp.asarray(np.load(INV / "v3" / "result.npz")["m"], jnp.float32))):
            rng = np.random.default_rng([a.seed, 999]); A = []
            for _ in range(probes_n):
                z = uniform_filter(rng.choice([-1.0, 1.0], np.asarray(m_[0]).shape), 5, mode="nearest")
                dm = jnp.asarray((z / np.sqrt(np.mean(z ** 2)))[None], jnp.float32)
                A += [np.asarray(lin_features(m_, dm, SI[j])) for j in range(NS)]
            A = np.concatenate(A, 1); s2 = 0.01 * np.median(np.sum(A ** 2, 1))
            for t in a.score_layouts:
                keep = np.load(PREP / "subsets" / f"{t}.npz")["keep"] & usable_all
                B = A[pos[keep]]
                out_c.setdefault(t, dict(points=int(keep.sum())))[name] = dict(
                    illumination=float(f(jnp.asarray(R_mm[keep] * 1e-3, jnp.float32), jnp.zeros((0, 2), jnp.float32), m_, key)),
                    logdet=float(np.linalg.slogdet(np.eye(len(B)) + B @ B.T / s2)[1]))
                log(f"{name} {t}: {out_c[t][name]}")
        (REPO / "ldv_pmma_results" / "criterion_check").mkdir(exist_ok=True)
        (REPO / "ldv_pmma_results" / "criterion_check" / f"{a.tag}.json").write_text(json.dumps(out_c, indent=1))
        log("score-layouts done")
        return
    if a.grad_check:
        m_chk = jnp.asarray(np.load(REPO / "ldv_pmma_results" / "inversion" / a.grad_check / "result.npz")["m"], jnp.float32)
        new = jnp.asarray(R_mm[np.concatenate(groups[:rounds])] * 1e-3, jnp.float32)
        old = jnp.zeros((0, 2), jnp.float32)
        kk = jax.random.fold_in(key, 1)
        t0 = time.perf_counter(); val, g = illum_vg_fast(new, old, m_chk, kk); g.block_until_ready()
        log(f"fast: value {float(val):.6f}, {time.perf_counter() - t0:.0f} s incl. compile")
        t0 = time.perf_counter(); vr, gr = illum_vg_remat(new, old, m_chk, kk); gr.block_until_ready()
        cos = float(jnp.sum(g * gr) / (jnp.linalg.norm(g) * jnp.linalg.norm(gr)))
        log(f"remat reference: value {float(vr):.6f}, {time.perf_counter() - t0:.0f} s; max |g - g_ref| / max |g_ref| "
            f"{float(jnp.max(jnp.abs(g - gr)) / jnp.max(jnp.abs(gr))):.2e}, cosine {cos:.6f}")
        f = jax.jit(illumination_fast)
        gn = np.asarray(jnp.linalg.norm(g, axis=1)); order = np.argsort(gn)
        picks = [("largest", int(i)) for i in order[::-1][:3]] + [("median", int(i)) for i in order[len(order) // 2:][:2]]
        for lab, i in picks:
            ax = int(np.argmax(np.abs(np.asarray(g[i]))))
            for eps in (4e-3, 2e-3):
                e = jnp.zeros_like(new).at[i, ax].set(eps)
                fd = (float(f(new + e, old, m_chk, kk)) - float(f(new - e, old, m_chk, kk))) / (2 * eps)
                log(f"{lab} gradient, point {i} axis {ax}, eps {eps * 1e3:.0f} mm: autodiff {float(g[i, ax]):+.4e}  finite diff {fd:+.4e}")
        log("grad check done")
        return
    if a.fd_check or a.bench:
        new = jnp.asarray(R_mm[groups[0]] * 1e-3, jnp.float32)
        old = jnp.zeros((0, 2), jnp.float32)
        kk = jax.random.fold_in(key, 0)
        t0 = time.perf_counter(); val, g = illum_vg(new, old, m, kk); g.block_until_ready()
        log(f"design value+grad: {time.perf_counter() - t0:.1f} s incl. compile, value {float(val):.5f}, "
            f"|g| median {float(jnp.median(jnp.linalg.norm(g, axis=1))):.3e}")
        if a.bench:
            t0 = time.perf_counter(); val, g = illum_vg(new, old, m, kk); g.block_until_ready()
            log(f"design value+grad (compiled, {a.design_impl}, {NS if a.src_batch else a.src_chunk} source(s) at a time, ckpt {ckpt}): "
                f"{time.perf_counter() - t0:.1f} s")
            try:
                log(f"peak device memory after design: {jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 2**30:.2f} GiB")
            except Exception as e:
                log(f"memory stats unavailable ({e})")
            if a.bench_ref:
                t0 = time.perf_counter(); vr, gr = illum_vg_remat(new, old, m, kk); gr.block_until_ready()
                log(f"remat reference: {time.perf_counter() - t0:.1f} s incl. compile; value {float(val):.6f} / {float(vr):.6f}, "
                    f"max |g - g_ref| / max |g_ref| {float(jnp.max(jnp.abs(g - gr)) / jnp.max(jnp.abs(gr))):.2e}")
            if a.bench_skip_fwi:
                return
            set_mask(np.isin(np.arange(len(R_mm)), groups[0]))
            vg = jax.jit(jax.value_and_grad(inv.loss, has_aux=True))
            (L, _), gm = vg(m, a.tv); t0 = time.perf_counter(); (L, _), gm = vg(m, a.tv); gm.block_until_ready()
            log(f"FWI update: {time.perf_counter() - t0:.1f} s")
            try:
                log(f"peak device memory: {jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 2**30:.2f} GiB")
            except Exception as e:  # CPU has no memory_stats
                log(f"memory stats unavailable ({e})")
            return
        vr, gr = illum_vg_ref(new, old, m, kk)
        log(f"split vs one-graph: value {float(val):.6f} / {float(vr):.6f}, max |g - g_ref| / max |g_ref| "
            f"{float(jnp.max(jnp.abs(g - gr)) / jnp.max(jnp.abs(gr))):.2e}")
        f = jax.jit(illumination)
        rng = np.random.default_rng(0)
        for _ in range(3):
            i, ax = int(rng.integers(len(new))), int(rng.integers(2))
            for eps in (2e-3, 1e-3):
                e = jnp.zeros_like(new).at[i, ax].set(eps)
                fd = (float(f(new + e, old, m, kk)) - float(f(new - e, old, m, kk))) / (2 * eps)
                log(f"point {i} axis {ax} eps {eps * 1e3:.0f} mm: autodiff {float(g[i, ax]):+.4e}  finite diff {fd:+.4e}")
        return
    if a.move_all:   # paper protocol: the whole layout moves; the outer Adam state is carried across rounds
        pos = jnp.asarray(R_mm[np.flatnonzero(np.load(PREP / "subsets" / f"{a.start_subset}.npz")["keep"] & usable_all)
                               if a.start_subset else np.concatenate(groups[:rounds])] * 1e-3, jnp.float32)
        opt_r = optax.adam(a.design_lr)
        st_r = opt_r.init(pos)
        ever = np.zeros(len(R_mm), bool)
    for k in range(rounds):
        entry = dict(round=k)
        if a.move_all:
            new, old = pos, jnp.zeros((0, 2), jnp.float32)
        else:
            new = jnp.asarray(R_mm[groups[k]] * 1e-3, jnp.float32)
            old = jnp.asarray(R_mm[measured] * 1e-3, jnp.float32).reshape(-1, 2)
        if a.select and k > 0:
            t0 = time.perf_counter()
            new = jnp.asarray(R_mm[select_points(m, measured, len(groups[k]), k)] * 1e-3, jnp.float32)
            entry["select_s"] = time.perf_counter() - t0
        start_pos = np.asarray(new)
        if a.random_moves_from and k > 0:
            rt = json.loads((REPO / "ldv_pmma_results" / "inversion" / a.random_moves_from / "settings.json").read_text())["trajectory"][k]
            lens = np.linalg.norm(np.array(rt["designed_mm"]) - np.array(rt["start_mm"]), axis=1) * 1e-3
            lens = np.resize(lens, len(start_pos)) if a.test else lens
            th = np.random.default_rng([a.seed, 77, k]).uniform(0, 2 * np.pi, len(lens))
            new = clip(new + jnp.asarray(np.c_[lens * np.cos(th), lens * np.sin(th)], jnp.float32))
            move = np.linalg.norm(np.asarray(new) - start_pos, axis=1) * 1e3
            entry.update(move_mm_median=float(np.median(move)), move_mm_max=float(move.max()))
        # (2) design
        if not a.no_design and not a.select and not a.random_moves_from and not (a.fwi_first and k == 0):
            t0 = time.perf_counter()
            md = jnp.zeros_like(m) if a.design_on_background else m     # model the design sees
            if not a.move_all:
                opt_r = optax.adam(a.design_lr)
                st_r = opt_r.init(new)
            vals = []
            if a.design_valid_every:
                f_v = jax.jit(illumination_fast)
                vk = [jax.random.PRNGKey(200000 + 10 * k + i) for i in range(4)]        # never used for the gradient
                valid_r = lambda r: float(np.mean([float(f_v(r, old, md, kv)) for kv in vk]))
                def snap_new(r):
                    t_ = []
                    for p_ in np.asarray(r) * 1e3:
                        t_.append(next(use_idx[j] for j in tree_all.query(p_, 16)[1] if not measured[use_idx[j]] and use_idx[j] not in t_))
                    return jnp.asarray(R_mm[t_] * 1e-3, jnp.float32)
                best_r = snap_new(new); best_v = valid_r(best_r); entry["valid_start"] = best_v
            for s in range(dsteps):
                kd = jax.random.fold_in(jax.random.fold_in(key, k), s) if a.design_fresh_probes else jax.random.fold_in(key, k)
                val, g = illum_vg(new, old, md, kd)   # default: probes fixed within a round
                if not bool(jnp.all(jnp.isfinite(g))):
                    log(f"round {k} design step {s}: non-finite gradient, design stopped for this round")
                    break
                if a.design_opt == "gdnorm":
                    gmax = jnp.max(jnp.linalg.norm(g, axis=1))
                    new = clip(new - a.design_move * g / jnp.maximum(gmax, 1e-30))
                else:
                    upd, st_r = opt_r.update(g, st_r)
                    new = clip(optax.apply_updates(new, upd))
                vals.append(float(val))
                if a.design_valid_every and ((s + 1) % a.design_valid_every == 0 or s + 1 == dsteps):
                    c_r = snap_new(new); v_ = valid_r(c_r)
                    log(f"round {k} design step {s + 1}: validation after snapping {v_:.4f} (best {min(v_, best_v):.4f})")
                    if v_ < best_v:
                        best_v, best_r = v_, c_r
            if a.design_valid_every:
                new = best_r; entry["valid_best"] = best_v
            move = np.linalg.norm(np.asarray(new) - start_pos, axis=1) * 1e3
            entry.update(design_objective=vals, design_s=time.perf_counter() - t0,
                         move_mm_median=float(np.median(move)), move_mm_max=float(move.max()))
        # (3) acquire: nearest scanned point not measured yet (no duplicates); move-all: not taken in this round
        if a.move_all:
            pos = new                                         # continuous positions carried to the next round
            measured = np.zeros(len(R_mm), bool)              # this round's FWI uses this round's points only
        taken = []
        for p in np.asarray(new) * 1e3:
            for j in tree_all.query(p, 16)[1]:
                gi = use_idx[j]
                if not measured[gi] and gi not in taken:
                    taken.append(gi)
                    break
        taken = np.array(taken)
        snap = np.linalg.norm(R_mm[taken] - np.asarray(new) * 1e3, axis=1)
        measured[taken] = True
        if a.move_all:
            entry.update(distinct_total=int((ever | measured).sum()),
                         remeasured=int((ever & measured).sum()))
            ever |= measured
        traj.append(dict(start_mm=(start_pos * 1e3).tolist(), designed_mm=(np.asarray(new) * 1e3).tolist(),
                         measured_idx=taken.tolist()))
        ns = set_mask(measured)
        entry.update(new_points=len(taken), measured_total=int(measured.sum()), nosurface=int(ns.sum()),
                     snap_mm_max=float(snap.max()))
        # (4) FWI on everything measured so far; model carried, Adam reset (paper protocol)
        vg = jax.jit(jax.value_and_grad(inv.loss, has_aux=True))
        opt = optax.adam(a.lr)
        st = opt.init(m)
        t0 = time.perf_counter()
        n_fwi = fwi_per_round + (max(0, a.fwi_total - rounds * fwi_per_round) if a.fwi_total and k == rounds - 1 else 0)
        for i in range(n_fwi):
            (L, (per, tv)), g = vg(m, a.tv)
            J = float(jnp.mean(per))
            if not np.isfinite(J):
                log(f"round {k} FWI {i}: misfit not finite, stop")
                break
            hist.append(J)
            upd, st = opt.update(g, st)
            m = optax.apply_updates(m, upd)
        h, _ = inv.fields(m)
        entry.update(misfit=hist[-1], fwi_s=time.perf_counter() - t0, min_h_mm=float(h.min()) * 1e3,
                     t=time.perf_counter() - t_start)
        rounds_log.append(entry)
        log(f"round {k}: +{entry['new_points']} -> {entry['measured_total']} points (no surface {entry['nosurface']}), "
            + (f"select {entry['select_s']:.0f} s, " if "select_s" in entry else "")
            + (f"random moves median {entry['move_mm_median']:.1f} / max {entry['move_mm_max']:.1f} mm, " if a.random_moves_from and "move_mm_median" in entry else "")
            + (f"design {entry['design_s']:.0f} s obj {entry['design_objective'][0]:.4f} -> {entry['design_objective'][-1]:.4f}, "
               f"move median {entry['move_mm_median']:.1f} / max {entry['move_mm_max']:.1f} mm, "
               + (f"validation {entry['valid_start']:.4f} -> {entry['valid_best']:.4f}, " if "valid_best" in entry else "") if "design_s" in entry else "")
            + (f"distinct so far {entry['distinct_total']} ({entry['remeasured']} re-measured), " if a.move_all else "")
            + f"snap max {entry['snap_mm_max']:.1f} mm | FWI {entry['fwi_s']:.0f} s misfit {entry['misfit']:.4f}, "
            f"min h {entry['min_h_mm']:.2f} mm | t {entry['t']:.0f} s")
        np.savez(out / "result.npz", m=np.asarray(m), Xc=inv.Xc, Yc=inv.Yc, iters_done=len(hist), hist=np.array(hist),
                 measured=measured)

    (PREP / "subsets").mkdir(exist_ok=True)
    np.savez(PREP / "subsets" / f"{a.tag}.npz", keep=measured, nosurface=ns,
             ever=ever if a.move_all else measured,
             rule="OED replay: measured points (move-all: last round; 'ever': all rounds); "
                  "no surface: subset rule of make_subsets.py")
    (out / "settings.json").write_text(json.dumps(dict(
        vars(a), param="log_thickness", hmax_mm=None, window_us=[it.T_WIN[0] * 1e6, it.T_WIN[1] * 1e6],
        taper_us=it.TAPER * 1e6, background_file=str(CALB), cube_model="none (unknown target)", s=inv.s, Q=inv.Q,
        grid_mm=cp.DX * 1e3, n_points=int(measured.sum()),
        active_files=list(inv.active), distinct_points=int((ever if a.move_all else measured).sum()), misfit_start=hist[0], misfit_end=hist[-1],
        rounds=rounds_log, trajectory=traj), indent=1))
    log(f"done {out}, total {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
