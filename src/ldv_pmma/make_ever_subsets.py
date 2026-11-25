"""Subsets for proposal A (23 Nov): all positions visited by a moving-array run, inverted jointly like a static layout.

The move-all runs (Q2: gradient ad_/a2_/a1_, random rn_/r2_/r1_) measure 30 positions per round for 5 rounds, but each
round's inversion only sees the current 30. Here every distinct visited position ('ever' in the run's subset file) is
inverted at once (100 updates from the blind background), and compared with a static jittered layout with exactly the
same number of points (one per start, matched to the gradient run). Question: at ~145 measured points, are the positions a
gradient-moved array walks through better than random-walked or regular ones?

Tags:  ev_<run>                 all visited positions of <run>
       mj<shots>_<start>        static jittered layout, n = distinct positions of the gradient run from <start>
No-surface rule as for the static layouts (make_subsets.nosurface_subset), the same for all three.
Usage:  python src/ldv_pmma/make_ever_subsets.py              (proposal A: ev_ + mj subsets)
        python src/ldv_pmma/make_ever_subsets.py RUN [RUN ...]  (only ev_<RUN>, files from the run's settings.json)
Output: ldv_work/prepared/subsets/{ev_*,mj*}.npz, subsets_ever.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import make_subsets as ms

PLAN = {(1, 2, 3, 4): ("ad", "rn", 4, ("rc30", "j30a", "x48c")),
        (1, 3): ("a2", "r2", 2, ("rc30", "j30a", "j30b", "x48a", "x48c"))}


def jitter(d, ks, n, seed):
    """One point per cell of the smallest nx x ny grid (aspect 6:5) holding n cells; surplus cells dropped at random."""
    R = d["R"]
    usable = np.flatnonzero(np.any([~d[f"bad{k}"] for k in ks], 0))
    lo, hi = R[usable].min(0), R[usable].max(0)
    ny = int(np.ceil(np.sqrt(n * 5 / 6))); nx = int(np.ceil(n / ny))
    cell = np.minimum(((R[usable] - lo) / (hi - lo) * [nx, ny]).astype(int), [nx - 1, ny - 1])
    rng = np.random.default_rng(seed)
    cells = [(i, j) for i in range(nx) for j in range(ny) if ((cell[:, 0] == i) & (cell[:, 1] == j)).any()]
    cells = [cells[c] for c in sorted(rng.choice(len(cells), n, replace=False))]
    keep = np.zeros(len(R), bool)
    keep[[rng.choice(usable[(cell[:, 0] == i) & (cell[:, 1] == j)]) for i, j in cells]] = True
    return keep, (nx, ny)


def main():
    d = np.load(ms.PREP / "scatter.npz")
    if len(sys.argv) > 1:
        for run in sys.argv[1:]:
            ms.KS = tuple(json.loads((ms.REPO / "ldv_pmma_results" / "inversion" / run / "settings.json").read_text())["active_files"])
            keep = np.load(ms.OUT / f"{run}.npz")["ever"].astype(bool)
            np.savez(ms.OUT / f"ev_{run}.npz", keep=keep, nosurface=ms.nosurface_subset(d, keep),
                     rule=f"all positions visited by moving run {run}; scan files {ms.KS}")
            print(f"ev_{run}", int(keep.sum()), ms.KS)
        return
    info = {}
    for ks, (g, r, shots, starts) in PLAN.items():
        ms.KS = ks
        for k, s in enumerate(starts):
            for run in (f"{g}_{s}", f"{r}_{s}"):
                keep = np.load(ms.OUT / f"{run}.npz")["ever"].astype(bool)
                np.savez(ms.OUT / f"ev_{run}.npz", keep=keep, nosurface=ms.nosurface_subset(d, keep),
                         rule=f"all positions visited by moving run {run}; scan files {ks}")
                info[f"ev_{run}"] = dict(files=list(ks), points=int(keep.sum()))
            n = info[f"ev_{g}_{s}"]["points"]
            keep, grid = jitter(d, ks, n, seed=100 + k)
            assert keep.sum() == n
            np.savez(ms.OUT / f"mj{shots}_{s}.npz", keep=keep, nosurface=ms.nosurface_subset(d, keep),
                     rule=f"jittered {grid[0]} x {grid[1]} grid, {n} cells kept (matched to {g}_{s}); scan files {ks}")
            info[f"mj{shots}_{s}"] = dict(files=list(ks), points=n, grid=grid)
    for t, v in info.items():
        print(t, v)
    (ms.OUT / "subsets_ever.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
