"""Additional reduced-scan layouts (CPU only, numpy); same files and keys as make_subsets.py, existing subsets untouched.

  rc30, rc48   regular, cell-centred: 6 x 5 and 8 x 6 cells over the scan, cell centres snapped to the nearest usable
               point (no points wasted on the scan edge, unlike the corner-anchored r<s> grids)
  j30a, j30b   jittered: the same 6 x 5 cells, one uniformly random usable point per cell (seeds 1, 2)
  x48c, x48d   random: 31 usable points, seeds 3, 4 (same draw as x48a/b, which use seeds 1, 2)

Usage:  python src/ldv_pmma/make_subsets_fair.py
Output: ldv_work/prepared/subsets/<id>.npz, subsets_fair.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

from make_subsets import KNN, KS, OUT, PREP, RATIO, nosurface_subset


def main():
    d = np.load(PREP / "scatter.npz")
    R = d["R"]
    usable = np.flatnonzero(np.any([~d[f"bad{k}"] for k in KS], 0))
    lo, hi = R[usable].min(0), R[usable].max(0)
    tree = cKDTree(R[usable])
    sets = {}

    def mask(idx):
        keep = np.zeros(len(R), bool); keep[idx] = True
        return keep

    for name, (nx, ny) in (("rc30", (6, 5)), ("rc48", (8, 6))):
        cen = [lo + (np.array([i, j]) + 0.5) * (hi - lo) / [nx, ny] for i in range(nx) for j in range(ny)]
        sets[name] = mask(usable[tree.query(cen)[1]])
    cell = np.minimum(((R[usable] - lo) / (hi - lo) * [6, 5]).astype(int), [5, 4])
    for tag, seed in (("a", 1), ("b", 2)):
        rng = np.random.default_rng(seed)
        sets[f"j30{tag}"] = mask([rng.choice(usable[(cell[:, 0] == i) & (cell[:, 1] == j)]) for i in range(6) for j in range(5)])
    for tag, seed in (("c", 3), ("d", 4)):
        sets[f"x48{tag}"] = mask(np.random.default_rng(seed).choice(usable, 31, replace=False))
    info = {}
    for name, keep in sets.items():
        assert not (OUT / f"{name}.npz").exists(), name
        ns = nosurface_subset(d, keep)
        np.savez(OUT / f"{name}.npz", keep=keep, nosurface=ns,
                 rule=f"no surface: rms > {RATIO} x median of {KNN} nearest subset neighbours")
        info[name] = dict(points=int(keep.sum()), nosurface=int(ns.sum()))
        print(name, info[name])
    (OUT / "subsets_fair.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
