"""Reduced-scan layouts for the two-source study (CPU only, numpy): scan files 1 (S1) and 3 (S1 + S3). S1 is on in every scan,
so a true two-source set is S1 plus one other transducer; S3 is the one farthest from S1 (338 mm).
Same constructions as make_subsets_fair.py, but usable points and "no surface" points come from files 1 and 3 only.

  s2rc30             cell-centred regular 6 x 5
  s2j30a, s2j30b     jittered 6 x 5, seeds 1, 2
  s2x48a ... s2x48d  random 31 points, seeds 1-4

Usage:  python src/ldv_pmma/make_subsets_s2.py
Output: ldv_work/prepared/subsets/s2*.npz, subsets_s2.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import make_subsets as ms

ms.KS = (1, 3)


def main():
    d = np.load(ms.PREP / "scatter.npz")
    R = d["R"]
    usable = np.flatnonzero(np.any([~d[f"bad{k}"] for k in ms.KS], 0))
    lo, hi = R[usable].min(0), R[usable].max(0)
    tree = cKDTree(R[usable])

    def mask(idx):
        keep = np.zeros(len(R), bool); keep[idx] = True
        return keep

    cen = [lo + (np.array([i, j]) + 0.5) * (hi - lo) / [6, 5] for i in range(6) for j in range(5)]
    sets = {"s2rc30": mask(usable[tree.query(cen)[1]])}
    cell = np.minimum(((R[usable] - lo) / (hi - lo) * [6, 5]).astype(int), [5, 4])
    for tag, seed in (("a", 1), ("b", 2)):
        rng = np.random.default_rng(seed)
        sets[f"s2j30{tag}"] = mask([rng.choice(usable[(cell[:, 0] == i) & (cell[:, 1] == j)]) for i in range(6) for j in range(5)])
    for tag, seed in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
        sets[f"s2x48{tag}"] = mask(np.random.default_rng(seed).choice(usable, 31, replace=False))
    info = {}
    for name, keep in sets.items():
        ns = ms.nosurface_subset(d, keep)
        np.savez(ms.OUT / f"{name}.npz", keep=keep, nosurface=ns,
                 rule=f"scan files 1, 3 (sources S1, S3); no surface: rms > {ms.RATIO} x median of {ms.KNN} nearest subset neighbours")
        info[name] = dict(points=int(keep.sum()), nosurface=int(ns.sum()))
        print(name, info[name])
    (ms.OUT / "subsets_s2.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
