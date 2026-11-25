"""Layouts for the same-budget study (Q1, 30 measured points), CPU only; same constructions as make_subsets_fair.py and
make_subsets_s2.py, existing layouts untouched.
  4 shots (all files):   j30c, j30d        jittered 6 x 5, seeds 5, 6
  2 shots (files 1, 3):  s2j30c, s2j30d    jittered 6 x 5, seeds 5, 6
  1 shot (file 3 only; every point valid in file 3, so 30 measured points):
                         f3rc30            cell-centred regular 6 x 5
                         f3j30a ... f3j30d jittered 6 x 5, seeds 1, 2, 5, 6
                         f3x30a, f3x30b    random 30 points, seeds 1, 2
Usage:  python src/ldv_pmma/make_subsets_q1.py
Output: ldv_work/prepared/subsets/<name>.npz, subsets_q1.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import make_subsets as ms


def build(d, ks, specs):
    ms.KS = ks
    R = d["R"]
    usable = np.flatnonzero(np.any([~d[f"bad{k}"] for k in ks], 0))
    lo, hi = R[usable].min(0), R[usable].max(0)
    cell = np.minimum(((R[usable] - lo) / (hi - lo) * [6, 5]).astype(int), [5, 4])

    def mask(idx):
        keep = np.zeros(len(R), bool); keep[idx] = True
        return keep

    out = {}
    for name, kind, seed in specs:
        if kind == "regular":
            cen = [lo + (np.array([i, j]) + 0.5) * (hi - lo) / [6, 5] for i in range(6) for j in range(5)]
            out[name] = mask(usable[cKDTree(R[usable]).query(cen)[1]])
        elif kind == "jitter":
            rng = np.random.default_rng(seed)
            out[name] = mask([rng.choice(usable[(cell[:, 0] == i) & (cell[:, 1] == j)]) for i in range(6) for j in range(5)])
        else:
            out[name] = mask(np.random.default_rng(seed).choice(usable, 30, replace=False))
    return out


def main():
    d = np.load(ms.PREP / "scatter.npz")
    plan = [((1, 2, 3, 4), [("j30c", "jitter", 5), ("j30d", "jitter", 6)]),
            ((1, 3), [("s2j30c", "jitter", 5), ("s2j30d", "jitter", 6)]),
            ((3,), [("f3rc30", "regular", None), ("f3j30a", "jitter", 1), ("f3j30b", "jitter", 2), ("f3j30c", "jitter", 5),
                    ("f3j30d", "jitter", 6), ("f3x30a", "random", 1), ("f3x30b", "random", 2)])]
    info = {}
    for ks, specs in plan:
        for name, keep in build(d, ks, specs).items():
            assert not (ms.OUT / f"{name}.npz").exists(), name
            ms.KS = ks
            ns = ms.nosurface_subset(d, keep)
            np.savez(ms.OUT / f"{name}.npz", keep=keep, nosurface=ns,
                     rule=f"scan files {ks}; no surface: rms > {ms.RATIO} x median of {ms.KNN} nearest subset neighbours")
            valid = int((keep & np.any([~d[f"bad{k}"] for k in ks], 0)).sum())
            info[name] = dict(files=list(ks), points=int(keep.sum()), valid=valid, nosurface=int(ns.sum()))
            print(name, info[name])
    (ms.OUT / "subsets_q1.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
