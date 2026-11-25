"""Regular reference layouts for proposal A (23 Nov): cell-centred 13 x 11 grid (143 points) over the usable scan area,
nearest scan point to each cell centre; same construction as the 6 x 5 rc30 layouts (make_subsets_q1.build, 'regular').
rg4_143: scan files 1-4;  rg2_143: files 1, 3. Within 10 points of every ev_/mj run of proposal A (138-153).
Usage:  python src/ldv_pmma/make_regular143.py      Output: ldv_work/prepared/subsets/rg{4,2}_143.npz
"""
import numpy as np
from scipy.spatial import cKDTree

import make_subsets as ms

d = np.load(ms.PREP / "scatter.npz"); R = d["R"]
for name, ks in (("rg4_143", (1, 2, 3, 4)), ("rg2_143", (1, 3))):
    ms.KS = ks
    usable = np.flatnonzero(np.any([~d[f"bad{k}"] for k in ks], 0))
    lo, hi = R[usable].min(0), R[usable].max(0)
    cen = [lo + (np.array([i, j]) + 0.5) * (hi - lo) / [13, 11] for i in range(13) for j in range(11)]
    keep = np.zeros(len(R), bool); keep[usable[cKDTree(R[usable]).query(cen)[1]]] = True
    assert not (ms.OUT / f"{name}.npz").exists(), name
    np.savez(ms.OUT / f"{name}.npz", keep=keep, nosurface=ms.nosurface_subset(d, keep),
             rule=f"cell-centred regular 13 x 11 grid; scan files {ks}")
    print(name, int(keep.sum()))
