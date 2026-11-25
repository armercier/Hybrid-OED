"""Reduced-scan point subsets of the defect scan (CPU only, numpy), for the layout comparison.

Each subset is a boolean mask over all 6418 points of ldv_work/prepared/scatter.npz:
  r<s>   regular: every (s/3)-th point of the 3 mm scan grid in u and v (s = 6, 9, 12, 18, 24, 36, 48 mm), offset 0
  x<s>a/b random: the same number of USABLE points as r<s>, drawn uniformly from the usable points (seeds 1, 2)
Usable = recorded without dropout in at least one file (same rule as Calib.use).
The blind "no surface" check is redone on each subset with only that subset's points: amplitude (vz RMS, 500-800 us)
> 1.5 x the median of its 8 nearest subset neighbours (median over files). The full-scan rule (neighbours within 12 mm)
has too few neighbours at 9-12 mm spacing, so the nearest-neighbour version is used for every subset.

Usage:  python src/ldv_pmma/make_subsets.py
Output: ldv_work/prepared/subsets/<id>.npz (keep, nosurface, rule), subsets.json (counts)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
PREP = REPO / "ldv_work" / "prepared"
OUT = PREP / "subsets"
KS = (1, 2, 3, 4)
STEP, RATIO, KNN = 3.0, 1.5, 8


def nosurface_subset(d, keep):
    idx = np.flatnonzero(keep)
    w = (d["t"] > 500e-6) & (d["t"] < 800e-6)
    bad = np.any([d[f"bad{k}"][idx] for k in KS], 0)
    rms = np.stack([np.sqrt(np.mean(d[f"vz{k}"][idx][:, w] ** 2, 1)) for k in KS])
    good = np.flatnonzero(~bad)
    tree = cKDTree(d["R"][idx][good])
    _, nn = tree.query(d["R"][idx], KNN + 1)
    ns = np.zeros(len(d["R"]), bool)
    for i, row in enumerate(nn):
        n = [good[j] for j in row if good[j] != i][:KNN]
        q = np.median(rms[:, i] / np.median(rms[:, n], 1))
        ns[idx[i]] = (q > RATIO) and not bad[i]
    return ns


def main():
    d = np.load(PREP / "scatter.npz")
    R = d["R"]
    usable = np.any([~d[f"bad{k}"] for k in KS], 0)
    ij = np.round((R - R.min(0)) / STEP).astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    for s in (6, 9, 12, 18, 24, 36, 48):
        q = s // 3
        reg = (ij[:, 0] % q == 0) & (ij[:, 1] % q == 0)
        n = int((reg & usable).sum())
        sets = {f"r{s}": reg}
        for tag, seed in (("a", 1), ("b", 2)):
            keep = np.zeros(len(R), bool)
            keep[np.random.default_rng(seed).choice(np.flatnonzero(usable), n, replace=False)] = True
            sets[f"x{s}{tag}"] = keep
        for name, keep in sets.items():
            ns = nosurface_subset(d, keep)
            np.savez(OUT / f"{name}.npz", keep=keep, nosurface=ns,
                     rule=f"no surface: rms > {RATIO} x median of {KNN} nearest subset neighbours")
            info[name] = dict(points=int(keep.sum()), usable=int((keep & usable).sum()), nosurface=int(ns.sum()))
            print(name, info[name])
    (OUT / "subsets.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
