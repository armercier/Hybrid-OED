"""Exploratory (23 Nov): adaptive design as run vs progressive static control, same scan cost (30 points, 3 groups of 10) and
same inversion (33/33/34 updates). Found counts under v3 and at fixed false-indication rates (feature-excluded background).
Unpaired: Mann-Whitney on counts centred on the progressive-static mean of each shot setting; paired: same start layout."""
import sys, numpy as np
sys.path.insert(0, ".")
exec(open("explore_views.py").read().split("for k, g in G.items():")[0])
from scipy.stats import binomtest, mannwhitneyu
def cf(t, q):
    h, roi, bg, r = maps(t); a = np.abs(h - 10); thr = np.quantile(a[bg], q)
    return int(sum(a[r[n] < 8].max() > thr for n in FEAT))
J = [f"j30{s}" for s in "abcd"]; Q1 = ("rc30", "j30a", "j30b", "j30c")
AD = {4: ["a30"] + [f"q4a_{s}" for s in J], 2: ["s2a30"] + [f"q2a_{s}" for s in J], 1: [f"q1a_{s}" for s in Q1]}
LG = {4: [None] + [f"q4l_{s}" for s in J], 2: [None] + [f"q2l_{s}" for s in J], 1: [f"q1l_{s}" for s in Q1]}
PR = {4: ["n30"] + [f"q4n_{s}" for s in J], 2: ["s2n30"] + [f"q2n_{s}" for s in J], 1: [f"q1n_{s}" for s in Q1]}
rules = [("rate 1 %", lambda t: cf(t, .99)), ("rate 5 %", lambda t: cf(t, .95)), ("v3 (3 mm)", lambda t: S[t]["found"])]
for name, A in (("old criterion", AD), ("log criterion", LG)):
    print(f"\n### adaptive as run, {name}, vs progressive static")
    for rn, f in rules:
        row, ca, cp, d = [], [], [], []
        for k in (4, 2, 1):
            pairs = [(a, p) for a, p in zip(A[k], PR[k]) if a]
            a = [f(x) for x, _ in pairs]; p = [f(t) for t in PR[k]]; m = np.mean(p)
            ca += [x - m for x in a]; cp += [x - m for x in p]; d += [f(x) - f(y) for x, y in pairs]
            row.append(f"{k}sh {np.mean(a):.2f} vs {m:.2f} (all5 {sum(x==5 for x in a)}/{len(a)} vs {sum(x==5 for x in p)}/{len(p)})")
        w, e, l = sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)
        print(f"  {rn:9} | " + " | ".join(row) + f" | MWU p={mannwhitneyu(ca, cp, alternative='greater').pvalue:.3f}"
              f" | paired {w}/{e}/{l} p={binomtest(w, w+l, .5, alternative='greater').pvalue if w+l else 1:.2f}")
