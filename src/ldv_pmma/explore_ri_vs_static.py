import sys, numpy as np
sys.path.insert(0, ".")
exec(open("explore_views.py").read().split("for k, g in G.items():")[0])
from scipy.stats import binomtest, mannwhitneyu, fisher_exact
def cf(t, q):
    h, roi, bg, r = maps(t); a = np.abs(h - 10); thr = np.quantile(a[bg], q)
    return int(sum(a[r[n] < 8].max() > thr for n in FEAT))
v3 = lambda t: S[t]["found"]
RIo = {4: ["ri_a30"]+[f"ri_q4a_j30{s}" for s in "abcd"], 2: ["ri_s2a30"]+[f"ri_q2a_j30{s}" for s in "abcd"], 1: [f"ri_q1a_{s}" for s in ("rc30","j30a","j30b","j30c")]}
RIl = {4: [f"ri_q4l_j30{s}" for s in "abcd"], 2: [f"ri_q2l_j30{s}" for s in "abcd"], 1: [f"ri_q1l_{s}" for s in ("rc30","j30a","j30b","j30c")]}
SAME = {"ri_a30":"rc30","ri_q4a_j30a":"j30a","ri_q4a_j30b":"j30b","ri_q4l_j30a":"j30a","ri_q4l_j30b":"j30b",
        "ri_s2a30":"s2rc30","ri_q2a_j30a":"s2j30a","ri_q2a_j30b":"s2j30b","ri_q2l_j30a":"s2j30a","ri_q2l_j30b":"s2j30b",
        **{f"ri_q1a_{s}":f"q1s_{s}" for s in ("rc30","j30a","j30b","j30c")}, **{f"ri_q1l_{s}":f"q1s_{s}" for s in ("rc30","j30a","j30b","j30c")}}
rules = [("v3 (3 mm)", v3), ("rate 5 %", lambda t: cf(t, .95)), ("rate 1 %", lambda t: cf(t, .99))]
for name, RI in (("old criterion", RIo), ("log criterion", RIl), ("both", {k: RIo[k]+RIl[k] for k in RIo})):
    print(f"\n### re-inverted adaptive points, {name}")
    for rn, f in rules:
        row = []; allr = []; alls = []
        for k in (4, 2, 1):
            r = [f(t) for t in RI[k]]; s = [f(t) for t in STATIC30[k]]; allr += [x - np.mean(s) for x in r]; alls += [x - np.mean(s) for x in s]
            row.append(f"{k}sh: {np.mean(r):.2f} vs {np.mean(s):.2f} (all5 {sum(x==5 for x in r)}/{len(r)} vs {sum(x==5 for x in s)}/{len(s)})")
        p = mannwhitneyu(allr, alls, alternative="greater").pvalue
        pr = [(t, SAME[t]) for k in RI for t in RI[k] if t in SAME]
        d = [f(a) - f(b) for a, b in pr]; w, e, l = sum(x>0 for x in d), sum(x==0 for x in d), sum(x<0 for x in d)
        ps = binomtest(w, w+l, .5, alternative="greater").pvalue if w+l else 1
        print(f"  {rn:9} | " + " | ".join(row) + f" | unpaired (centred per shot) MWU p={p:.3f} | same start {w}/{e}/{l} p={ps:.2f}")
