import sys, json, numpy as np
sys.path.insert(0, ".")
exec(open("explore_views.py").read().split("for k, g in G.items():")[0])
from scipy.stats import binomtest
def cf(t, q, blind):
    h, roi, bg, r = maps(t); a = np.abs(h - 10)
    thr = np.quantile(a[roi if blind else bg], q)
    return int(sum(a[r[n] < 8].max() > thr for n in FEAT))
P = {4: [("a30","n30","rc30")]+[(f"q4a_j30{s}",f"q4n_j30{s}",{"a":"j30a","b":"j30b"}.get(s)) for s in "abcd"],
     2: [("s2a30","s2n30","s2rc30")]+[(f"q2a_j30{s}",f"q2n_j30{s}",{"a":"s2j30a","b":"s2j30b"}.get(s)) for s in "abcd"],
     1: [(f"q1a_{s}",f"q1n_{s}",f"q1s_{s}") for s in ("rc30","j30a","j30b","j30c")]}
for blind in (False, True):
  for q in (0.95, 0.98, 0.99, 0.995):
    out=[]
    for lab, f in (("ri>prog", lambda a,n,s: ("ri_"+a, n)), ("ri>same-start", lambda a,n,s: ("ri_"+a, s)), ("adapt>prog", lambda a,n,s: (a, n))):
        w=e=l=0
        for k in P:
            for a,n,s in P[k]:
                x,y=f(a,n,s)
                if y is None: continue
                d=cf(x,q,blind)-cf(y,q,blind); w+=d>0; e+=d==0; l+=d<0
        out.append(f"{lab} {w}/{e}/{l} p={binomtest(w,w+l,.5,alternative='greater').pvalue if w+l else 1:.3f}")
    mg=[cf(t,q,blind) for k in (4,2,1) for t in G[k]["move_grad"]]; mr=[cf(t,q,blind) for k in (4,2,1) for t in G[k]["move_rand"]]
    print(f"{'blind (whole ROI)' if blind else 'feature-excluded'} q={q}: " + " | ".join(out) + f" | moves all5 grad {sum(x==5 for x in mg)}/13 rand {sum(x==5 for x in mr)}/13")
