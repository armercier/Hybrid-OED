"""Exploratory (23 Nov): relaxed case, equal inversion cost only. Moving array (30 sensors, 5 rounds, every sensor moved each
round, 20 FWI updates per round = 100 in total, final 30 points; ~145 positions measured) vs static 30-point layouts (100
updates). Gradient moves (ad_/a2_/a1_) and, as reference, random moves (rn_/r2_/r1_).
Unpaired: Mann-Whitney on found counts centred on the static mean per shot (static = STATIC30); paired: the static run of the
same start layout (k1* at 1 shot, the earlier 1-shot statics the a1_/r1_ runs started from). Also median v3 false indications."""
import sys, numpy as np
sys.path.insert(0, ".")
exec(open("explore_views.py").read().split("for k, g in G.items():")[0])
from scipy.stats import binomtest, mannwhitneyu
def cf(t, q):
    h, roi, bg, r = maps(t); a = np.abs(h - 10); thr = np.quantile(a[bg], q)
    return int(sum(a[r[n] < 8].max() > thr for n in FEAT))
ST = {4: ("rc30", "j30a", "x48c"), 2: ("rc30", "j30a", "j30b", "x48a", "x48c"), 1: ("rc30", "j30a", "j30b", "x48a", "x48c")}
PF = {4: ("ad", "rn", ""), 2: ("a2", "r2", "s2"), 1: ("a1", "r1", "k1")}
rules = [("rate 1 %", lambda t: cf(t, .99)), ("rate 5 %", lambda t: cf(t, .95)), ("v3 (3 mm)", lambda t: S[t]["found"])]
for which, name in ((0, "gradient moves"), (1, "random moves")):
    print(f"\n### moving array, {name}, vs static 30 points")
    for rn, f in rules:
        row, ca, cs, d = [], [], [], []
        for k in (4, 2, 1):
            mv = [f"{PF[k][which]}_{s}" for s in ST[k]]; same = [f"{PF[k][2]}{s}" for s in ST[k]]
            a = [f(t) for t in mv]; s_ = [f(t) for t in STATIC30[k]]; m = np.mean(s_)
            ca += [x - m for x in a]; cs += [x - m for x in s_]; d += [f(x) - f(y) for x, y in zip(mv, same)]
            row.append(f"{k}sh {np.mean(a):.2f} vs {m:.2f} (all5 {sum(x==5 for x in a)}/{len(a)} vs {sum(x==5 for x in s_)}/{len(s_)})")
        w, e, l = sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)
        print(f"  {rn:9} | " + " | ".join(row) + f" | MWU p={mannwhitneyu(ca, cs, alternative='greater').pvalue:.3f}"
              f" | paired {w}/{e}/{l} p={binomtest(w, w+l, .5, alternative='greater').pvalue if w+l else 1:.2f}")
    fi = [f"{k}sh {int(np.median([S[f'{PF[k][which]}_{s}']['false_alarm'] for s in ST[k]]))} vs {int(np.median([S[t]['false_alarm'] for t in STATIC30[k]]))}" for k in (4, 2, 1)]
    print("  v3 false indications (median): " + " | ".join(fi))
