import json, sys, numpy as np
sys.path.insert(0, "/home/merciera/git/Hybrid-OED/src/ldv_pmma")
from score_inversions import INV, PREP, thickness
from run_overview import STATIC30
S = json.load(open(INV / "scores_v3.json"))
geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
R = np.load(PREP / "scatter.npz")["R"]; O = 0.5 * (R.min(0) + R.max(0))
FEAT = ("D1","D2","D3","D4","M1")
REF = {4: "v3", 2: "s2full", 1: "k1full"}
def maps(t):
    h, Xc, Yc, _ = thickness(t); X, Y = Xc*1e3 - O[0], Yc*1e3 - O[1]
    r = {n: np.hypot(X - d["x"] + O[0], Y - d["y"] + O[1]) for n, d in geo.items()}
    roi = (np.abs(X) < 140) & (np.abs(Y) < 90)
    bg = roi & (np.min([r[n] for n in FEAT], 0) > 20)
    return h, roi, bg, r
refs = {k: maps(v)[0] for k, v in REF.items()}
def view(t, k):
    h, roi, bg, r = maps(t); a = np.abs(h - 10)
    peaks = np.array([a[r[n] < 8].max() for n in FEAT])
    q95 = np.quantile(a[bg], 0.95)
    rmse = np.sqrt(np.mean((h - refs[k])[roi] ** 2))
    cc = np.corrcoef(a[roi], np.abs(refs[k] - 10)[roi])[0, 1]
    # AUC: feature peaks vs background cells
    auc = np.mean([(a[bg] < p).mean() for p in peaks])
    return dict(rmse=rmse, cc=cc, minpk_q95=peaks.min() / q95, auc=auc, D4=peaks[3], q95=q95)
G = {4: dict(static=STATIC30[4], adaptive=["a30"]+[f"q4a_j30{s}" for s in "abcd"], log=[f"q4l_j30{s}" for s in "abcd"],
             prog=["n30"]+[f"q4n_j30{s}" for s in "abcd"], ri=["ri_a30"]+[f"ri_q4a_j30{s}" for s in "abcd"], sim=["d30a","d30b","d30c"],
             move_grad=["ad_rc30","ad_j30a","ad_x48c"], move_rand=["rn_rc30","rn_j30a","rn_x48c"], static110=["r24","x24a"], static47=["x36a","x36b","r36","rc48"]),
     2: dict(static=STATIC30[2], adaptive=["s2a30"]+[f"q2a_j30{s}" for s in "abcd"], log=[f"q2l_j30{s}" for s in "abcd"],
             prog=["s2n30"]+[f"q2n_j30{s}" for s in "abcd"], ri=["ri_s2a30"]+[f"ri_q2a_j30{s}" for s in "abcd"],
             move_grad=[f"a2_{s}" for s in ("rc30","j30a","j30b","x48a","x48c")], move_rand=[f"r2_{s}" for s in ("rc30","j30a","j30b","x48a","x48c")]),
     1: dict(static=STATIC30[1], adaptive=[f"q1a_{s}" for s in ("rc30","j30a","j30b","j30c")], log=[f"q1l_{s}" for s in ("rc30","j30a","j30b","j30c")],
             prog=[f"q1n_{s}" for s in ("rc30","j30a","j30b","j30c")], ri=[f"ri_q1a_{s}" for s in ("rc30","j30a","j30b","j30c")],
             move_grad=[f"a1_{s}" for s in ("rc30","j30a","j30b","x48a","x48c")], move_rand=[f"r1_{s}" for s in ("rc30","j30a","j30b","x48a","x48c")])}
for k, g in G.items():
    print(f"\n== {k} shots   (median over runs)        RMSE vs full scan | corr with full-scan anomaly | min peak / bg q95 | AUC | D4 peak")
    for name, tags in g.items():
        V = [view(t, k) for t in tags if (INV / t / "result.npz").exists()]
        m = {q: np.median([v[q] for v in V]) for q in V[0]}
        print(f"  {name:10} n={len(V):2}  rmse {m['rmse']:.2f}  cc {m['cc']:.2f}  minpk/q95 {m['minpk_q95']:.2f}  auc {m['auc']:.3f}  D4 {m['D4']:.1f}")

print("\n\n#### fixed false-indication rate: found = peak > q-quantile of the map's own background")
from scipy.stats import binomtest, fisher_exact
def cfar(t, q):
    h, roi, bg, r = maps(t); a = np.abs(h - 10); thr = np.quantile(a[bg], q)
    return int(sum(a[r[n] < 8].max() > thr for n in FEAT))
PAIRS = {4: [("a30","n30","rc30")]+[(f"q4a_j30{s}",f"q4n_j30{s}",{"a":"j30a","b":"j30b"}.get(s)) for s in "abcd"],
         2: [("s2a30","s2n30","s2rc30")]+[(f"q2a_j30{s}",f"q2n_j30{s}",{"a":"s2j30a","b":"s2j30b"}.get(s)) for s in "abcd"],
         1: [(f"q1a_{s}",f"q1n_{s}",f"q1s_{s}") for s in ("rc30","j30a","j30b","j30c")]}
for q in (0.95, 0.99):
    print(f"\n-- threshold = background quantile {q}  (all-five found / mean found)")
    tot = {}
    for k, g in G.items():
        line = []
        for name in ("static","adaptive","log","prog","ri","sim","move_grad","move_rand"):
            if name not in g: continue
            f = [cfar(t, q) for t in g[name]]
            line.append(f"{name} {sum(x==5 for x in f)}/{len(f)} {np.mean(f):.2f}")
        print(f"  {k} shots: " + " | ".join(line))
        for a_, n_, s_ in PAIRS[k]:
            for lab, x, y in (("adaptive>prog", a_, n_), ("ri>prog", "ri_"+a_, n_), ("ri>same-start static", "ri_"+a_, s_), ("adaptive>same-start static", a_, s_)):
                if y is None: continue
                d = cfar(x, q) - cfar(y, q); tot.setdefault(lab, [0,0,0]); tot[lab][0 if d>0 else 1 if d==0 else 2] += 1
    for lab, (w, e, l) in tot.items():
        print(f"  pooled {lab:28} more {w}, same {e}, fewer {l}  sign p = {binomtest(w, w+l, .5, alternative='greater').pvalue if w+l else float('nan'):.3f}")
