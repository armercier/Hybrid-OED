import json, numpy as np, sys
sys.path.insert(0, "src/ldv_pmma")
from score_inversions import thickness, PREP
geo = json.loads((PREP/"geometry.json").read_text()); R = np.load(PREP/"scatter.npz")["R"]; O = 0.5*(R.min(0)+R.max(0))
D = {n: (d["x"]-O[0], d["y"]-O[1]) for n, d in geo["scatter"]["defects"].items()}
m1 = np.array(D["M1"]); Ru = R - O
ref = thickness("v3")[0]
print("tag      falseNearM1(20-50mm) falseElse  maxdev@M1<25  RMSv3  RMSv3_noM1(>40mm)  measured<30mm of M1 (share)")
for t in ["v3","s2full","rc30","ad_rc30","rn_rc30","j30a","ad_j30a","rn_j30a","x48c","ad_x48c","rn_x48c"]:
    h, Xc, Yc, st = thickness(t); X, Y = Xc*1e3-O[0], Yc*1e3-O[1]
    inner = (np.abs(X)<140)&(np.abs(Y)<90)
    dist = np.min([np.hypot(X-u, Y-v) for u, v in D.values()], 0); dm = np.hypot(X-m1[0], Y-m1[1])
    fc = inner&(dist>20)&(h<9)
    near = fc&(dm<50)
    r_all = np.sqrt(np.mean((h-ref)[inner]**2)); r_no = np.sqrt(np.mean((h-ref)[inner&(dm>40)]**2))
    sh = ""
    if "trajectory" in st:
        idx = np.unique(np.concatenate([tr["measured_idx"] for tr in st["trajectory"]]))
        sh = f"{np.mean(np.hypot(*(Ru[idx]-m1).T)<30):.2f} of {len(idx)}"
    print(f"{t:8s} {near.sum():8d} {(fc&~near).sum():18d} {np.abs(h-10)[dm<25].max():10.2f} {r_all:9.2f} {r_no:10.2f}       {sh}")
# share of all usable scan points within 30 mm of M1 (random expectation)
print("all scan points within 30 mm of M1:", f"{np.mean(np.hypot(*(Ru-m1).T)<30):.2f}")
