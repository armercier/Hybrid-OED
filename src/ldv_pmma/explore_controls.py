"""Exploratory (24 Nov): the design controls. Case (i): gradient vs offline moves (designed on the background) vs random
moves vs static 30; case (ii): simulation design vs static 30 and vs re-inverted adaptive; case (iv): all visited positions
(gradient / random / offline) vs static of the same size (mj*) and the regular 143-point grid. Found counts under the 1 %
and 5 % fixed-rate rules and v3; v3 false indications (median). Paired by start layout (more/same/fewer, sign test)."""
import json, numpy as np
from scipy.stats import binomtest, mannwhitneyu
from plot_fixed_rate import found
from run_overview import STATIC30
from score_inversions import INV, PREP
S = json.load(open(INV / "scores_v3.json"))
geo = json.loads((PREP / "geometry.json").read_text())["scatter"]["defects"]
R = np.load(PREP / "scatter.npz")["R"]; O = 0.5 * (R.min(0) + R.max(0))
C = {}
def sc(t):
    if t not in C:
        f = found(t, geo, O); C[t] = (f[0], f[1], S[t]["found"])
    return C[t]
RULES = ("1 %", "5 %", "v3")
MV = {4: ("rc30", "j30a", "x48c"), 2: ("rc30", "j30a", "j30b", "x48a", "x48c"), 1: ("rc30", "j30a", "j30b", "x48a", "x48c")}
P = {4: ("ad", "rn", "od", ""), 2: ("a2", "r2", "o2", "s2"), 1: ("a1", "r1", "o1", "k1")}
def fi(ts): return int(np.median([S[t]["false_alarm"] for t in ts]))
def line(name, G, base=None):
    out = []
    for r, rn in enumerate(RULES):
        cells = []
        for k in (4, 2, 1):
            if not G.get(k): cells.append("   -   "); continue
            v = [sc(t)[r] for t in G[k]]; cells.append(f"{np.mean(v):.2f} ({sum(x == 5 for x in v)}/{len(v)})")
        out.append(f"{rn}: " + " | ".join(cells))
    fis = " / ".join(str(fi(G[k])) if G.get(k) else "-" for k in (4, 2, 1))
    print(f"  {name:26} " + "   ".join(out) + f"   FI v3 {fis}")
def paired(name, pairs):
    for r, rn in enumerate(RULES):
        d = [sc(a)[r] - sc(b)[r] for a, b in pairs]; w, e, l = sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d)
        print(f"    {name:40} {rn}: {w}/{e}/{l}  p={binomtest(w, w + l, .5, alternative='greater').pvalue if w + l else 1:.3f}")
print("mean found (all five) per 4 / 2 / 1 shots, rules 1 %, 5 %, v3; median v3 false indications\n")
print("## case (i): moving array (final 30 points)")
line("static 30", STATIC30)
for i, n in ((0, "gradient moves"), (2, "offline moves"), (1, "random moves")):
    line(n, {k: [f"{P[k][i]}_{s}" for s in MV[k]] for k in MV})
prs = lambda i, j: [(f"{P[k][i]}_{s}", f"{P[k][j]}_{s}") for k in MV for s in MV[k]]
paired("gradient vs offline (same start)", prs(0, 2))
paired("gradient vs random (same start)", prs(0, 1))
paired("offline vs random (same start)", prs(2, 1))
paired("offline vs static same start", [(f"{P[k][2]}_{s}", f"{P[k][3]}{s}") for k in MV for s in MV[k]])
print("\n## case (ii): fixed scan budget, one-pass inversion")
SIM = {4: ["d30a", "d30b", "d30c"], 2: ["s2d30", "s2d30b", "s2d30c"], 1: ["q1d_rc30", "q1d_j30a", "q1d_j30b"]}
RI = {4: ["ri_a30"] + [f"ri_q4a_j30{s}" for s in "abcd"], 2: ["ri_s2a30"] + [f"ri_q2a_j30{s}" for s in "abcd"], 1: [f"ri_q1a_{s}" for s in ("rc30", "j30a", "j30b", "j30c")]}
line("static 30", STATIC30); line("simulation design", SIM); line("adaptive, re-inverted", RI)
paired("simulation vs same-start static", [("d30a", "rc30"), ("d30b", "j30a"), ("d30c", "j30b"), ("s2d30", "s2rc30"), ("s2d30b", "s2j30a"),
        ("s2d30c", "s2j30b"), ("q1d_rc30", "q1s_rc30"), ("q1d_j30a", "q1s_j30a"), ("q1d_j30b", "q1s_j30b")])
paired("re-inverted adaptive vs simulation, same start", [("ri_a30", "d30a"), ("ri_q4a_j30a", "d30b"), ("ri_q4a_j30b", "d30c"),
        ("ri_s2a30", "s2d30"), ("ri_q2a_j30a", "s2d30b"), ("ri_q2a_j30b", "s2d30c"), ("ri_q1a_rc30", "q1d_rc30"),
        ("ri_q1a_j30a", "q1d_j30a"), ("ri_q1a_j30b", "q1d_j30b")])
for r, rn in enumerate(RULES[:3]):
    a = []; b = []
    for k in (4, 2, 1):
        m = np.mean([sc(t)[r] for t in STATIC30[k]]); a += [sc(t)[r] - m for t in SIM[k]]; b += [sc(t)[r] - m for t in STATIC30[k]]
    print(f"    simulation vs static, MWU centred per shot {rn}: p={mannwhitneyu(a, b, alternative='greater').pvalue:.3f}")
print("\n## case (iv): all visited positions (~145), one pass")
MV2 = {4: MV[4], 2: MV[2]}
for i, n in ((0, "gradient-visited"), (2, "offline-visited"), (1, "random-visited")):
    line(n, {k: [f"ev_{P[k][i]}_{s}" for s in MV2[k]] for k in MV2})
line("static, same size (jittered)", {k: [f"mj{k}_{s}" for s in MV2[k]] for k in MV2})
line("regular 13x11 (143)", {4: ["rg4_143"], 2: ["rg2_143"]})
ev = lambda i: [(f"ev_{P[k][i]}_{s}") for k in MV2 for s in MV2[k]]
mj = [f"mj{k}_{s}" for k in MV2 for s in MV2[k]]
paired("gradient-visited vs static same size", list(zip(ev(0), mj)))
paired("gradient-visited vs offline-visited", list(zip(ev(0), ev(2))))
paired("gradient-visited vs random-visited", list(zip(ev(0), ev(1))))
print("    v3 FI per start: grad " + str([S[t]["false_alarm"] for t in ev(0)]) + " offline " + str([S[t]["false_alarm"] for t in ev(2)])
      + " static " + str([S[t]["false_alarm"] for t in mj]))
print("\n## 1-shot all visited (offline only; no ev_a1/ev_r1 runs)")
line("offline-visited", {1: [f"ev_o1_{s}" for s in MV[1]]})
