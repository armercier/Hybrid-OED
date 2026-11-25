"""Overview of all LDV inversions (v3 scores, scores_v3.json) for the two storyline questions:
  Q1  same scan-time budget (same number of distinct measured points): do designed layouts find more features than
      regular / jittered / random layouts, and with how many false indications?
  Q2  same sensor count and move lengths: are gradient-directed moves better than random moves?
for 4, 2 and 1 shots. Budget = distinct measured LDV points (scan time ~ 3.8 s per point).

Usage:  python src/ldv_pmma/run_overview.py
Output: reports/2026-11_mssp_paper/notes/run_overview.md (tables), printed summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "ldv_pmma_results" / "inversion"
OUT = REPO / "reports" / "2026-11_mssp_paper" / "notes" / "run_overview.md"
S = json.loads((INV / "scores_v3.json").read_text())
FEAT = ("D1", "D2", "D3", "D4", "M1")

STATIC30 = {4: ["rc30", "r48", "j30a", "j30b", "x48a", "x48b", "x48c", "x48d"],
            2: ["s2rc30", "s2j30a", "s2j30b", "s2x48a", "s2x48b", "s2x48c", "s2x48d"],
            1: [f"q1s_{s}" for s in ("rc30", "j30a", "j30b", "j30c", "j30d", "x30a", "x30b")]}   # file 3 only, exactly 30 points
DESIGN30 = {  # (tag, method)
    4: [("d30a", "simulation design"), ("d30b", "simulation design"), ("d30c", "simulation design"),
        ("a30", "adaptive, 3 groups of 10"), ("o48b", "adaptive, points added per round (early settings)"),
        ("s31t", "greedy selection (targeted)"), ("s31c", "greedy selection (coverage)")],
    2: [("s2d30", "simulation design"), ("s2a30", "adaptive, 3 groups of 10")]
       + [(f"q2a_j30{s}", "adaptive, 3 groups of 10") for s in "abcd"],
    1: [(f"q1a_{s}", "adaptive, 3 groups of 10") for s in ("rc30", "j30a", "j30b", "j30c")]}
DESIGN30[4] += [(f"q4a_j30{s}", "adaptive, 3 groups of 10") for s in "abcd"]
LOGC = "adaptive (log criterion), 3 groups of 10"   # C5: -mean log I, same protocol
DESIGN30[4] += [(f"q4l_j30{s}", LOGC) for s in "abcd"]
DESIGN30[2] += [(f"q2l_j30{s}", LOGC) for s in "abcd"]
DESIGN30[1] += [(f"q1l_{s}", LOGC) for s in ("rc30", "j30a", "j30b", "j30c")]
PROGC = "static, progressive (3 groups, no design)"   # C4b control, same 33/33/34 inversion
DESIGN30[4] += [(t, PROGC) for t in ["n30"] + [f"q4n_j30{s}" for s in "abcd"]]
DESIGN30[2] += [(t, PROGC) for t in ["s2n30"] + [f"q2n_j30{s}" for s in "abcd"]]
DESIGN30[1] += [(t, PROGC) for t in [f"q1n_{s}" for s in ("rc30", "j30a", "j30b", "j30c")]]
STATIC_MID = {4: ["x36a", "x36b", "r36", "rc48", "r24", "x24a", "r18"], 2: [],
              1: ["k1rc30", "k1j30a", "k1j30b", "k1x48a", "k1x48c"]}   # earlier 1-shot statics, 24-27 usable points
# planned runs appear once scored (queue_q1.sh still running)
_done = lambda t: t in S and (INV / t / "settings.json").exists()
STATIC30 = {k: [t for t in v if _done(t)] for k, v in STATIC30.items()}
DESIGN30 = {k: [(t, m) for t, m in v if _done(t)] for k, v in DESIGN30.items()}
FULL = {4: ["r12", "r9", "x9b", "v3"], 2: ["s2full"], 1: ["k1full"]}
PAIRS = {4: [("ad_rc30", "rn_rc30", "rc30"), ("ad_j30a", "rn_j30a", "j30a"), ("ad_x48c", "rn_x48c", "x48c"),
             ("o48f", "o48r", "r48 (12 probes, early)")],
         2: [(f"a2_{s}", f"r2_{s}", f"s2{s}") for s in ("rc30", "j30a", "x48c", "j30b", "x48a")],
         1: [(f"a1_{s}", f"r1_{s}", f"k1{s}") for s in ("rc30", "j30a", "x48c", "j30b", "x48a")]}


def distinct(tag):
    st = json.loads((INV / tag / "settings.json").read_text())
    return st.get("distinct_points") or st.get("n_points")


def row(tag, kind=""):
    s = S[tag]
    miss = ",".join(n for n in FEAT if n not in s["found_list"]) or "-"
    return f"| {tag} | {kind} | {distinct(tag) or 'all'} | {s['found']} | {miss} | {s['false_alarm']} | {s['bg_rms']:.3f} |"


def key(t):
    s = S[t]
    return (-s["found"], s["false_alarm"], s["bg_rms"])


def main():
    L = ["# LDV inversions: overview for the storyline (v3 scores)", "",
         "Budget = distinct measured LDV points (about 3.8 s each; full scan 6418 points, about 7 h). found = features of 5 "
         "(D1-D4, M1) with |h - 10 mm| > 3 mm within 8 mm; false ind. = cells (3 mm) with |h - 10| > 1 mm farther than 20 mm "
         "from every feature; bg RMS = RMS of h - 10 mm there.", ""]
    summ = []
    for shots in (4, 2, 1):
        L += [f"## {shots} shot{'s' if shots > 1 else ''}", "", "| run | layout | points | found | missing | false ind. | bg RMS |",
              "|---|---|---|---|---|---|---|"]
        for t in FULL[shots]:
            L.append(row(t, "full / dense static"))
        for t in STATIC_MID[shots]:
            L.append(row(t, "static, other budgets"))
        for t in STATIC30[shots]:
            L.append(row(t, "static, 30 points"))
        for t, m in DESIGN30[shots]:
            L.append(row(t, m))
        for a, r, st in PAIRS[shots]:
            L.append(row(a, f"moving, gradient (start {st})"))
            L.append(row(r, f"moving, random (start {st})"))
        L.append("")
        # Q1 at 30 points
        base = [S[t] for t in STATIC30[shots]]
        des = [(t, m) for t, m in DESIGN30[shots] if not m.startswith(("greedy", "static")) and "early" not in m and "log" not in m]
        nb5 = sum(b["found"] == 5 for b in base)
        nd5 = sum(S[t]["found"] == 5 for t, _ in des)
        p = fisher_exact([[nd5, len(des) - nd5], [nb5, len(base) - nb5]], alternative="greater")[1] if des else float("nan")
        summ.append(f"Q1, {shots} shot(s), 30 points: all 5 found in {nd5}/{len(des)} illumination designs (simulation or adaptive) "
                    f"vs {nb5}/{len(base)} static layouts (one-sided Fisher p = {p:.3f}); false indications designs "
                    f"{[S[t]['false_alarm'] for t, _ in des]} vs static median {int(np.median([b['false_alarm'] for b in base]))}")
        # Q2 pairs
        P = [(a, r) for a, r, st in PAIRS[shots] if "early" not in st]
        w = sum(key(a) < key(r) for a, r in P)
        wf = sum(S[a]["found"] > S[r]["found"] for a, r in P); lf = sum(S[a]["found"] < S[r]["found"] for a, r in P)
        wfi = sum(S[a]["false_alarm"] < S[r]["false_alarm"] for a, r in P)
        wbg = sum(S[a]["bg_rms"] < S[r]["bg_rms"] for a, r in P)
        p2 = binomtest(w, len(P), 0.5, alternative="greater").pvalue if P else float("nan")
        summ.append(f"Q2, {shots} shot(s): gradient ranks above random in {w}/{len(P)} pairs (sign test p = {p2:.3f}); "
                    f"more features {wf}, fewer {lf}; fewer false indications {wfi}/{len(P)}; lower bg RMS {wbg}/{len(P)}")
    L += ["## Summary", ""] + [f"- {s}" for s in summ]
    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(summ))
    print(OUT)


if __name__ == "__main__":
    main()
