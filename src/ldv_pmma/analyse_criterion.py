"""Illumination criterion check (CPU, numpy): probe noise and bias of the two score terms, and how well the score ranks
real layouts for different beta. Score = var(h / mean h) - beta * log(mean h), lower is better (paper: beta 0.05).

Inputs: ldv_pmma_results/criterion_check/terms{4,2}_p{12,48,192}.json (oed_replay.py --score-terms, background model,
4 mm grid) and ldv_pmma_results/inversion/scores.json (outcomes; the defects are used only here).
"found" hole = min h < 7 mm within 8 mm. Correlations are Spearman, signed so that + means the criterion agrees with quality.

Usage:  python src/ldv_pmma/analyse_criterion.py
Output: printed tables, ldv_pmma_results/criterion_check/analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
CC = REPO / "ldv_pmma_results" / "criterion_check"
S = json.loads((REPO / "ldv_pmma_results" / "inversion" / "scores.json").read_text())
PS = (12, 48, 192)
BETAS = (0.0, 0.01, 0.05, 0.2, 1.0, 5.0)
GROUPS = {
    "4 scans, 30-31 pts, baselines": ["r48", "rc30", "j30a", "j30b", "x48a", "x48b", "x48c", "x48d"],
    "4 scans, 30-31 pts, all": ["r48", "rc30", "j30a", "j30b", "x48a", "x48b", "x48c", "x48d",
                                "o48b", "s31t", "s31c", "d30a", "d30b", "d30c", "a30"],
    "2 sources, 30-31 pts, all": ["s2rc30", "s2j30a", "s2j30b", "s2x48a", "s2x48b", "s2x48c", "s2x48d", "s2d30", "s2a30"],
}


def holes(t):
    return sum(S[t][d] < 7 for d in ("D1", "D2", "D3", "D4"))


def main():
    T = {}
    for nsrc in (4, 2):
        for p in PS:
            f = CC / f"terms{nsrc}_p{p}.json"
            if f.exists():
                for t, v in json.loads(f.read_text()).items():
                    T.setdefault(t, {})[p] = v
    out = {}
    print("1. Probe noise and bias of the variance term (mean over layouts)")
    print(f"{'probes':>7s} {'var term':>9s} {'sd between seeds':>17s} {'sd of log mean':>15s}")
    for p in PS:
        ts = [t for t in T if p in T[t]]
        v = np.array([np.mean(T[t][p]["var"]) for t in ts]); sd = np.array([np.std(T[t][p]["var"], ddof=1) for t in ts])
        lm = np.array([np.std(np.log(T[t][p]["mean"]), ddof=1) for t in ts])
        print(f"{p:7d} {v.mean():9.4f} {sd.mean():17.4f} {lm.mean():15.4f}")
        out[f"p{p}"] = dict(var=float(v.mean()), sd_var=float(sd.mean()), sd_logmean=float(lm.mean()))
    if all(48 in T[t] and 192 in T[t] for t in T):
        vinf = {t: (192 * np.mean(T[t][192]["var"]) - 48 * np.mean(T[t][48]["var"])) / 144 for t in T}
        frac = np.mean([1 - vinf[t] / np.mean(T[t][12]["var"]) for t in T])
        print(f"   share of the 12-probe variance term that is probe noise (extrapolated, 48 -> 192 probes): {frac:.0%}")
        out["noise_share_12"] = float(frac)
    else:
        vinf = None
    print("\n2. Ranking of real layouts (Spearman; + = criterion agrees): holes / fewer false cells / lower RMS")
    for g, ts in GROUPS.items():
        ts = [t for t in ts if t in T and t in S]
        print(f"\n   {g} (n = {len(ts)})")
        q = dict(h=[holes(t) for t in ts], f=[-S[t]["false_cells"] for t in ts], r=[-S[t]["rms_vs_v3"] for t in ts])
        for p in PS:
            if not all(p in T[t] for t in ts):
                continue
            for b in BETAS:
                x = [-(np.mean(T[t][p]["var"]) - b * np.mean(np.log(T[t][p]["mean"]))) for t in ts]
                c = [spearmanr(x, q[k])[0] for k in "hfr"]
                print(f"   probes {p:3d} beta {b:5.2f}: {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}")
                out.setdefault(g, {})[f"p{p}_b{b}"] = [float(v) for v in c]
            x = [np.mean(np.log(T[t][p]["mean"])) for t in ts]
            c = [spearmanr(x, q[k])[0] for k in "hfr"]
            print(f"   probes {p:3d} mean only : {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}")
            out[g][f"p{p}_meanonly"] = [float(v) for v in c]
        if vinf is not None:
            x = [-vinf[t] for t in ts]
            c = [spearmanr(x, q[k])[0] for k in "hfr"]
            print(f"   noise-corrected variance only: {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}")
            out[g]["var_corrected"] = [float(v) for v in c]
    (CC / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
