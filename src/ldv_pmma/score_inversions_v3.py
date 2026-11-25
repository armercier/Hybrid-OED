"""Post-hoc scores, version 3 (19 Nov): detect, locate, contain. Every feature (blind holes D1, D2, D4, through-hole D3,
glued cube M1) is treated alike through the anomaly map a = |h - 10 mm|, whatever the model makes of it (thinning or plug).
Defects are used for scoring only.

Per run, inside |u| < 140, |v| < 90 mm (3 mm map):
  found          a > 3 mm somewhere within 8 mm of the feature (count out of 5)
  loc_err        distance from the feature centre to the peak of a within 25 mm (median over found features)
  false_alarm    (false indications) cells with a > 1 mm farther than 20 mm from every feature (thin and thick alike; 20 mm ~ half an A0 wavelength)
  bg_rms         RMS of h - 10 mm over those same background cells
  spill_M1/D3    radius (mm) holding 90 % of sum(a^2) within 60 mm of the feature, cells near the other features left out
Ranking: found (more is better), then false indications, then background RMS.

Usage:  python src/ldv_pmma/score_inversions_v3.py [tags]     Output: ldv_pmma_results/inversion/scores_v3.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from score_inversions import INV, PREP, thickness

FEAT = ("D1", "D2", "D3", "D4", "M1")
R_FOUND, A_FOUND, R_PEAK, R_BG, A_FALSE, R_SPILL = 8.0, 3.0, 25.0, 20.0, 1.0, 60.0


def spill(a, r, other):
    m = (r < R_SPILL) & ~other
    order = np.argsort(r[m]); e = np.cumsum(a[m][order] ** 2)
    return float(r[m][order][np.searchsorted(e, 0.9 * e[-1])]) if e[-1] > 0 else 0.0


def main():
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    D = {n: (d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    tags = sys.argv[1:] or sorted(p.name for p in INV.iterdir() if (p / "result.npz").exists() and (p / "settings.json").exists())
    out_f = INV / "scores_v3.json"
    S = json.loads(out_f.read_text()) if out_f.exists() else {}
    for tag in tags:
        t = thickness(tag)
        if t is None:
            continue
        h, Xc, Yc, st = t
        X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
        a = np.abs(h - 10.0)
        inner = (np.abs(X) < 140) & (np.abs(Y) < 90)
        r = {n: np.hypot(X - u, Y - v) for n, (u, v) in D.items()}
        dmin = np.min([r[n] for n in FEAT], 0)
        s = {}
        for n in FEAT:
            s[f"{n}_a"] = round(float(a[r[n] < R_FOUND].max()), 2)
            w = r[n] < R_PEAK
            k = np.argmax(np.where(w, a, -1))
            s[f"{n}_loc"] = round(float(r[n].flat[k]), 1)
        found = [n for n in FEAT if s[f"{n}_a"] > A_FOUND]
        bg = inner & (dmin > R_BG)
        s.update(found=len(found), found_list=found,
                 loc_err=round(float(np.median([s[f"{n}_loc"] for n in found])), 1) if found else None,
                 false_alarm=int(np.sum(bg & (a > A_FALSE))), bg_rms=round(float(np.sqrt(np.mean((h[bg] - 10) ** 2))), 3),
                 points=st.get("n_points"), active_files=st.get("active_files"))
        for n in ("M1", "D3"):
            other = np.min([r[m] for m in FEAT if m != n], 0) < R_BG
            s[f"spill_{n}"] = round(spill(a, r[n], other | ~inner), 1)
        S[tag] = s
    out_f.write_text(json.dumps(S, indent=1))
    rows = sorted((t for t in tags if t in S), key=lambda t: (-S[t]["found"], S[t]["false_alarm"], S[t]["bg_rms"]))
    print(f"{'tag':12s} {'pts':>4s} found  missing        loc  false  bgRMS  spillM1 spillD3")
    for t in rows:
        s = S[t]
        miss = ",".join(n for n in FEAT if n not in s["found_list"]) or "-"
        print(f"{t:12s} {str(s['points']):>4s} {s['found']:5d}  {miss:12s} {s['loc_err']!s:>5s} {s['false_alarm']:6d} {s['bg_rms']:6.3f}"
              f" {s['spill_M1']:7.1f} {s['spill_D3']:7.1f}")


if __name__ == "__main__":
    main()
