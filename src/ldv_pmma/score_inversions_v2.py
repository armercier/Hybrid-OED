"""Post-hoc scores, version 2 (19 Nov): the glued cube M1 and the through-hole D3 are scored as what the thickness
model makes of them (a stiff plug with a thinner ring), and their zones are kept out of the false-cell count.
Defects are used for scoring only.

Per run:
  holes      D1, D2, D4 (blind) and D3 (through): found if min h < 7 mm within 8 mm (unchanged)
  M1         found if max h > 13 mm within 8 mm (a glued mass appears as a thick, stiff plug)
  false thin cells   h < 9 mm, inside |u| < 140, |v| < 90, > 20 mm from D1/D2/D4 and > 40 mm from M1 and D3
  false thick cells  h > 11 mm, same region
  RMS vs v3, all cells / outside the M1 and D3 zones (40 mm); RMS vs as built outside those zones
  (as built: 10 mm, blind holes 4 mm over their diameter); RMS vs as built on all cells, with D3 at h_min (1 mm, the model's
  lower limit for a through-hole) and M1 at 10 mm (the cube does not change the plate thickness).
Usage:  python src/ldv_pmma/score_inversions_v2.py [tags]    Output: ldv_pmma_results/inversion/scores_v2.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from score_inversions import INV, PREP, thickness

ZONE = 40.0


def main():
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    D = {n: (d["x"] - O[0], d["y"] - O[1], d) for n, d in geo["scatter"]["defects"].items()}
    tags = sys.argv[1:] or sorted(p.name for p in INV.iterdir() if (p / "result.npz").exists() and (p / "settings.json").exists())  # finished runs only
    ref = thickness("v3")[0]
    out_f = INV / "scores_v2.json"
    S = json.loads(out_f.read_text()) if out_f.exists() else {}
    print(f"{'tag':9s} {'pts':>4s} holes  M1max  falseThin falseThick  RMSv3_all RMSv3_out RMSbuilt_out RMSbuilt_all")
    for tag in tags:
        t = thickness(tag)
        if t is None:
            continue
        h, Xc, Yc, st = t
        X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
        r = {n: np.hypot(X - u, Y - v) for n, (u, v, _) in D.items()}
        inner = (np.abs(X) < 140) & (np.abs(Y) < 90)
        zone = (r["M1"] < ZONE) | (r["D3"] < ZONE)
        region = inner & ~zone & (np.min([r[n] for n in ("D1", "D2", "D4")], 0) > 20)
        built = np.full_like(h, 10.0)
        for n in ("D1", "D2", "D4"):
            built[r[n] <= D[n][2]["diameter_mm"] / 2] = 10.0 - D[n][2]["depth_mm"]
        built_all = built.copy()
        built_all[r["D3"] <= D["D3"][2]["diameter_mm"] / 2] = st.get("hmin", 1.0)   # through-hole: model lower limit
        s = {n: round(float(h[r[n] < 8].min()), 2) for n in ("D1", "D2", "D3", "D4")}
        s["holes"] = int(sum(s[n] < 7 for n in ("D1", "D2", "D3", "D4")))
        s["M1_max_h"] = round(float(h[r["M1"] < 8].max()), 2)
        s["M1_found"] = bool(s["M1_max_h"] > 13)
        s["false_thin"] = int(np.sum(region & (h < 9)))
        s["false_thick"] = int(np.sum(region & (h > 11)))
        same = h.shape == ref.shape
        s["rms_v3_all"] = round(float(np.sqrt(np.mean((h - ref)[inner] ** 2))), 3) if same else None
        s["rms_v3_out"] = round(float(np.sqrt(np.mean((h - ref)[inner & ~zone] ** 2))), 3) if same else None
        s["rms_built_out"] = round(float(np.sqrt(np.mean((h - built)[inner & ~zone] ** 2))), 3)
        s["rms_built_all"] = round(float(np.sqrt(np.mean((h - built_all)[inner] ** 2))), 3)
        s["points"] = st.get("n_points")
        S[tag] = s
        print(f"{tag:9s} {str(s['points']):>4s} {s['holes']:5d} {s['M1_max_h']:6.1f} {s['false_thin']:9d} {s['false_thick']:10d}"
              f"  {s['rms_v3_all']!s:>9s} {s['rms_v3_out']!s:>9s} {s['rms_built_out']:12.3f} {s['rms_built_all']:11.3f}")
    out_f.write_text(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
