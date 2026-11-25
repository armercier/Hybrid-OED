"""Score defect-scan inversions against the as-built defects and the blind full-scan reference v3 (CPU only, numpy).
Post-hoc evaluation only: the defect positions are used here, never in the inversions.

Per run: min thickness within 8 mm of each defect (D1-D4) and the cube M1; number of "false" cells (h < 9 mm farther
than 20 mm from every defect, inside |u| < 140, |v| < 90 mm); RMS thickness difference to v3 in the same area.

Usage:  python src/ldv_pmma/score_inversions.py [tags ...]     (default: all runs with result.npz)
Output: ldv_pmma_results/inversion/scores.json (merged, one entry per tag), table on stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREP = REPO / "ldv_work" / "prepared"
INV = REPO / "ldv_pmma_results" / "inversion"
NAMES = ("D1", "D2", "D3", "D4", "M1")


def thickness(tag):
    r = np.load(INV / tag / "result.npz")
    st = json.loads((INV / tag / "settings.json").read_text()) if (INV / tag / "settings.json").exists() else {}
    if st.get("param", "log_thickness") != "log_thickness":
        return None
    hmin = st.get("hmin", 1.0)
    return hmin + (10.0 - hmin) * np.exp(r["m"][0]), r["Xc"], r["Yc"], st


def main():
    geo = json.loads((PREP / "geometry.json").read_text())
    R = np.load(PREP / "scatter.npz")["R"]
    O = 0.5 * (R.min(0) + R.max(0))
    D = {n: (d["x"] - O[0], d["y"] - O[1]) for n, d in geo["scatter"]["defects"].items()}
    tags = sys.argv[1:] or sorted(p.name for p in INV.iterdir() if (p / "result.npz").exists())
    ref = thickness("v3")
    out_f = INV / "scores.json"
    scores = json.loads(out_f.read_text()) if out_f.exists() else {}
    print(f"{'tag':8s} {'pts':>5s} " + " ".join(f"{n:>5s}" for n in NAMES) + "  false  RMSv3")
    for tag in tags:
        t = thickness(tag)
        if t is None:
            continue
        h, Xc, Yc, st = t
        X, Y = Xc * 1e3 - O[0], Yc * 1e3 - O[1]
        inner = (np.abs(X) < 140) & (np.abs(Y) < 90)
        dist = np.min([np.hypot(X - u, Y - v) for u, v in D.values()], 0)
        s = {n: round(float(h[np.hypot(X - u, Y - v) < 8].min()), 2) for n, (u, v) in D.items()}
        s["false_cells"] = int(np.sum(inner & (dist > 20) & (h < 9)))
        s["false_min_h"] = round(float(h[inner & (dist > 20)].min()), 2)
        s["rms_vs_v3"] = round(float(np.sqrt(np.mean((h - ref[0])[inner] ** 2))), 3) if h.shape == ref[0].shape else None
        s["points"] = st.get("n_points")
        scores[tag] = s
        print(f"{tag:8s} {str(s['points']):>5s} " + " ".join(f"{s[n]:5.2f}" for n in NAMES)
              + f"  {s['false_cells']:5d}  {s['rms_vs_v3']}  (min h away from defects {s['false_min_h']})")
    out_f.write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
