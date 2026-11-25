"""Step 2: geometry of the PMMA specimen in the frames of both LDV scans.

- Sources: fitted from the arrival times of each source alone (file k - file 1) in each scan
  (locate_sources_scans.py -> ldv_pmma_results/scatter/sources_scans.json). Hand-measured positions
  were 10-245 mm off and are not used.
- Plate movement between the scans: wavefield registration (scatter_maps.py -> registration.json):
  pristine point p  ->  defect-scan location  R(theta)(p - O) + O + T,  O = pristine grid centre.
- Defects: planned positions relative to O (the drilling plan), as built (ldv_work/defects.txt),
  mapped into the defect-scan frame with the registration.
- PZT transducers: blocks 10 x 10 mm, 7 mm high, bonded on the back face (axis-aligned assumed).

Usage:  python src/ldv_pmma/geometry.py   ->  ldv_work/prepared/geometry.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "ldv_pmma_results" / "scatter"
OUT = REPO / "ldv_work" / "prepared"

PLATE = dict(material="PMMA", thickness_mm=10.0, size_mm=[900, 835], vp=2730.0, vs=1340.0, rho=1180.0,
             note="vp, vs, rho as in the Salvus specimen; the calibration fits a velocity scale")
PZT = dict(shape="block", size_mm=[10.0, 10.0], height_mm=7.0, rho=7600.0, vp=4350.0, vs=1700.0,
           mass_g=7600.0 * 10e-3 * 10e-3 * 7e-3 * 1e3, face="back")
# as built (defects.txt): position (u, v) mm from the pristine grid centre, diameter, depth from the back
DEFECTS = {"D1": dict(u=-75, v=47, kind="blind hole", diameter_mm=10, depth_mm=6),
           "D2": dict(u=70, v=52, kind="blind hole", diameter_mm=10, depth_mm=6),
           "D3": dict(u=0, v=-13, kind="through-hole", diameter_mm=10, depth_mm=10),
           "D4": dict(u=-85, v=-53, kind="blind hole", diameter_mm=8, depth_mm=6),
           "M1": dict(u=80, v=-48, kind="steel cube (glued, back)", size_mm=10, mass_g=7.85)}


def main():
    src = json.loads((RES / "sources_scans.json").read_text())
    reg = json.loads((RES / "registration.json").read_text())
    O = np.array(reg["O"])
    th = np.radians(reg["rotation_deg"]); T = np.array(reg["shift_mm"])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    to_scatter = lambda p: Rm @ (np.asarray(p) - O) + O + T
    geo = dict(plate=PLATE, pzt=PZT, registration=reg, frames="scanner coordinates, mm")
    for scan in ("pristine", "scatter"):
        g = dict(sources={})
        for k in (1, 2, 3, 4):
            s = src[scan][f"file{k}"]
            g["sources"][f"S{k}"] = dict(file=k, x=s["x"], y=s["y"], arrival_resid_us=s["resid_us"])
        g["defects"] = {}
        for n, d in DEFECTS.items():
            p = O + np.array([d["u"], d["v"]])
            if scan == "scatter":
                p = to_scatter(p)
            g["defects"][n] = dict(d, x=float(p[0]), y=float(p[1]))
        geo[scan] = g
    # check: sources mapped with the registration vs fitted in the defect scan
    geo["check_sources_pristine_to_scatter_mm"] = {
        f"S{k}": (np.array([src["scatter"][f"file{k}"]["x"], src["scatter"][f"file{k}"]["y"]])
                  - to_scatter([src["pristine"][f"file{k}"]["x"], src["pristine"][f"file{k}"]["y"]])).round(1).tolist()
        for k in (1, 2, 3, 4)}
    (OUT / "geometry.json").write_text(json.dumps(geo, indent=1))
    print(json.dumps({k: geo[k] for k in ("pristine", "scatter", "check_sources_pristine_to_scatter_mm")}, indent=1))


if __name__ == "__main__":
    main()
