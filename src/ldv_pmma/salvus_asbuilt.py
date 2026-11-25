"""Salvus 3-D model of the lab PMMA plate as built, for the waveform check Mindlin vs Salvus vs LDV (paper figure).

Plate frame (front face, mm) of reports/2026-11_specimen/make_drawing_pmma_final.py: 900 x 835 x 10 mm, x right, y up,
LDV on the front face (z = H), PZT blocks, holes and the cube on the back face (z = 0). The scanner frame maps to the
plate frame by a shift: D3 is at (450, 405) in the plate frame and at (u, v) = (0, -13) from the scan centre O, so
plate = scanner - O + (450, 418) (the registration rotation, -0.09 deg, is neglected).

Two configurations, each with its own scan's calibrated values (no defect knowledge enters the pristine run):
  pristine : PZT blocks at the pristine-calibrated source positions, s 0.995, Q 34.5 (calib_pristine.json)
  asbuilt  : PZT blocks at the blind defect-scan source positions, s 0.990, Q 30 (calib_blind.json), plus the as-built
             defects: D1, D2 blind holes 10 mm, D4 8 mm, 6 mm deep from the back; D3 through-hole 10 mm; M1 steel cube
             10 mm glued on the back (perfect bond).
Material PMMA vp 2730 s, vs 1340 s, rho 1180, Q_mu = Q, Q_kappa = 2Q; PZT blocks 10 x 10 x 7 mm (vp 4350, vs 1700,
rho 7600); steel cube vp 5900, vs 3200, rho 7850. Free plate edges. Vertical point force at the block centre on the
back face, Ricker 45 kHz (same wavelet as the Mindlin runs; the real signature is estimated later per file).
Receivers: velocity on the front face at the scan points used by calibrate_pristine.Calib for that scan; points within
6 mm of the D3 centre (no plate surface) are left out in the asbuilt run.

Hexahedral mesh on rectilinear grid lines: 2 mm over the scan area, 3 mm to the sources, 6 mm outside; order 4; layers
z = 0, 3, 6, 10 mm in the plate (hole bottoms at 6 mm), -7 and -10 mm below for the blocks and the cube.

Run with the Salvus venv (GPU site local_gpu):
  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/ldv_pmma/salvus_asbuilt.py [--configs pristine asbuilt] [--dry]
Output: ldv_pmma_results/salvus_asbuilt/traces_<config>.npz (vz per source on the receivers, time axis, receiver
        positions in the scanner frame), project in ldv_pmma_results/salvus_asbuilt/project
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn
from salvus.mesh.simple_mesh import basic_mesh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "salvus_specimen"))
from build_and_run import STEEL, axis, run_one  # noqa: E402

PREP = REPO / "ldv_work" / "prepared"
CAL = REPO / "ldv_pmma_results" / "calib"
OUT = REPO / "ldv_pmma_results" / "salvus_asbuilt"
PMMA = dict(vp=2730.0, vs=1340.0, rho=1180.0)
PZT = dict(vp=4350.0, vs=1700.0, rho=7600.0, side=10.0, height=7.0)
L, H = (900.0, 835.0), 10.0
XG = axis([(0, 180, 30), (180, 290, 37), (290, 610, 160), (610, 720, 37), (720, 900, 30)])
YG = axis([(0, 210, 35), (210, 300, 30), (300, 540, 120), (540, 630, 30), (630, 835, 34)])
ZG = np.array([-10.0, -7.0, 0.0, 3.0, 6.0, 10.0]) * 1e-3
F0, END = 45e3, 0.36e-3
KS = (1, 2, 3, 4)


def setup(cfg):
    geo = json.loads((PREP / "geometry.json").read_text())
    scan = "pristine" if cfg == "pristine" else "scatter"
    d3 = geo[scan]["defects"]["D3"]
    O = np.array([d3["x"] - d3["u"], d3["y"] - d3["v"]])
    off = O - np.array([450.0, 418.0])                     # scanner -> plate frame: p - off
    c = json.loads((CAL / ("calib_pristine.json" if cfg == "pristine" else "calib_blind.json")).read_text())
    src = np.array([c["sources_mm"][f"S{k}"] for k in KS]) - off
    d = np.load(PREP / f"{scan}.npz")
    R = d["R"]
    Oc = 0.5 * (R.min(0) + R.max(0))
    use = np.zeros(len(R), bool)
    for k in KS:                                           # same receiver set as calibrate_pristine.Calib
        m = ~d[f"bad{k}"]
        if scan == "pristine" and k in (3, 4):
            m &= R[:, 1] < Oc[1]
        use |= m
    Ru = R[use]
    defects = {n: (v["x"] - off[0], v["y"] - off[1], v) for n, v in geo[scan]["defects"].items()}
    keep = np.ones(len(Ru), bool)
    if cfg == "asbuilt":
        keep = np.hypot(*(Ru - off - defects["D3"][:2]).T) > 6.0
    return dict(scan=scan, off=off, src=src, Ru=Ru, keep=keep, s=c["velocity_scale"], Q=c["Q"], defects=defects)


def build_mesh(S, with_defects):
    nx, ny, nz = len(XG) - 1, len(YG) - 1, len(ZG) - 1
    m = basic_mesh.CartesianHomogeneousIsotropicElastic3D(
        vp=1.0, vs=1.0, rho=1.0, x_max=float(nx), y_max=float(ny), z_max=float(nz),
        max_frequency=1.0, elements_per_wavelength=1.0, tensor_order=1).create_mesh()
    pts = m.points.copy()
    for k, g in enumerate((XG, YG, ZG)):
        pts[:, k] = np.interp(pts[:, k], np.arange(len(g)), g)
    m.points[:] = pts
    x, y, z = (m.get_element_centroid() * 1e3).T
    in_plate = z > 0
    removed, in_mass, in_pzt = (np.zeros(len(x), bool) for _ in range(3))
    counts = {}
    for k, (sx, sy) in enumerate(S["src"]):
        sel = (np.abs(x - sx) < PZT["side"] / 2) & (np.abs(y - sy) < PZT["side"] / 2) & (z < 0) & (z > -PZT["height"])
        in_pzt |= sel; counts[f"S{k + 1}"] = int(sel.sum())
    if with_defects:
        for n, (cx, cy, d) in S["defects"].items():
            if n.startswith("D"):
                sel = in_plate & (np.hypot(x - cx, y - cy) < d["diameter_mm"] / 2) & (z < d["depth_mm"])
                removed |= sel
            else:
                a = d["size_mm"]
                sel = (np.abs(x - cx) < a / 2) & (np.abs(y - cy) < a / 2) & (z < 0) & (z > -a)
                in_mass |= sel
            counts[n] = int(sel.sum())
    keep = (in_plate & ~removed) | in_mass | in_pzt
    m = m.apply_element_mask(keep)
    in_mass, in_pzt = in_mass[keep], in_pzt[keep]
    m.change_tensor_order(4)
    shape = m.connectivity.shape
    s = S["s"]
    pm = dict(vp=PMMA["vp"] * s, vs=PMMA["vs"] * s, rho=PMMA["rho"])
    for name, key in (("VP", "vp"), ("VS", "vs"), ("RHO", "rho")):
        val = np.where(in_mass, STEEL[key], np.where(in_pzt, PZT[key], pm[key]))
        m.attach_field(name, val[:, None] * np.ones(shape))
    hard = in_mass | in_pzt
    m.attach_field("QMU", np.where(hard[:, None], 1000.0, S["Q"]) * np.ones(shape))
    m.attach_field("QKAPPA", np.where(hard[:, None], 2000.0, 2 * S["Q"]) * np.ones(shape))
    from salvus.mesh.layered_meshing.detail.mesh_from_domain import _attach_linear_solids

    m = _attach_linear_solids(m, 2.0 * F0, sn.LinearSolids(reference_frequency=40e3, min_frequency=10e3, max_frequency=90e3))
    return m, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["pristine", "asbuilt"])
    ap.add_argument("--site", default="local_gpu")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    p = None
    for cfg in a.configs:
        S = setup(cfg)
        mesh, counts = build_mesh(S, cfg == "asbuilt")
        print(f"{cfg}: {mesh.nelem} elements (order 4); elements per feature {counts}; {int(S['keep'].sum())} receivers; "
              f"sources (plate frame) {np.round(S['src'], 1).tolist()}", flush=True)
        if a.dry:
            continue
        if p is None:
            p = sn.Project.from_domain(path=OUT / "project", domain=sn.domain.dim3.BoxDomain(
                x0=0.0, x1=L[0] * 1e-3, y0=0.0, y1=L[1] * 1e-3, z0=float(ZG[0]), z1=H * 1e-3), load_if_exists=True)
        Rp = S["Ru"][S["keep"]] - S["off"]
        ev = []
        for k, (sx, sy) in enumerate(S["src"]):
            name = f"{cfg}_S{k + 1}"
            ev.append(name)
            if name in set(p.events.list()):
                continue
            p.add_to_project(sn.Event(
                event_name=name,
                sources=sn.simple_config.source.cartesian.VectorPoint3D(x=sx * 1e-3, y=sy * 1e-3, z=0.0, fx=0.0, fy=0.0, fz=1.0),
                receivers=[sn.simple_config.receiver.cartesian.Point3D(x=rx * 1e-3, y=ry * 1e-3, z=H * 1e-3, station_code=f"R{i:05d}",
                                                                       fields=["velocity"]) for i, (rx, ry) in enumerate(Rp)]))
        wavelet = sn.simple_config.stf.Ricker(center_frequency=F0)
        if cfg not in set(p.entities.list("simulation_configuration")):
            p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
                name=cfg, unstructured_mesh=mesh,
                event_configuration=sn.EventConfiguration(
                    wavelet=wavelet,
                    waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                        start_time_in_seconds=wavelet.get_auto_start_time(), end_time_in_seconds=END,
                        spectral_element_order=4, receiver_sampling_rate_in_time_steps=10, attenuation=True))))
        res = dict(R_scanner=S["Ru"][S["keep"]], keep=S["keep"], off=S["off"], src_plate=S["src"], s=S["s"], Q=S["Q"])
        for name in ev:
            outdir = run_one(p, cfg, name, a.site)
            if outdir is None:
                continue
            with h5py.File(outdir / "receivers.h5") as f:
                v = f["point/velocity"][()]
                order = np.argsort(f["names_ELASTIC_point"][()])
                at = f["point"].attrs
                t = at["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / at["sampling_rate_in_hertz"][0]
                coords = f["coordinates_ELASTIC_point"][()][order, :2]
            assert np.allclose(coords * 1e3, Rp, atol=1e-3), "receiver order mismatch"
            res[f"vz_{name.split('_')[-1]}"] = v[order][:, 2, :].T.astype(np.float32)     # (nt, nrec)
            res["t"] = t
            print(f"done {name}: {v.shape}", flush=True)
            np.savez_compressed(OUT / f"traces_{cfg}.npz", **res)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
