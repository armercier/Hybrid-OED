"""Salvus 3-D model of the LDV experiment: 10 mm PMMA plate with a bonded 10 mm steel cube.

Coordinates: the LDV scan plane frame (ldv_assessment/mindlin_ldv_fit.npz: P, src),
shifted by an origin that is a multiple of the element size so the cube aligns with
elements. The plate extends 0.35 m beyond the scan on all sides (free edges; A0 edge
reflections arrive after the analysis window). Mesh: 5 mm hexahedra, order 4, two
layers through the plate; a third layer keeps only the 2 x 2 x 2 cube elements.
Source: normal force on the top surface at the transducer position, Ricker 45 kHz
(the real wavelet and trigger delay are estimated later by deconvolution).
Receivers: vz at the scan points on the plate top, except the cube footprints.

Run with the Salvus 2025.1.2 venv:
  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/ldv_assessment/salvus_ldv_cube.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn
from salvus.mesh.simple_mesh import basic_mesh

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "ldv_assessment"
PROJECT = OUT / "salvus_project"
EL = 0.005
H = 0.010
PMMA = dict(vp=2730.0, vs=1340.0, rho=1180.0)
STEEL = dict(vp=5900.0, vs=3200.0, rho=7850.0)
CUBE_Y = -0.005
CUBE_XS = (-0.025, -0.020, -0.015)
MARGIN = 0.35
F_SRC = 45e3
END_TIME = 560e-6


def build_mesh(Lx, Ly, cube_xy, q_mu=None):
    mesh = basic_mesh.CartesianHomogeneousIsotropicElastic3D(
        vp=PMMA["vp"], vs=PMMA["vs"], rho=PMMA["rho"], x_max=Lx, y_max=Ly, z_max=2 * H,
        max_frequency=PMMA["vs"] / (2.0 * EL), elements_per_wavelength=2.0, tensor_order=4,
    ).create_mesh()
    c = mesh.get_element_centroid()
    in_cube = np.zeros(len(c), bool)
    if cube_xy is not None:
        in_cube = (np.abs(c[:, 0] - cube_xy[0]) < EL) & (np.abs(c[:, 1] - cube_xy[1]) < EL) & (c[:, 2] > H)
    keep = (c[:, 2] < H) | in_cube
    mesh = mesh.apply_element_mask(keep)
    in_cube = in_cube[keep]
    for name, key in (("VP", "vp"), ("VS", "vs"), ("RHO", "rho")):
        field = mesh.element_nodal_fields[name].copy()
        field[in_cube] = STEEL[key]
        mesh.attach_field(name, field)
    if q_mu is not None:  # viscoelastic PMMA (QKAPPA = 2 QMU); steel nearly elastic
        shape = mesh.element_nodal_fields["VP"].shape
        mesh.attach_field("QMU", np.where(in_cube[:, None], 1000.0, q_mu) * np.ones(shape))
        mesh.attach_field("QKAPPA", np.where(in_cube[:, None], 2000.0, 2 * q_mu) * np.ones(shape))
        # constant Q over the data band via standard linear solids (private Salvus helper,
        # the same one Salvus uses when meshing from a domain with LinearSolids)
        from salvus.mesh.layered_meshing.detail.mesh_from_domain import _attach_linear_solids

        mesh = _attach_linear_solids(mesh, 70e3, sn.LinearSolids(
            reference_frequency=39e3, min_frequency=10e3, max_frequency=70e3))
    return mesh, int(in_cube.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_gpu")
    parser.add_argument("--q", type=float, nargs="*", default=[], help="extra attenuating runs, cube at -20 mm")
    args = parser.parse_args()

    fit = np.load(OUT / "mindlin_ldv_fit.npz")
    P, src = fit["P"], fit["src"]
    origin = np.floor((P.min(0) - MARGIN) / EL) * EL
    Lx, Ly = (np.ceil((P.max(0) + MARGIN - origin) / EL) * EL).tolist()
    footprint = np.zeros(len(P), bool)
    for cx in CUBE_XS:
        footprint |= (np.abs(P[:, 0] - cx) <= EL + 1e-4) & (np.abs(P[:, 1] - CUBE_Y) <= EL + 1e-4)
    rec_idx = np.where(~footprint)[0]
    print(f"plate {Lx:.3f} x {Ly:.3f} m, origin {origin}, {len(rec_idx)} receivers ({footprint.sum()} under cubes)")

    p = sn.Project.from_domain(path=PROJECT, domain=sn.domain.dim3.BoxDomain(
        x0=0.0, x1=Lx, y0=0.0, y1=Ly, z0=0.0, z1=2 * H), load_if_exists=True)
    if "ldv_shot" not in p.events.list():
        sx, sy = src - origin
        p.add_to_project(sn.Event(
            event_name="ldv_shot",
            sources=sn.simple_config.source.cartesian.VectorPoint3D(x=float(sx), y=float(sy), z=H,
                                                                     fx=0.0, fy=0.0, fz=-1.0),
            receivers=[sn.simple_config.receiver.cartesian.Point3D(
                x=float(P[i, 0] - origin[0]), y=float(P[i, 1] - origin[1]), z=H,
                station_code=f"R{k:05d}", fields=["velocity"]) for k, i in enumerate(rec_idx)],
        ))
    wavelet = sn.simple_config.stf.Ricker(center_frequency=F_SRC)
    configs = {"no_cube": None, **{f"cube_x{cx*1e3:+.0f}mm": (cx, CUBE_Y) for cx in CUBE_XS}}
    q_of = {f"cube_x-20mm_Q{q:.0f}": q for q in args.q}
    configs.update({name: (-0.020, CUBE_Y) for name in q_of})
    existing = set(p.entities.list("simulation_configuration"))
    for name, cube in configs.items():
        if name in existing:
            continue
        mesh, n_cube = build_mesh(Lx, Ly, None if cube is None else np.asarray(cube) - origin, q_of.get(name))
        print(f"{name}: {mesh.nelem} elements ({n_cube} steel)", flush=True)
        p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
            name=name, unstructured_mesh=mesh,
            event_configuration=sn.EventConfiguration(
                wavelet=wavelet,
                waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                    start_time_in_seconds=wavelet.get_auto_start_time(), end_time_in_seconds=END_TIME,
                    spectral_element_order=4, receiver_sampling_rate_in_time_steps=10,
                    attenuation=name in q_of))))
    for name in configs:  # one configuration at a time (sorted single event; see flexural_events.py)
        p.simulations.launch(simulation_configuration=name, events=["ldv_shot"], site_name=args.site,
                             ranks_per_job=1, verbosity=0)
        p.simulations.query(block=True)
        print("done", name, flush=True)

    out = {}
    for name in configs:
        with h5py.File(Path(p.simulations.get_simulation_output_directory(name, "ldv_shot")) / "receivers.h5") as f:
            v = f["point/velocity"][()]
            order = np.argsort(f["names_ELASTIC_point"][()])
            a = f["point"].attrs
            t = a["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / a["sampling_rate_in_hertz"][0]
        out[name] = v[order][:, 2, :].T.astype(np.float32)  # vz (nt, nrec)
        out[f"t__{name}"] = t  # time step differs between models (steel sets the CFL limit)
    np.savez_compressed(OUT / "salvus_ldv_cube.npz", rec_idx=rec_idx, **out)
    (OUT / "salvus_ldv_cube.json").write_text(json.dumps(dict(
        origin=origin.tolist(), Lx=Lx, Ly=Ly, element=EL, plate=PMMA, steel=STEEL, cube_y=CUBE_Y,
        cube_xs=CUBE_XS, source=src.tolist(), f_src=F_SRC, end_time=END_TIME), indent=1))
    print("wrote", OUT / "salvus_ldv_cube.npz")


if __name__ == "__main__":
    main()
