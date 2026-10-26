"""Salvus 3-D plate benchmark with every parameter known (run in the ``salvus`` env).

Free plate (no absorbing boundaries) of PMMA, 0.6 m x 0.6 m x 10 mm, coordinates
x, y in [0, 0.6] m (plate centre at 0.3, 0.3), z in [0, 0.01] m.

* Mesh: 60 x 60 x 1 hexahedra of 10 mm, spectral order 4.
* Inclusion: 60 mm square through the full thickness, centred at (0.25, 0.35) m,
  i.e. (-0.05, +0.05) m from the plate centre, edges on element boundaries;
  vs 1100 m/s, vp 2240 m/s (same vp/vs), same density.
* Sources: 12 in-plane point forces at mid-plane (symmetric excitation: S0 + SH0
  only), Ricker 25 kHz (Salvus default: peak at t = 0), on the four sides 0.2 m
  from the centre, offsets -0.1 / 0 / +0.1 m, force transverse to the side.
* Receivers: 59 x 59 grid, 10 mm spacing, on the top surface, 3-component velocity.
* Two simulations: with the inclusion and homogeneous.

Usage:
  /home/merciera/miniconda3/envs/salvus/bin/python src/salvus_plate/build_and_run.py \
      [--events event_00] [--site local_gpu] [--ranks 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import salvus.namespace as sn
from salvus.mesh.simple_mesh import basic_mesh

REPO = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO / "salvus_plate" / "project"

L, H = 0.6, 0.01
CENTRE = np.array([L / 2, L / 2])
PMMA = dict(vp=2730.0, vs=1340.0, rho=1180.0)
INCLUSION = dict(vp=2240.0, vs=1100.0, rho=1180.0, centre=(0.25, 0.35), side=0.06)
ELEMENT_SIZE = 0.01
ORDER = 4
F0 = 25e3
END_TIME = 1.0e-3
RECEIVER_SPACING = 0.01
SOURCE_SIDE, SOURCE_OFFSETS = 0.2, (-0.1, 0.0, 0.1)


def source_layout():
    """(position relative to the centre, force direction) in the order x0, y0, x1, y1."""
    out = []
    for side, axis, sign in (("x0", 0, -1), ("y0", 1, -1), ("x1", 0, 1), ("y1", 1, 1)):
        for o in SOURCE_OFFSETS:
            pos = np.zeros(2)
            pos[axis], pos[1 - axis] = sign * SOURCE_SIDE, o
            force = np.zeros(2)
            force[1 - axis] = 1.0  # transverse to the side
            out.append((side, pos, force))
    return out


def receiver_grid():
    g = np.arange(RECEIVER_SPACING, L - RECEIVER_SPACING / 2, RECEIVER_SPACING)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def build_mesh(with_inclusion: bool) -> sn.UnstructuredMesh:
    # vs / (f * epw) = ELEMENT_SIZE  ->  60 x 60 x 1 elements of 10 mm
    epw = 2.0
    mesh = basic_mesh.CartesianHomogeneousIsotropicElastic3D(
        vp=PMMA["vp"], vs=PMMA["vs"], rho=PMMA["rho"], x_max=L, y_max=L, z_max=H,
        max_frequency=PMMA["vs"] / (epw * ELEMENT_SIZE), elements_per_wavelength=epw,
        tensor_order=ORDER,
    ).create_mesh()
    if with_inclusion:
        c = mesh.get_element_centroid()
        half = INCLUSION["side"] / 2
        inside = (np.abs(c[:, 0] - INCLUSION["centre"][0]) < half) & (
            np.abs(c[:, 1] - INCLUSION["centre"][1]) < half)
        for name, key in (("VP", "vp"), ("VS", "vs"), ("RHO", "rho")):
            field = mesh.element_nodal_fields[name].copy()
            field[inside] = INCLUSION[key]
            mesh.attach_field(name, field)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", default=None, help="default: all 12")
    parser.add_argument("--configs", nargs="*", default=["plate_inclusion", "plate_homogeneous"])
    parser.add_argument("--site", default="local_gpu")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--sampling", type=int, default=4, help="receiver sampling in time steps")
    args = parser.parse_args()

    PROJECT_DIR.parent.mkdir(parents=True, exist_ok=True)
    p = sn.Project.from_domain(
        path=PROJECT_DIR,
        domain=sn.domain.dim3.BoxDomain(x0=0.0, x1=L, y0=0.0, y1=L, z0=0.0, z1=H),
        load_if_exists=True,
    )

    receivers = [
        sn.simple_config.receiver.cartesian.Point3D(
            x=float(x), y=float(y), z=H, station_code=f"R{i:04d}", fields=["velocity"])
        for i, (x, y) in enumerate(receiver_grid())
    ]
    existing = set(p.events.list())
    names = []
    for k, (side, pos, force) in enumerate(source_layout()):
        name = f"event_{k:02d}"
        names.append(name)
        if name in existing:
            continue
        x, y = CENTRE + pos
        p.add_to_project(sn.Event(
            event_name=name,
            sources=sn.simple_config.source.cartesian.VectorPoint3D(
                x=float(x), y=float(y), z=H / 2, fx=float(force[0]), fy=float(force[1]), fz=0.0),
            receivers=receivers,
        ))

    wavelet = sn.simple_config.stf.Ricker(center_frequency=F0)
    existing_cfg = set(p.entities.list("simulation_configuration"))
    for cfg in args.configs:
        if cfg in existing_cfg:
            continue
        p.add_to_project(sn.UnstructuredMeshSimulationConfiguration(
            name=cfg,
            unstructured_mesh=build_mesh(with_inclusion=(cfg == "plate_inclusion")),
            event_configuration=sn.EventConfiguration(
                wavelet=wavelet,
                waveform_simulation_configuration=sn.WaveformSimulationConfiguration(
                    start_time_in_seconds=wavelet.get_auto_start_time(),
                    end_time_in_seconds=END_TIME,
                    spectral_element_order=ORDER,
                    receiver_sampling_rate_in_time_steps=args.sampling,
                ),
            ),
        ))

    meta = dict(L=L, H=H, pmma=PMMA, inclusion=INCLUSION, element_size=ELEMENT_SIZE, order=ORDER,
                f0=F0, start_time=wavelet.get_auto_start_time(), end_time=END_TIME,
                receiver_spacing=RECEIVER_SPACING,
                sources=[dict(event=f"event_{k:02d}", side=s, position=(CENTRE + pos).tolist(),
                              force=f.tolist(), z=H / 2) for k, (s, pos, f) in enumerate(source_layout())])
    (PROJECT_DIR.parent / "benchmark_definition.json").write_text(json.dumps(meta, indent=2))

    events = args.events or names
    for cfg in args.configs:
        p.simulations.launch(simulation_configuration=cfg, events=events, site_name=args.site,
                             ranks_per_job=args.ranks)
    p.simulations.query(block=True)


if __name__ == "__main__":
    main()
