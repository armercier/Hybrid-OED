"""Salvus 'LDV measurement' at given scan-frame points (Salvus venv).

Adds one event (the transducer shot) with receivers at the requested points to the LDV
Salvus project (ldv_assessment/salvus_project), runs the given configuration on the GPU
and writes vz traces: npz with t (s) and vz (nt, nrec).

  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/oed_ldv/acquire_salvus_ldv.py \
      --receivers rec.npy --name oed_r0 --config cube_x-20mm_Q40 --out data.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "ldv_assessment" / "salvus_project"
META = json.loads((REPO / "ldv_assessment" / "salvus_ldv_cube.json").read_text())
H = 0.010


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receivers", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="cube_x-20mm_Q40")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    origin = np.asarray(META["origin"])
    rec = np.load(args.receivers) - origin
    src = np.asarray(META["source"]) - origin
    p = sn.Project(path=PROJECT)
    if args.name not in p.events.list():
        p.add_to_project(sn.Event(
            event_name=args.name,
            sources=sn.simple_config.source.cartesian.VectorPoint3D(x=float(src[0]), y=float(src[1]), z=H,
                                                                     fx=0.0, fy=0.0, fz=-1.0),
            receivers=[sn.simple_config.receiver.cartesian.Point3D(x=float(x), y=float(y), z=H,
                                                                   station_code=f"R{k:04d}", fields=["velocity"])
                       for k, (x, y) in enumerate(rec)]))
    p.simulations.launch(simulation_configuration=args.config, events=[args.name], site_name="local_gpu",
                         ranks_per_job=1, verbosity=0)
    p.simulations.query(block=True)
    with h5py.File(Path(p.simulations.get_simulation_output_directory(args.config, args.name)) / "receivers.h5") as f:
        v = f["point/velocity"][()]
        order = np.argsort(f["names_ELASTIC_point"][()])
        a = f["point"].attrs
        t = a["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / a["sampling_rate_in_hertz"][0]
        coords = f["coordinates_ELASTIC_point"][()][order, :2]
    assert np.allclose(coords, rec, atol=1e-6), "receiver order mismatch"
    np.savez(args.out, t=t, vz=v[order][:, 2, :].T.astype(np.float32))


if __name__ == "__main__":
    main()
