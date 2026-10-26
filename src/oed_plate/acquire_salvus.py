"""'Lab measurement': simulate the true Salvus plate for given receiver positions.

Plays the role of the LDV in the OED loop. Adds one Salvus event per source with
the requested receivers (top surface, velocity), runs the existing
``plate_inclusion`` configuration of the benchmark project on the GPU, and writes
the traces. Run with the Salvus 2025.1.2 venv:

  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/oed_plate/acquire_salvus.py \
      --receivers rec.npy --events 1 4 7 10 --name oed_illum_r0 --out data_r0.npz

Output npz: t (nt,), receivers_m (nrec, 2), and v_<event index> (nt, nrec, 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "salvus_plate"))
from build_and_run import CENTRE, H, PROJECT_DIR, source_layout  # noqa: E402

CONFIG = "plate_inclusion"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receivers", type=Path, required=True, help=".npy (nrec, 2) positions in m")
    parser.add_argument("--events", type=int, nargs="+", required=True)
    parser.add_argument("--name", required=True, help="unique acquisition name")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--site", default="local_gpu")
    args = parser.parse_args()

    rec = np.load(args.receivers)
    p = sn.Project(path=PROJECT_DIR)
    layout = source_layout()
    receivers = [sn.simple_config.receiver.cartesian.Point3D(
        x=float(x), y=float(y), z=H, station_code=f"R{i:04d}", fields=["velocity"])
        for i, (x, y) in enumerate(rec)]
    names = []
    existing = set(p.events.list())
    for e in args.events:
        _, pos, force = layout[e]
        name = f"{args.name}_e{e:02d}"
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
    p.simulations.launch(simulation_configuration=CONFIG, events=names, site_name=args.site,
                         ranks_per_job=1, verbosity=0)
    p.simulations.query(block=True)

    out = {}
    for e, name in zip(args.events, names):
        path = Path(p.simulations.get_simulation_output_directory(CONFIG, name)) / "receivers.h5"
        with h5py.File(path, "r") as f:
            v = f["point/velocity"][()]
            order = np.argsort(f["names_ELASTIC_point"][()])
            attrs = f["point"].attrs
            t = attrs["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / attrs["sampling_rate_in_hertz"][0]
            coords = f["coordinates_ELASTIC_point"][()][order, :2]
        assert np.allclose(coords, rec, atol=1e-6), "receiver order mismatch"
        out[f"v_{e}"] = np.transpose(v[order], (2, 0, 1)).astype(np.float32)
    np.savez(args.out, t=t, receivers_m=rec, **out)
    print(f"acquired {len(names)} events x {len(rec)} receivers -> {args.out}")


if __name__ == "__main__":
    main()
