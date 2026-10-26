"""Export the Salvus plate benchmark to one npz readable without Salvus.

Run in the Salvus 2025.1.2 venv (matching the solver binary):
  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/salvus_plate/export_data.py

Output: salvus_plate/plate_benchmark.npz with
  t (nt,)                      sample times in s (Ricker peak at t = 0)
  receivers_m (nrec, 3)        receiver coordinates (x, y, z) in m
  <config>__<event> (nt, nrec, 3)  particle velocity (vx, vy, vz) in m/s
and salvus_plate/benchmark_definition.json (written by build_and_run.py).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "salvus_plate" / "project"
OUT = REPO / "salvus_plate" / "plate_benchmark.npz"


def main() -> None:
    p = sn.Project(path=PROJECT)
    arrays = {}
    reference = None
    for cfg in p.entities.list("simulation_configuration"):
        for event in p.events.list():
            path = Path(p.simulations.get_simulation_output_directory(cfg, event)) / "receivers.h5"
            with h5py.File(path, "r") as f:
                v = f["point/velocity"][()]  # (nrec, 3, nt)
                names = f["names_ELASTIC_point"][()]
                coords = f["coordinates_ELASTIC_point"][()]
                attrs = f["point"].attrs
                t = attrs["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / attrs["sampling_rate_in_hertz"][0]
            order = np.argsort(names)  # station codes R0000.. follow the receiver grid order
            if reference is None:
                reference = dict(t=t, receivers_m=coords[order], names=names[order])
            assert np.array_equal(names[order], reference["names"]), (cfg, event)
            arrays[f"{cfg}__{event}"] = np.transpose(v[order], (2, 0, 1)).astype(np.float32)
            print(f"{cfg} {event}: {v.shape}")
    np.savez_compressed(OUT, t=reference["t"], receivers_m=reference["receivers_m"], **arrays)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
