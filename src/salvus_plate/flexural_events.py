"""Add out-of-plane (flexural) force events to the Salvus plate benchmark and export them.

Normal point forces (fz = 1) at the positions of benchmark events 1 and 4, either at
mid-plane (excites only antisymmetric modes: A0) or on the top surface (like a
transducer: A0 plus a weaker symmetric part). Both models (homogeneous, inclusion)
are run on the GPU; output salvus_plate/flexural_benchmark.npz with keys
<config>__flex_<mid|top>_e<NN> (nt, nrec, 3), t, receivers_m.

  /home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python src/salvus_plate/flexural_events.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import salvus.namespace as sn

from build_and_run import CENTRE, H, PROJECT_DIR, receiver_grid, source_layout

OUT = Path(PROJECT_DIR).parent / "flexural_benchmark.npz"
EVENTS = (1, 4)
DEPTHS = {"mid": H / 2, "top": H}
CONFIGS = ("plate_homogeneous", "plate_inclusion")


def main() -> None:
    p = sn.Project(path=PROJECT_DIR)
    receivers = [sn.simple_config.receiver.cartesian.Point3D(
        x=float(x), y=float(y), z=H, station_code=f"R{i:04d}", fields=["velocity"])
        for i, (x, y) in enumerate(receiver_grid())]
    names, existing = [], set(p.events.list())
    for e in EVENTS:
        _, pos, _ = source_layout()[e]
        x, y = CENTRE + pos
        for label, zs in DEPTHS.items():  # noqa: B007
            name = f"flex_{label}_e{e:02d}"  # names are launched sorted (see below)
            names.append(name)
            if name not in existing:
                p.add_to_project(sn.Event(event_name=name, receivers=receivers,
                                          sources=sn.simple_config.source.cartesian.VectorPoint3D(
                                              x=float(x), y=float(y), z=zs, fx=0.0, fy=0.0, fz=1.0)))
    # Launch with sorted event names: an unsorted list gave results stored under permuted
    # event names; each dataset's source location is verified before export below.
    names = sorted(set(names))
    for cfg in CONFIGS:
        p.simulations.launch(simulation_configuration=cfg, events=names, site_name="local_gpu",
                             ranks_per_job=1, verbosity=0)
    p.simulations.query(block=True)
    out = {}
    for cfg in CONFIGS:
        for name in names:
            with h5py.File(Path(p.simulations.get_simulation_output_directory(cfg, name)) / "receivers.h5") as f:
                v = f["point/velocity"][()]
                order = np.argsort(f["names_ELASTIC_point"][()])
                a = f["point"].attrs
                t = a["start_time_in_seconds"][0] + np.arange(v.shape[-1]) / a["sampling_rate_in_hertz"][0]
                rec = f["coordinates_ELASTIC_point"][()][order]
            vz_peak = np.abs(v[order][:, 2, :]).max(1)
            e = int(name[-2:])
            expected = CENTRE + source_layout()[e][1]
            nearest = rec[np.argmax(vz_peak), :2]
            assert np.hypot(*(nearest - expected)) < 0.02, (cfg, name, nearest, expected)
            out[f"{cfg}__{name}"] = np.transpose(v[order], (2, 0, 1)).astype(np.float32)
    np.savez_compressed(OUT, t=t, receivers_m=rec, **out)
    print("wrote", OUT, sorted(out))


if __name__ == "__main__":
    main()
