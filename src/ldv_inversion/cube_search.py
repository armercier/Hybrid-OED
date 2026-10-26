"""Stage A: locate the bonded cube from the LDV data with the rigid-body Mindlin model.

Grid search of the cube centre (coarse 2 mm, then 1 mm around the best), rigid 6 mm
effective footprint, the steel cube's physical mass and rotary inertia (calibration from
src/ldv_assessment/mindlin_equivalent.py). Objective: held-out misfit (train-fitted
source filter, test points behind + before the cube). The homogeneous plate is the
reference.

Usage:  python src/ldv_inversion/cube_search.py
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

from ldv_problem import CUBE_SALVUS, OUT, LDVProblem

import jax

M_CUBE = 7850.0 * 0.010**3
J_CUBE = M_CUBE * (0.010**2 / 12 + 0.010**2)
FOOTPRINT = 0.006


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prob = LDVProblem()
    idx = np.concatenate([prob.train, prob.test])
    rec = prob.receivers(idx)
    run = jax.jit(lambda mask: prob.simulate(rec, rigid_mask=mask, rigid_mass=M_CUBE, rigid_inertia=J_CUBE))
    run_h = jax.jit(lambda: prob.simulate(rec))

    def score(c):
        mask = (np.abs(prob.X - c[0]) <= FOOTPRINT / 2) & (np.abs(prob.Y - c[1]) <= FOOTPRINT / 2)
        m, _ = prob.evaluate(run(jax.numpy.asarray(mask)), idx)
        return m

    started = time.perf_counter()
    base, _ = prob.evaluate(run_h(), idx)
    print(f"homogeneous plate: behind {base['behind']:.3f} before {base['before']:.3f}", flush=True)
    results = []

    def search(centres, label):
        for c in centres:
            m = score(c)
            results.append(dict(x=float(c[0]), y=float(c[1]), **m, total=m["behind"] + m["before"], stage=label))
            print(f"{label} cube at ({c[0]*1e3:+.0f}, {c[1]*1e3:+.0f}) mm: behind {m['behind']:.3f} "
                  f"before {m['before']:.3f}  t={time.perf_counter()-started:.0f}s", flush=True)

    coarse = [CUBE_SALVUS + np.array([dx, dy]) for dx, dy in itertools.product((-0.004, -0.002, 0, 0.002, 0.004),
                                                                                 (-0.004, -0.002, 0, 0.002, 0.004))]
    search(coarse, "coarse")
    best = min(results, key=lambda r: r["total"])
    fine = [np.array([best["x"], best["y"]]) + np.array([dx, dy]) for dx, dy in itertools.product((-0.001, 0, 0.001), repeat=2)
            if (dx, dy) != (0, 0)]
    search(fine, "fine")
    best = min(results, key=lambda r: r["total"])
    print("best:", best)
    (OUT / "cube_search.json").write_text(json.dumps(dict(homogeneous=base, best=best, results=results,
                                                          footprint=FOOTPRINT, salvus_position=CUBE_SALVUS.tolist()),
                                                     indent=1))


if __name__ == "__main__":
    main()
