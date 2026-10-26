"""Run the paper's multi-seed full-autodiff experiment with rematerialised FWI updates.

Thin wrapper around ``run_oed_multiseed_full_autodiff.main`` (imported unchanged):
the module's ``Simulation`` is replaced by ``RematSimulation`` before ``main`` runs,
so the protocol (seeds, pairing, checkpoints, metrics, outputs) is the paper's.
Only the oracle and waveform-misfit hypergradients are affected; baseline and
illumination never differentiate through FWI and run the identical code.

Each method runs in its own process (``--resume``), so the GPU peak memory
recorded in ``<run_root>/gpu_memory/`` is per method.  Use ``--device auto``
(CUDA is picked up); the paper flag ``--device gpu`` requests the ROCm backend
in this environment.

Usage (paper seeds, GPU):
  python src/remat/run_multiseed_remat.py --num-repetitions 10 \
      --output-dir remat_results/multiseed_gpu
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHOD_ORDER = ("baseline", "illumination", "waveform_misfit", "oracle")


def run_one(argv: list[str]) -> int:
    """Child process: run the paper ``main`` with ``RematSimulation`` swapped in."""
    for p in (REPO / "src", REPO / "src" / "scripts", HERE):
        sys.path.insert(0, str(p))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import run_oed_multiseed_full_autodiff as paper
    from remat_hypergrad import make_remat_class

    paper.Simulation = make_remat_class()
    started = time.perf_counter()
    status = paper.main(argv)

    import jax

    stats = jax.devices()[0].memory_stats() or {}
    args = paper.parse_args(argv)
    mode = "smoke-test" if args.smoke_test else "scientific"
    out = args.output_dir.resolve() / mode / "gpu_memory"
    out.mkdir(parents=True, exist_ok=True)
    methods = "+".join(args.methods)
    (out / f"{methods}.json").write_text(json.dumps({
        "device": str(jax.devices()[0]),
        "methods": args.methods,
        "peak_device_gib": stats.get("peak_bytes_in_use", 0) / 2**30,
        "process_wall_s": time.perf_counter() - started,
        "status": status,
    }, indent=2))
    return status


def main() -> int:
    if "--child" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--child"]
        return run_one(argv)
    passthrough = sys.argv[1:]
    if "--device" not in passthrough:
        passthrough += ["--device", "auto"]
    status = 0
    for method in METHOD_ORDER:
        cmd = [sys.executable, __file__, "--child", *passthrough, "--methods", method, "--resume"]
        print("$", " ".join(cmd[1:]), flush=True)
        status |= subprocess.call(cmd)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
