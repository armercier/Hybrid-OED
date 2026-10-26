"""Shared geometry and paths for the PINN feasibility study (step 1).

Nothing here modifies the paper code: the paper ``Simulation`` is imported
read-only to obtain the exact grid, sources and true model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "src" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = REPO / "pinn_feasibility_results"
REFERENCE_DIR = RESULTS / "reference"

# Target-oriented box centred on the inclusion (metres).  All four sources lie
# outside it, so the incident field u0 is smooth inside.
BOX_CENTER = np.array([-0.05, 0.05])
BOX_HALF = 0.25
C0 = 2100.0
T_SCALE = BOX_HALF / C0  # time unit so that c0 = 1 in normalised coordinates
F0 = 25e3


def eval_receivers_m() -> dict[str, np.ndarray]:
    """Evaluation receivers: rings around the inclusion and the paper's nominal diagonal."""
    radii = np.array([0.08, 0.14, 0.20])
    angles = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    ring = np.array(
        [BOX_CENTER + r * np.array([np.cos(a), np.sin(a)]) for r in radii for a in angles]
    )
    diag = np.stack([np.linspace(-0.15, 0.15, 30)] * 2, axis=1)
    return {"ring": ring, "diag": diag}


def to_normalised(xy_m: np.ndarray, t_s: np.ndarray | None = None):
    xy = (np.asarray(xy_m) - BOX_CENTER) / BOX_HALF
    return xy if t_s is None else (xy, np.asarray(t_s) / T_SCALE)
