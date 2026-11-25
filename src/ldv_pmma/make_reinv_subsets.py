"""Subsets for the re-inversion test (21 Nov): the final 30 measured points of an adaptive run (result.npz 'measured'),
written as a static subset so they can be inverted exactly like the static layouts (100 updates from the blind background).
Separates the effect of the point positions (design) from the progressive 33/33/34 inversion of the adaptive protocol.

Usage:  python src/ldv_pmma/make_reinv_subsets.py TAG [TAG ...]
Output: ldv_work/prepared/subsets/ri_<TAG>.npz
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for tag in sys.argv[1:]:
    m = np.load(REPO / "ldv_pmma_results" / "inversion" / tag / "result.npz")["measured"].astype(bool)
    out = REPO / "ldv_work" / "prepared" / "subsets" / f"ri_{tag}.npz"
    np.savez(out, keep=m, nosurface=np.zeros_like(m), rule=np.array(f"final measured points of adaptive run {tag}"))
    print(out, int(m.sum()))
