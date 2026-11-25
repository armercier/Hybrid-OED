"""Timing settings for a multi-source LDV scan driven by two independent programs.

The excitation script (PlaySamplesArnaud.py) fires one source per LDV scan on its own clock; the Polytec
macro (Auto_Arnaud.wwb) runs the scans one after the other. They only share the trigger. The LDV records
a point only when a trigger arrives, so the schedule is safe when:

  1. every scan finishes before the script switches the trigger off:      real scan <= D + M
  2. the LDV restarts while the trigger is off (it then waits for the next
     source and starts exactly with it):                  D + M <= real scan + W <= D + M + G

  D  = expected scan time,  M = safety_margin (trigger kept on after the expected end),
  G  = delay_before_next_trigger (trigger off between sources),  W = LDV Wait before the next scan.

With W = M + G/2 (restart in the middle of the gap) the real scan time may be off by
-G/2 ... +min(M, G/2). Choosing M = G/2 = f * uncertainty gives a safety factor f on the uncertainty.
Because the LDV re-synchronises on the trigger at every switch, the error does not add up from scan
to scan: the tolerance holds for each scan separately.

Usage examples:
  python src/ldv_pmma/scan_schedule.py --scan-time 25700 --uncertainty 1800
  python src/ldv_pmma/scan_schedule.py --points 6418 --sec-per-point 4.0 --uncertainty-pct 5
  options: --sources 4  --factor 2  --min-gap 600   (min-gap: time the LDV needs to save a file and re-arm)
Start order: start the LDV macro first (it waits for the trigger), then the Python script.
"""
from __future__ import annotations

import argparse
import math


def schedule(scan_time, uncertainty, n_sources=4, factor=2.0, min_gap=600.0, round_to=60.0):
    """Return the settings (s) for an expected scan time and its +/- uncertainty (s)."""
    up = lambda x: round_to * math.ceil(x / round_to)
    half_gap = up(max(factor * uncertainty, min_gap / 2))
    margin = up(max(factor * uncertainty, 0.0))
    gap = 2 * half_gap
    wait = margin + half_gap
    tol_minus, tol_plus = half_gap, min(margin, half_gap)
    total = n_sources * (scan_time + margin) + (n_sources - 1) * gap
    return dict(scan_time=float(scan_time), uncertainty=float(uncertainty), factor=factor, safety_margin=float(margin),
                delay_before_next_trigger=float(gap), ldv_wait=float(wait), tolerance_minus=float(tol_minus),
                tolerance_plus=float(tol_plus), total_duration=float(total), n_sources=n_sources)


def check(s, real_scan):
    """True if a scan of duration real_scan is recorded entirely with its own source."""
    D, M, G, W = s["scan_time"], s["safety_margin"], s["delay_before_next_trigger"], s["ldv_wait"]
    return real_scan <= D + M and D + M <= real_scan + W <= D + M + G


def hm(x):
    return f"{int(x // 3600)} h {int(round(x % 3600 / 60)):02d} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan-time", type=float, help="expected duration of one LDV scan (s)")
    g.add_argument("--points", type=int, help="number of scan points (with --sec-per-point)")
    ap.add_argument("--sec-per-point", type=float, help="measured time per point (s)")
    u = ap.add_mutually_exclusive_group(required=True)
    u.add_argument("--uncertainty", type=float, help="+/- uncertainty on the scan time (s)")
    u.add_argument("--uncertainty-pct", type=float, help="+/- uncertainty on the scan time (%%)")
    ap.add_argument("--sources", type=int, default=4)
    ap.add_argument("--factor", type=float, default=2.0, help="safety factor on the uncertainty (default 2)")
    ap.add_argument("--min-gap", type=float, default=600.0, help="minimum trigger-off gap (s), LDV save + re-arm")
    a = ap.parse_args()
    D = a.scan_time if a.scan_time is not None else a.points * a.sec_per_point
    if a.points is not None and a.sec_per_point is None:
        ap.error("--points needs --sec-per-point")
    dU = a.uncertainty if a.uncertainty is not None else D * a.uncertainty_pct / 100
    s = schedule(D, dU, a.sources, a.factor, a.min_gap)
    for real in (D - s["tolerance_minus"], D, D + s["tolerance_plus"]):
        assert check(s, real), "internal check failed"
    print(f"Expected scan time {D:.0f} s ({hm(D)}), uncertainty +/- {dU:.0f} s, safety factor {a.factor:g}\n")
    print("Python excitation script (PlaySamplesArnaud.py):")
    print(f"    scan_time = {D:.0f}")
    print(f"    safety_margin = {s['safety_margin']:.1f}")
    print(f"    delay_before_next_trigger = {s['delay_before_next_trigger']:.0f}")
    print("\nPolytec macro (Auto_Arnaud.wwb), the Wait before the next scan:")
    print(f"    Wait {s['ldv_wait']:.0f}")
    print(f"\nEach scan may last between {D - s['tolerance_minus']:.0f} and {D + s['tolerance_plus']:.0f} s "
          f"(-{s['tolerance_minus']:.0f} / +{s['tolerance_plus']:.0f} s) without recording a wrong source.")
    print(f"Total duration with {a.sources} sources: {hm(s['total_duration'])}.")
    print("Start the LDV macro first, then the Python script.")


if __name__ == "__main__":
    main()
