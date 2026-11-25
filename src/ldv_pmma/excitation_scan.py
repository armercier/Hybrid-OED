"""Multi-source excitation for an LDV area scan, with the timing computed from the scan time.

Self-contained replacement for PlaySamplesArnaud.py. Set the SETTINGS block below (or use the
command-line options), run it, set the printed Wait in the Polytec macro, start the LDV macro first,
then confirm here.

How the timing works (see ldv_scan_schedule_sheet.pdf): the script and the LDV macro run on separate
clocks and only share the trigger; the LDV records a point only when a trigger arrives. Each source
stays on for scan_time + safety_margin, then the trigger is off for delay_before_next_trigger. The LDV
macro waits Wait = safety_margin + delay_before_next_trigger / 2 after each scan, so it restarts in the
middle of the trigger-off gap and starts exactly with the next source. A scan may then be up to
+/- delay_before_next_trigger / 2 (and at most +safety_margin) off from scan_time without recording
the wrong source, and the error does not accumulate from scan to scan.

Usage:
  python excitation_scan.py                                    # uses the SETTINGS block
  python excitation_scan.py --points 6418 --sec-per-point 4.0 --uncertainty-pct 5
  python excitation_scan.py --scan-time 25700 --uncertainty 1800
  python excitation_scan.py ... --dry-run                     # print the plan, no hardware
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timedelta

import numpy as np

# ============================== SETTINGS (edit here) ==============================
SCAN_TIME = None            # expected LDV scan time (s); or leave None and give POINTS and SEC_PER_POINT
POINTS = 6418               # number of scan points
SEC_PER_POINT = 4.0         # measured time per point (s), from a short test scan with the same settings
UNCERTAINTY_PCT = 5.0       # +/- uncertainty on the scan time (%)  (or set UNCERTAINTY_S below)
UNCERTAINTY_S = None        # +/- uncertainty on the scan time (s); overrides UNCERTAINTY_PCT if set
SAFETY_FACTOR = 2.0         # margin and half-gap = SAFETY_FACTOR x uncertainty
MIN_GAP = 600.0             # minimum trigger-off gap (s): time for the LDV to save the file and re-arm
N_CHANNELS = 4              # sources, fired in order: channel 0 (S1), 1 (S2), ...
KEEP_S1_ON = False          # False: one source per file. True: channel 0 stays on in every file (as in the
                            # Nov 2026 pristine and defect scans, which were S1 + Sk)
PRINT_ONLY = True           # True: only print the timing values (Wait for the macro, schedule) and exit,
                            # nothing is sent to the hardware. Set to False to run the excitation.
HOST, TRIGGER_PORT, GENERATOR_PORT = "172.31.190.24", 2059, 2057
HARDWARE_PATH = "/root/All_codes/ExperimentsOptimizedGeometries"   # folder of OfflineExperiments.py (as in PlaySamplesArnaud.py)

# Wavelet (as in PlaySamplesArnaud.py)
T_MAX = 2e-3                # length of the drive record (s)
N_SAMPLES = 2500
T0 = 0.5e-3                 # Ricker centre after the trigger (s)
F0 = 40e3                   # centre frequency (Hz)
AMPLITUDE = 1.5
# ==================================================================================


def schedule(scan_time, uncertainty, factor=SAFETY_FACTOR, min_gap=MIN_GAP, n=N_CHANNELS, round_to=60.0):
    up = lambda x: round_to * math.ceil(x / round_to)
    half_gap = up(max(factor * uncertainty, min_gap / 2))
    margin = up(factor * uncertainty)
    gap = 2 * half_gap
    return dict(scan_time=scan_time, uncertainty=uncertainty, safety_margin=margin, delay_before_next_trigger=gap,
                ldv_wait=margin + half_gap, tol_minus=half_gap, tol_plus=min(margin, half_gap),
                total=n * (scan_time + margin) + (n - 1) * gap)


def hm(sec):
    sec = int(round(sec))
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min"


def wavelet():
    t = np.linspace(0, T_MAX, N_SAMPLES)
    a = (np.pi * F0 * (t - T0)) ** 2
    return AMPLITUDE * (1 - 2 * a) * np.exp(-a)


def sleep_with_progress(seconds, label):
    """Sleep in short chunks (Ctrl+C stays responsive) and print the remaining time every 30 min."""
    end = time.time() + seconds
    next_print = time.time() + 1800
    while True:
        left = end - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 10.0))
        if time.time() >= next_print and end - time.time() > 60:
            print(f"   {datetime.now():%a %H:%M}  {label}: {hm(end - time.time())} left", flush=True)
            next_print += 1800


def main():
    ap = argparse.ArgumentParser(description="Multi-source excitation for an LDV scan")
    ap.add_argument("--scan-time", type=float, help="expected scan time (s)")
    ap.add_argument("--points", type=int)
    ap.add_argument("--sec-per-point", type=float)
    ap.add_argument("--uncertainty", type=float, help="+/- uncertainty (s)")
    ap.add_argument("--uncertainty-pct", type=float, help="+/- uncertainty (%%)")
    ap.add_argument("--channels", type=int, default=N_CHANNELS)
    ap.add_argument("--keep-s1-on", action="store_true", default=KEEP_S1_ON)
    ap.add_argument("--dry-run", action="store_true", help="print the plan only, no hardware")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before starting")
    a = ap.parse_args()

    if a.scan_time is not None:
        D = a.scan_time
    elif SCAN_TIME is not None and a.points is None:
        D = SCAN_TIME
    else:
        D = (a.points or POINTS) * (a.sec_per_point or SEC_PER_POINT)
    if a.uncertainty is not None:
        dD = a.uncertainty
    elif a.uncertainty_pct is not None:
        dD = D * a.uncertainty_pct / 100
    else:
        dD = UNCERTAINTY_S if UNCERTAINTY_S is not None else D * UNCERTAINTY_PCT / 100
    n = a.channels
    s = schedule(D, dD, n=n)
    on_time, gap = D + s["safety_margin"], s["delay_before_next_trigger"]

    start = datetime.now()
    print("=" * 70)
    print(f"Expected scan time {D:.0f} s ({hm(D)}), uncertainty +/- {dD:.0f} s")
    print(f"Each source on {on_time:.0f} s, then trigger off {gap:.0f} s; {n} sources"
          + ("; channel 0 (S1) stays on in every file" if a.keep_s1_on else "; one source per file"))
    print(f"Safe if each real scan lasts {D - s['tol_minus']:.0f} - {D + s['tol_plus']:.0f} s")
    print(f"Total: {hm(s['total'])}, ends around {start + timedelta(seconds=s['total']):%a %d %b %H:%M} if started now")
    print("-" * 70)
    print(f">>> In the Polytec macro (Auto_Arnaud.wwb, line 45) set:   Wait {s['ldv_wait']:.0f}")
    print("-" * 70)
    t = 0.0
    for j in range(n):
        print(f"   S{j + 1}: trigger on  +{hm(t):>12s}  ->  off +{hm(t + on_time):>12s}")
        t += on_time + (gap if j < n - 1 else 0)
    print("=" * 70)
    if a.dry_run or PRINT_ONLY:
        print("PRINT_ONLY = True (or --dry-run): values printed, nothing sent to the hardware. "
              "Set PRINT_ONLY = False in the SETTINGS block to run the excitation.")
        return

    # ---- hardware: same import as PlaySamplesArnaud.py (lines 8-9), provides Trigger and SignalGenerator ----
    sys.path.append(HARDWARE_PATH)
    from OfflineExperiments import SignalGenerator, Trigger  # noqa: E402

    if not a.yes:
        input("Set the Wait in the macro, START THE LDV MACRO FIRST, then press Enter here to begin... ")
    trigger = Trigger(HOST, TRIGGER_PORT)
    generator = SignalGenerator(HOST, GENERATOR_PORT)
    signal = wavelet()
    zeros = np.zeros(len(signal))
    log_name = f"excitation_log_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(log_name, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["time", "event", "channel"])
        try:
            trigger.disableTrigger()
            for j in range(n):
                for ch in range(n):                       # all channels to zero (channel 0 kept if asked)
                    if ch == 0 and a.keep_s1_on and j > 0:
                        continue
                    generator.setChannelData(ch, zeros)
                    time.sleep(0.5)
                generator.setChannelData(j, signal)
                time.sleep(0.5)
                trigger.enableTrigger()
                print(f"{datetime.now():%a %H:%M:%S}  channel {j} (S{j + 1}) ON for {on_time:.0f} s", flush=True)
                log.writerow([datetime.now().isoformat(timespec="seconds"), "trigger_on", j]); fh.flush()
                sleep_with_progress(on_time, f"S{j + 1}")
                trigger.disableTrigger()
                print(f"{datetime.now():%a %H:%M:%S}  trigger OFF", flush=True)
                log.writerow([datetime.now().isoformat(timespec="seconds"), "trigger_off", j]); fh.flush()
                if j < n - 1:
                    sleep_with_progress(gap, "gap (LDV restarts and waits for the trigger)")
        finally:
            trigger.disableTrigger()
            for ch in range(n):
                generator.setChannelData(ch, zeros)
                time.sleep(0.5)
            log.writerow([datetime.now().isoformat(timespec="seconds"), "end", ""])
            print(f"{datetime.now():%a %H:%M:%S}  done: trigger off, all channels zero. Log: {log_name}", flush=True)


if __name__ == "__main__":
    main()
