---
title: "Why the pristine LDV scan mixed up sources"
date: "14 November 2026"
geometry: margin=2cm
---

The pristine PMMA scan of 13–14 Nov 2026 is usable, but it has two acquisition bugs: source S1 fires in every file, and part of files 2–4 was recorded while the previous source was still firing. Both come from the timing and channel logic of the excitation script, not from the LDV or the plate.

![The two bugs at a glance: (A1) timeline of the pristine night, (A2) the fixed schedule of the defect scan, (B) the channel-zeroing bug, (C) where the wrong-source points are.](pristine_scan_bug_figure.png){width=100%}

# What we saw in the data

Each of the four files should hold one source. Instead they hold:

| File | Should contain | Actually contains |
|---|---|---|
| excitation_1 | S1 | S1 |
| excitation_2 | S2 | S1 + S2, and at some points S1 only |
| excitation_3 | S3 | S1 + S3, and at some points S1 + S2 |
| excitation_4 | S4 | S1 + S4, and at some points S1 + S3 |

- **S1 everywhere:** source S1 is heard in all four files, at every point.
- **Previous source at some points:** in files 2–4, 5–8 % of the points (up to 12 % if noisy points are counted) hold the shot of the previous file instead of the new source. They are all in the upper half of the grid, where about 15 % of points are affected.

We found the second problem by comparing each file with the previous one at the same point. At a wrong point the two traces are identical apart from noise; at a correct point they differ by 30–70 times the noise.

# How the setup works

Two programs run side by side, each on its own clock. They never talk to each other directly; the only link is the trigger signal.

- **Excitation script** (Python, `PlaySamplesArnaud.py`): loads the 40 kHz Ricker wavelet on one generator channel, switches the trigger on for a fixed time, switches it off, waits, then moves to the next channel.
- **LDV macro** (Polytec, `Auto_Arnaud.wwb`): starts a scan, waits until the scan is finished, waits a fixed time, then starts the next scan and saves it as the next `excitation_i` file.

The LDV only records a point when a trigger arrives. It does not know which source is firing: the file name only reflects how many scans the macro has done. So the data are right only if each LDV scan runs entirely while the matching source is on.

# Cause 1: the two timers drifted apart

The script kept each source on for 7743 s, but the LDV cycle was only about 7230 s. So each LDV scan started about 500 s earlier, relative to the source switch, than the one before.

The settings involved:

| Where | Line | Setting | Value |
|----------|---|---------------|----------|
| `PlaySamplesArnaud.py` | 37 | `scan_time` | 7443 s (estimate) |
| `PlaySamplesArnaud.py` | 38–39 | `safety_margin`; `excitation_time = scan_time + safety_margin` | 300 s, giving 7743 s per source |
| `PlaySamplesArnaud.py` | 40 | `delay_before_next_trigger` | 30 s (trigger off between sources) |
| `Auto_Arnaud.wwb` | 39–41 | wait until the scan is stopped | real scan about 6630 s |
| `Auto_Arnaud.wwb` | 45 | `Wait 600` before the next scan | 600 s |

The real scan took about 6630 s (1650 points × 4.0 s), not 7443 s. The files were saved at 18:52, 20:53, 22:53 and 00:54, i.e. one LDV cycle every 7230 s or so (6630 s scan + 600 s wait).

What happened, counting from the start:

| Time from start | Excitation script | LDV | Result |
|---|---|---|---|
| 0 to 6630 s | S1 on | scan 1 | correct |
| 6630 to 7230 s | S1 on | idle (`Wait 600`) | nothing recorded |
| 7230 to 7743 s | **S1 still on** | scan 2 has started | about 510 s recorded with S1 instead of S2 |
| 7743 to 7775 s | trigger off, switching to S2 | waits for trigger | nothing recorded |
| 7775 s to end of scan 2 | S2 on | scan 2 continues | correct |
| start of scan 3 | **S2 still on** for about 1030 s | scan 3 | wrong source |
| start of scan 4 | **S3 still on** for about 1540 s | scan 4 | wrong source |

The error grows by about 510 s at every changeover, which matches the data: file 1 is clean, and the band of wrong points extends further down the grid in files 3 and 4 than in file 2.

One detail is not fully explained: inside the affected band only part of the points (10–45 % per row) are wrong, not all of them. The switching may also be unreliable right around the changeover. The fix below covers both.

# Cause 2: S1 is never switched off

Before loading the next source, the script sets the other channels to zero (`PlaySamplesArnaud.py`, lines 49–51):

```python
for ch in range(1, N_channels):
    signalGenerator.setChannelData(ch, Zeros)
```

`range(1, 4)` gives channels 1, 2 and 3 only. Channel 0, which drives S1, is never cleared, so once S1 has its wavelet it keeps firing for the rest of the run. The correct line is `range(0, N_channels)`.

We kept this line unchanged for the weekend defect scan on purpose, so that every defect file is S1 + Sk exactly like the pristine file and the two can be subtracted directly. It should be fixed for future scans.

# Fix used for the weekend defect scan

The idea is to make the LDV restart while the trigger is off. Because the LDV only records when a trigger arrives, it then waits and starts exactly when the next source switches on, so the two programs realign at every changeover and errors cannot add up.

Settings for the 3 mm grid (6418 points × 4.0–4.1 s, i.e. 25 700–26 300 s per scan):

| Where | Setting | Old | New |
|-----------|--------------|----|----|
| `PlaySamplesArnaud.py`, line 37 | `scan_time` | 7443 s | 25 700 s |
| `PlaySamplesArnaud.py`, line 38 | `safety_margin` (trigger stays on after the expected end of the scan) | 300 s | 7200 s |
| `PlaySamplesArnaud.py`, line 40 | `delay_before_next_trigger` | 30 s (trigger off between sources) | 7200 s |
| `Auto_Arnaud.wwb`, line 45 | `Wait` before the next scan | 600 s | 10 800 s |

The LDV wait is set to margin + half the gap (7200 + 3600 s), so the LDV restarts in the middle of the 2 h trigger-off gap. Two conditions keep the data clean:

1. The scan must finish before the trigger switches off: the real scan can be up to 7200 s longer than 25 700 s.
2. The LDV must restart inside the gap: the real scan can be up to ±3600 s off from 25 700 s.

So each scan can be off by ±1 h (±14 %) without any wrong-source points; the measured 4.0–4.1 s per point gives −30 to +610 s. Start order: LDV macro first, then the Python script. Total duration about 42.5 h (launched Sat 14 Nov 09:48, LDV done around Mon 16 Nov 02:30).

For future scans, the cleaner fix is to let the LDV drive the source switch (its trigger or "scan finished" output), so the two programs cannot get out of step at all.

# What it means for the pristine data

The pristine scan is usable and does not need to be repeated.

- **S1 in every file:** not a problem as long as the defect scan has the same S1 + Sk content, which it does. Single-source fields can also be recovered by subtracting file 1: the leftover after removing S1 is 0.4–0.5 %, close to the noise level of 0.2–0.3 %.
- **Wrong-source points:** they are easy to detect (identical to the previous file at the same point) and are masked before comparing with the defect scan. This removes 5–8 % of the points in files 2–4, all in the upper half of the grid. Every point has a correct S1 reference, and the lower half is complete for all sources.
- **Signal quality:** otherwise good: median SNR 110–130, the shot timing is identical at all points, and the plate response is repeatable to better than 1 %.
