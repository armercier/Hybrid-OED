# PMMA plate, 3-D LDV: acquisition and processing log

Running record of every acquisition and processing step on the PMMA lab specimen (November 2026).
New steps are appended at the end, in order, with the script, its settings, its outputs and the key numbers.
Paths are relative to the repository root. Python: `/home/merciera/miniconda3/envs/hybridOED/bin/python`.

## 0. Specimen and setup

| Item | Value |
|---|---|
| Plate | PMMA, 900 × 835 × 10 mm, on foam pads |
| Model material | vp 2730 m/s, vs 1340 m/s, rho 1180 kg/m³ (calibrated velocity scale 0.995) |
| Transducers | 4 PZT blocks, 10 × 10 mm, 7 mm high (≈ 5.3 g), bonded on the back face |
| Drive | Ricker wavelet, 40 kHz, centred 0.5 ms after the trigger, amplitude × 1.5 (`PlaySamplesArnaud.py`) |
| LDV | Polytec 3-D PSV, vib_x/y/z exported to .mat, fs 3.125 MHz, 5000 samples (1.6 ms), 20 averages |
| Shot timing | wave leaves the source ≈ 508 µs after the trigger; `ref1` in the files is only the trigger gate |
| Main mode | A0: λ ≈ 25 mm, phase velocity ≈ 1000 m/s, group velocity ≈ 1336 m/s at 40 kHz; S0/SH0 not above noise |

Coordinates: scanner frame (x, y) in mm. Layout drawings are given from the centre O of the pristine scan grid,
O = (912.3, 511.9) mm: `reports/2026-11_specimen/make_drawing_pmma_centred.py` (plan) and
`make_drawing_pmma_final.py` → `specimen_pmma_final_drawing.{pdf,png}` (as built).

## 1. Acquisitions

| Date | Data (original, never modified) | Grid | Notes |
|---|---|---|---|
| 13 Nov | `ldv_data_comprehensive/Soruce_1_40kHz_1-5V_20avg_320nt_3125kHz.mat` | 6 mm, 50 × 33 | coarse pristine scan, two sources fired at once by mistake |
| 13–14 Nov | `ldv_data_comprehensive/pristine_mat/excitation_{1..4}_pristine.mat` | 6 mm, 50 × 33 = 1650, x 765.3–1059.3, y 415.9–607.9 | pristine, one file per source; see the bugs below |
| 14 Nov | — | — | defects drilled from the back face (`ldv_data_comprehensive/defects.txt`) |
| 14 Nov | `ldv_data_comprehensive/scatter_test_mat/excitation_{1..4}.mat` | 4 corner points | quick check after drilling |
| 14–16 Nov | `ldv_data_comprehensive/scatt_mat/excitation_{1..4}_scatt.mat` | 3 mm, 100 × 66 minus 182 edge points = 6418, origin (760.3, 413.1) | defect scan |

### Pristine scan: two acquisition bugs

Full explanation for colleagues: `reports/2026-11_specimen/pristine_scan_bug_report.{md,pdf}`, with the figure
`pristine_scan_bug_figure.{png,pdf}` (`make_bug_figure.py`: timeline of both programs, the zeroing bug, map of the wrong-source points).

1. **S1 fires in every file.** The zeroing loop in `PlaySamplesArnaud.py` uses `range(1, N_channels)` and never clears channel 0.
   Files: ex1 = S1, ex2 = S1 + S2, ex3 = S1 + S3, ex4 = S1 + S4. Kept unchanged for the defect scan so both scans have the same content.
2. **Wrong source at some points.** The excitation script (7743 s per source) and the LDV macro (≈ 6630 s scan + `Wait 600`) ran on
   independent clocks; each scan started ≈ 510 s earlier relative to the source switch. Some points of files 2–4, all in the upper half
   of the grid, hold the previous file's shot. Detection: file k identical to file k−1 within noise at the same point.

### Defect scan: timing used (no wrong-source points)

`scan_time = 25700`, `safety_margin = 7200`, `delay_before_next_trigger = 7200` (Python); `Wait 10800` (PSV macro, line 45).
The LDV restarts in the middle of the 2 h trigger-off gap and waits for the trigger, so the two programs realign at every switch.
For future scans: `src/ldv_pmma/scan_schedule.py` computes these settings from the scan time and its uncertainty;
one-page explanation `reports/2026-11_specimen/ldv_scan_schedule_sheet.pdf` (`make_schedule_sheet.py`).
Self-contained excitation script with the timing built in: `src/ldv_pmma/excitation_scan.py` (replaces `PlaySamplesArnaud.py`;
zeroes all channels by default, `--keep-s1-on` reproduces the Nov 2026 S1 + Sk files; writes a CSV log of every switch).
Files saved Sat 17:34, Sun 04:07, Sun 14:55, Mon 02:03 (≈ 3.84 s per point).

### As-built defects

Positions from O (pristine grid centre), the drilling plan; depths and diameters from `defects.txt`.

| Defect | (u, v) mm | As built |
|---|---|---|
| D1 | (−75, +47) | blind hole 10 mm, 6 mm deep, from the back |
| D2 | (+70, +52) | blind hole 10 mm, 6 mm deep, from the back |
| D3 | (0, −13) | through-hole 10 mm |
| D4 | (−85, −53) | blind hole 8 mm, 6 mm deep, from the back |
| M1 | (+80, −48) | steel cube 10 mm, glued on the back |

## 2. Data quality assessments

| Scan | Script | Outputs | Key results |
|---|---|---|---|
| Coarse | `src/ldv_pmma/assess_coarse_scan.py`, `locate_sources.py`, `fit_mindlin_coarse.py` | `ldv_pmma_results/{quality,snapshots,fk,sources,mindlin_fit}*` | A0 dominant; SNR_z ≈ 95; Mindlin misfit 0.077, s 0.99, Q 40 |
| Pristine | scratch analyses (logic now in `prep_dataset.py`) | `ldv_pmma_results/pristine/onset_check.png` | SNR 110–134; ≈ 5–8 % wrong-source points in files 2–4 (upper half); S1 separation by subtraction leaves 0.4–0.5 % |
| Defect | `src/ldv_pmma/assess_scatter.py` | `ldv_pmma_results/scatter/{quality,summary}*` | correct source at 99.96–100 % of points; noise 11 µm/s (file 1: 18.5 µm/s, SNR 61); ≈ 10 % dropouts |

### Plate movement between the scans

The direct defect − pristine difference was 60–100 % of the signal everywhere: the plate moved (most likely when it was taken off for drilling).

- `src/ldv_pmma/scatter_maps.py`: wavefield registration on the direct waves (520–820 µs) of all four files.
  Convention: pristine point p ↦ defect-scan location R(θ)(p − O) + O + T. Result (`ldv_pmma_results/scatter/registration.json`):
  θ = −0.09° (another start gave −0.25°: rotation poorly constrained), **T = (−3.6, −2.7) mm**, per-file delays 0.3–0.6 µs.
- `src/ldv_pmma/locate_sources_scans.py`: sources located in both scans from the arrival times of each source alone (file k − file 1);
  shift between scans (−2.8, −3.5) mm, consistent.
- After registration: files 1–2 direct-wave difference ≈ 15 % far from the defects; D3 (and its shadow), D2, M1 visible.
  Files 3–4 spoiled in the upper half by the pristine wrong-source points. Late part (> 650 µs) changed.
- Hand-measured source positions were 10–245 mm off the data and are not used. The file 2 source is at x ≈ 1128 mm (right),
  the file 4 source at x ≈ 689 mm (left).
- Animations: `src/ldv_pmma/animate_scans.py` → `ldv_pmma_results/scatter/anim_scatter_ex{k}.gif`;
  `scatter_maps.py` → `anim_diff_ex{k}.gif` (pristine | registered defect scan | difference).

## 3. Workflow steps

### Step 1 — working copy and inversion dataset (16 Nov)

- Raw files copied to `ldv_work/{pristine_mat,scatt_mat}` (md5-verified, `ldv_work/md5_copy.txt`, set read-only), plus `ldv_work/defects.txt`.
  **All processing reads the copy; the originals in `ldv_data_comprehensive/` are never touched.**
- `src/ldv_pmma/prep_dataset.py` → `ldv_work/prepared/{pristine,scatter}.npz`, `prepared_info.json`:
  vz band-passed 15–90 kHz (4th-order Butterworth, zero phase), resampled to 625 kHz, window 400–1500 µs;
  noise RMS (0–480 µs); dropouts = noise > 3 × median; wrong-source points (pristine) = |file k − file k−1|² < 3 × noise².

| Usable points | ex1 | ex2 | ex3 | ex4 |
|---|---|---|---|---|
| Pristine (of 1650) | 1492 | 1413 | 1360 | 1385 |
| Defect (of 6418) | 5727 | 5711 | 5781 | 5772 |

In-plane components are not included yet.

### Step 2 — geometry (16 Nov)

`src/ldv_pmma/geometry.py` → `ldv_work/prepared/geometry.json`: plate, PZT blocks, registration, sources (arrival-time fits) and defects
(as built, planned positions) in both scan frames. Pristine sources mapped with the registration land within 2–6 mm of the defect-scan fits.

Ruler check (16 Nov, block centre to D3 centre, mm) against the arrival-time fit in the defect scan with a common shot delay
(508 µs) and group velocity (1335 m/s):

| Source | Ruler (from D3) | Data (from D3) | Difference |
|---|---|---|---|
| S1 | (0, −160) | (−2.2, −159.3) | 2 mm |
| S2 | (+223, +24) | (+219.5, +22.1) | 4 mm |
| S3 | (0, +180) | (+0.9, +179.2) | 1 mm |
| S4 | (−216, 0) (first reading −205, re-measured) | (−219.3, +2.6) | 4 mm |

All four agree within 1–4 mm: confirms the source positions, the D3 position and the registration between scans.

### Step 3 — Mindlin calibration on the pristine scan (16 Nov)

`src/ldv_pmma/calibrate_pristine.py --scan pristine` → `ldv_pmma_results/calib/calib_pristine.{json,log}`, `_snapshots.png`, `_traces.png`, `_filters.npz`.

- Model: homogeneous Mindlin plate, 1 mm grid (626 × 518), sponge 60 cells (plate edges not modelled), kappa matched to A0 at 40 kHz,
  mass-proportional damping (Q at 40 kHz), vertical point force at each block centre, PZT blocks as added mass (5.3 g) on their footprint,
  all four blocks present in every simulation.
- Data: window 450–950 µs; masks from step 1; files 3–4 restricted to the lower half (not all wrong-source points are detectable).
- Source signatures: one complex filter per frequency per transducer, estimated jointly from the four files (file 1 = f1 G1, file k = f1 G1 + fk Gk).
- Fitted by coordinate search: velocity scale s, Q, four source positions.

| Result | Value |
|---|---|
| Velocity scale s | 0.995 |
| Q (40 kHz) | 34.5 |
| Sources, pristine frame (mm) | S1 (910.4, 338.7), S2 (1130.8, 522.4), S3 (913.4, 680.4), S4 (685.7, 502.9) |
| Misfit 450–950 µs | **0.041** (ex1 0.042, ex2 0.039, ex3 0.040, ex4 0.042); noise floor ≈ 0.004 |
| Misfit per window | 450–650 µs 0.028, 650–800 µs 0.027, 800–950 µs ≈ 1.0 (edge echoes, not modelled) |
| Per-point correlation | median 0.98 in every file |

PZT block models compared at the calibrated parameters (misfit): bilayer stiff layer 0.044, mass only 5.3 g 0.041, mass 2.6 g 0.040,
mass 10.6 g 0.055, no blocks 0.040. The blocks barely affect the scanned field; mass only is kept as the physical choice.

Validation on the defect scan (pristine model, no defects, 450–800 µs, sources mapped with the registration):
misfit 0.10–0.13 (vs 0.041 on the pristine); the residual concentrates at D1, D2, D3 and M1 in all files, D4 weak
(`ldv_pmma_results/calib/defect_scan_vs_pristine_model.png`). Conclusion: invert the defect scan directly, no pristine subtraction.

### Step 4 — full-waveform inversion of the defect scan (16 Nov, running)

`src/ldv_pmma/invert_defect_scan.py --iters 150 --tag v1` → `ldv_pmma_results/inversion/v1/{run.log, result.npz, settings.json, maps.png, history.png}`.

- Data: defect scan, all usable points (6343 receivers), window 450–800 µs with a 30 µs cosine taper at the end.
  The edge echoes arrive after ≈ 800 µs (step 3 misfit per window), so this window removes them; the model has no edges (sponge).
- Model: calibrated pristine model (s, Q, sources mapped to the defect-scan frame, PZT block masses) plus two unknown maps on a 3 mm grid
  over the scan area (100 × 66), bilinear onto the 1 mm FD grid, pristine outside:
  thickness h = h_min + (h_max − h_min) · sigmoid(m0 + b) with **h_min = 1 mm** (conservative: hole depths, through or not, are unknown),
  h_max = 12 mm, m0 = 0 → 10 mm; mass scale = exp(m1) on ρh (the cube; thinning already removes mass through h).
- Source signatures re-estimated jointly at every iteration (variable projection, no gradient through them).
- Loss: mean over files of the normalised residual energy + TV (1e-4) on (m0, m1); Adam, learning rate 0.03, 150 iterations; ≈ 19 s per iteration.
- Starting misfit per file: 0.126, 0.103, 0.100, 0.119.
- Results (run v1, 150 iterations, ≈ 60 min; iterations slowed to 22–28 s while another user's jobs shared the GPU):
  - Misfit 0.112 → 0.061 (per file 0.070, 0.063, 0.055, 0.056); most of the drop in the first 20 iterations, then a plateau.
    The pristine calibration reached 0.041, so part of the defect signal is still unexplained.
  - **All four holes are localised, D4 included, but in the wrong parameter:** the mass map rises at D1 (×1.58), D2 (×1.63),
    D3 (×2.83) and D4 (×1.56), while the thickness map barely moves (≥ 9.86 mm at every hole; true 4 mm, D3 0 mm).
    A local mass increase and a local thinning both slow the A0 wave, so the data cannot easily tell them apart,
    and with Adam a step of the mass parameter changes the model ≈ 6× more than a step of the thickness parameter
    (exp(m1) vs h_min + (h_max − h_min)·sigmoid), so the mass map took the defects.
  - **The cube M1 is not recovered** (mass 1.04, true 7.65), although its residual was among the strongest before the inversion.
  - Background: thickness 9.86–9.97 mm, mass 0.97–1.07 (5–95 %). Artefact: the minimum thickness (2.8 mm) is at the top-right corner of the model grid.
  - Outputs: `ldv_pmma_results/inversion/v1/{maps.png, history.png, settings.json, result.npz, run.log}`.
  - Clearer figure (CPU, `src/ldv_pmma/plot_inversion.py --tag v1` → `maps_clear.png`): as-built outlines, colour ranges set to
    the inverted values, 50 × 50 mm zooms. Each hole is a **ring** of higher mass centred on its as-built outline (positions match
    to within a few mm, confirming the registration); the thickness map only shows a smooth 9.85–10.0 mm trend (thinner on the left),
    no defects; around M1 the mass is *lower* (≈ 0.9), the wrong sign for a glued cube.

### Step 4b — truth test (16 Nov)

`src/ldv_pmma/truth_test.py` → `ldv_pmma_results/inversion/truth_test.json`; forward runs only, same data, window, sources and
signature estimation as the inversion. "As built" = the defects as actually drilled (`defects.txt`: sizes and depths) at their
planned positions: blind holes as a sharp step to 4 mm, D3 (through) as 1 mm (h_min), M1 as added mass (7.85 g on 10 × 10 mm).

| Model | Misfit |
|---|---|
| v1 inversion result | 0.061 |
| as built, D3 at 2 mm | 0.106 |
| holes only | 0.112 |
| pristine (no defects) | 0.112 |
| as built | 0.124 |
| cube only | 0.124 |
| as built, D3 at 0.5 mm | 0.154 |
| as built, all defects shifted 3 mm (4 directions) | 0.133–0.142 |

Per-defect sweep (one defect at a time, gain vs pristine 0.1123): thinning helps every hole but only a little
(D1 −0.005 at 4 mm, D2 −0.007 at 2 mm, D3 −0.006 at 4 mm, D4 −0.001 at 5 mm); a uniform mass increase in a hole does not help;
the cube as added mass makes the fit worse at every mass up to ×7.65 (+0.012), slightly better as a thicker/stiffer patch (12 mm, −0.001)
or a mass decrease (−0.001).
Conclusions: (1) the positions are right (any 3 mm shift is worse); (2) a through-hole modelled as a 1 mm or 0.5 mm plate scatters
far too strongly; (3) the defects alone explain only a small part of the misfit: most of the drop 0.112 → 0.061 in v1 came from the
smooth background changes (thickness trend, mass patterns), i.e. from model errors between the pristine calibration and the defect
scan (source positions mapped by the registration with 2–6 mm errors, small velocity/temperature changes), not from the defects.

### Steps A/B — ORACLE experiment (16 Nov, running)

**Not a blind method: uses ground truth.** Step A (`src/ldv_pmma/calibrate_defect.py`) recalibrates s, Q and the source positions
on the defect scan with all points within 25 mm of the *known* defects removed, and puts the cube in the model as a known rigid
body (October 2026 model: 6 mm footprint, 7.85 g, full rotary inertia). Step B (`src/ldv_pmma/invert_thickness.py --tag v2`) is a
thickness-only inversion (h = h_min + (H − h_min)·exp(m), mass follows ρh) on that background.
Purpose: upper bound — can the Mindlin model image the holes if the background is right? Step A start: misfit 0.076 on the
background points (sources from the defect-scan arrival times, rigid cube), vs 0.112 for v1's start on all points.
Step A result (`ldv_pmma_results/calib/calib_defect.{json,log}`, `calib_defect_residual.png`, 92 evaluations):
s 0.990, Q 30; misfit 0.064 on the background points (pristine calibration: 0.041), 0.088 on all points
(per file 0.101, 0.099, 0.087, 0.076); sources relative to D3 (mm): S1 (−3.0, −159.3), S2 (+218.8, +22.9), S3 (+1.6, +179.2),
S4 (−218.6, +2.6), within 1–4 mm of the ruler. The residual still peaks at D1, D3 and around M1 (largest, below-right of the
cube, even with the rigid-cube model), D2 and D4 weaker.
Step B result (`ldv_pmma_results/inversion/v2/{run.log, result.npz, settings.json, maps_clear.png}`, 150 iterations,
≈ 21 s each): misfit 0.092 → 0.053 (per file 0.061, 0.056, 0.049, 0.044; v1 ended at 0.061).

| Defect | As built | Inverted thickness within outline + 3 mm |
|---|---|---|
| D1 | 4 mm left (10 mm Ø) | min **2.9 mm** |
| D2 | 4 mm left (10 mm Ø) | min **3.75 mm** |
| D4 | 4 mm left (8 mm Ø) | min **3.7 mm** |
| D3 | through-hole (10 mm Ø) | **13.8–29 mm**: a stiff plug with a thinner ring (≈ 7 mm) around it |
| M1 | rigid cube in the model | 3.3–55 mm: artefacts (leftover not explained by the rigid-cube model) |
| Background (> 25 mm from defects) | 10 mm | 9.85–10.43 mm (1–99 %) |

The three blind holes are recovered as thinning at the right place and close to the true depth. The through-hole is
detected and located but with the wrong sign, for two reasons: (1) the plate model has no void / free-edge condition — a very
thin region is a slow, ringing soft patch (truth test: 0.5–1 mm makes the fit worse), so the strongest reflector it can make
is a stiff plug; (2) the 22 LDV points within 8 mm of the D3 centre are not flagged as dropouts but have ≈ 2× the amplitude of
their surroundings (400–580 vs 240–370 µm/s): no plate surface there, the beam measures inside the hole, and no plate model fits
these traces. Proposed for the blind pipeline: flag such "no surface" points from the data (anomalous amplitude / signal level),
report an anomaly map |h − 10 mm| for detection, and present through-holes as a limitation of the plate model.

Decision (16 Nov, for the full paper): every step used to produce results must be blind to the defects. Legitimate: pristine
(baseline) calibration, registration from the direct waves, source positions from the arrival times, robust self-calibration or a
joint inversion of background + defects; the as-built defects only for evaluation. A/B is reported only as an oracle bound.

### Blind pipeline (16 Nov, running)

Decisions: background from both scans without defect knowledge — model choices and starting values (s 0.995, Q 34.5) from the
pristine calibration; source positions from the defect-scan arrival times; final s, Q and sources from a robust self-calibration
on the defect scan. The cube is an unknown target (not in the model).

1. "No surface" points (`src/ldv_pmma/calibrate_blind.py` → `ldv_work/prepared/scatter_nosurface.npz`): vz RMS (500–800 µs) more
   than 1.5 × the median of the neighbours within 12 mm (median over the four files). Fixed round threshold (amplitude ratio
   distribution: median 1.0, MAD 0.028, 99th percentile 1.25; the points in the through-hole are ≈ 2×). Flags 17 of 6418 points
   (for reading only: 11 at D3, 2 at D1, 2 at D2, 1 near D4, 1 elsewhere). Excluded from calibration and inversion.
2. Robust self-calibration (same script → `ldv_pmma_results/calib/calib_blind.{json,log}`, `calib_blind_weights.png`):
   IRLS per-point weight w = min(1, 2·median(e)/e), e = residual energy of the point, updated at each of 3 sweeps of the
   coordinate search (s, Q, four source positions). Start misfit (weighted) 0.082; 165 points with weight < 0.5 at the start.
   Also writes `background_pristine_mapped.json` (pristine s, Q, sources mapped with the registration).
3. Blind thickness-only inversion `invert_thickness.py --tag v3 --background calib_blind.json --no-cube --nosurface`.
4. Comparison: pristine-only background `--tag v4 --background background_pristine_mapped.json --no-cube --nosurface`.

Blind calibration result: s 0.990, Q 30 — the same as the oracle step A; sources from D3 (mm) S1 (−3.0, −159.3),
S2 (+218.0, +22.9), S3 (+1.6, +179.2), S4 (−218.6, +2.6), within 0.8 mm of the oracle and 1–5 mm of the ruler; misfit 0.084 on
all points (oracle step A: 0.088), 0.070 weighted; 221–251 points down-weighted (w < 0.5). The low weights sit, from the data
alone, at D3, around M1 (largest zone, the cube is not in the model), at D1 and D2, and on a short row of points at v ≈ +58 mm,
u ≈ +100…+145 mm (a scan-line artefact, also visible in the earlier residual maps); none at D4.
Inversion v3 (blind; 150 Adam iterations, about 20–29 s each, slower while the GPU was shared;
`ldv_pmma_results/inversion/v3/{result.npz,settings.json,run.log,maps_clear.png}`): misfit 0.088 → 0.047, against
0.092 → 0.053 for the oracle v2. Minimum thickness within 8 mm of each defect, v3 (v2):

| Defect | As built | v3 min h | v2 min h |
|---|---|---|---|
| D1 | 4 mm left | 2.97 | 2.89 |
| D2 | 4 mm left | 3.68 | 3.75 |
| D3 | through | 2.49 (max 13.3, stiff ring above/below) | 13.8 (stiff plug only) |
| D4 | 4 mm left | 3.76 | 3.72 |
| M1 | cube, not in model | thick ring 6.5–59 mm (unknown target found) | rigid cube modelled |

Background median 10.04 mm; outside 20 mm of the defects no cell is below 9.7 mm (v2: min 8.5 mm). A thick, smooth blob at
about (+65, −70) mm, 25 mm south-west of M1, is an artefact compensating the unmodelled cube. Excluding the blind
"no surface" points makes D3 appear as a thinning (2.5 mm, the h_min floor not reached) instead of a stiff plug.
Note: the first line of `run.log` prints the hard-coded label "background from step A, rigid cube"; for v3 the background was
`calib_blind.json` and there was no cube (see `settings.json`).
Inversion v4 (pristine-only background: s 0.995, Q 34.5, pristine sources mapped with the registration; otherwise as v3;
`ldv_pmma_results/inversion/v4/`): misfit 0.101 → 0.0485 (v3 0.088 → 0.0469); per file 0.051 / 0.052 / 0.044 / 0.048
(v3 0.051 / 0.053 / 0.045 / 0.039, i.e. S4 is clearly worse). Defects almost identical to v3 (min h D1 2.92, D2 3.65,
D3 2.53, D4 3.81 mm; same M1 ring and SW blob). The background is not: broad, smooth thin bands along the left (u < −130 mm)
and bottom edges of the scan area, 105 cells below 9 mm more than 20 mm from any defect (v3: none; min 8.1 mm); RMS
difference v4 − v3 0.29 mm, max 1.9 mm. These bands compensate the background error (mainly the S4 position/speed).
Conclusion: defects are robust to the background choice; the blind self-calibration removes the false background
anomalies. Decision for the reduced-scan study: use `calib_blind.json` for all layouts (same background for every layout,
so the comparison between layouts is fair); caveat: it was computed on the full defect scan.

## 4. Open issues and decisions

- Source positions: data-fitted ones used; confirmed by the ruler check against D3 (1–4 mm, step 2).
- S4 was ≈ 20 % weaker in the defect scan than in the pristine (bond possibly affected during drilling).
- Plate edges are not located in scanner coordinates; needed only if data after ≈ 800 µs are to be used.
- For future scans: fix `range(0, N_channels)` in `PlaySamplesArnaud.py`, or let the LDV drive the source switch.
- Next steps: 5) OED replay on the dense scan (point subsets vs uniform/random), 6) emulated sequential OED, 7) evaluation and figures.

## 5. Reduced-scan study (layouts: regular, random, adaptive OED)

Run table with status and schedule: `ldv_pmma_results/reduced_scan_runs.md` (kept up to date).
1. Subsets (`src/ldv_pmma/make_subsets.py` → `ldv_work/prepared/subsets/`): regular 6/9/12 mm sub-grids of the 3 mm scan
   (1626/716/413 points), random subsets with the same number of usable points (seeds 1, 2). "No surface" points
   recomputed on each subset from the subset alone (8 nearest neighbours, ratio 1.5): 1–4 per subset.
2. `invert_thickness.py` gained `--subset` (default behaviour unchanged). Settings as v3, 100 iterations (v3 had converged
   by 75–100: misfit 0.0473 at 75, 0.0468 at 100, 0.0469 at 150), background `calib_blind.json` for all layouts.
3. 16 Nov 14:40: launched r9, then x9a.
4. r9 done (16 Nov 15:13): min h D1 2.94, D2 3.71, D3 2.34, D4 4.59 mm; no false spots; RMS difference to v3 0.24 mm.
   x9a and x9b (random) run after it. Misfits are normalised on each subset's own points, so layouts are compared on the
   thickness map (values at the defects, false spots, difference to v3), not on misfit.
5. OED replay script `src/ldv_pmma/oed_replay.py` (counting rule (b)): budget N = regular layout of the chosen spacing,
   split into R = 5 interleaved groups ((i + 2j) mod 5); round k: group k is the starting position of the new points,
   design = Adam on their coordinates (5 steps, lr 3 mm) with the paper's illumination criterion (12 Rademacher probes
   fixed within a round, VJP per source, h = mean s², 3×3 smoothing, var(h/mean) − 0.05·log(mean)) over new + measured
   points on the current model; acquire = nearest scanned point not yet measured; 20 FWI updates on all measured points
   (model carried, Adam reset as in the paper; total 100 as the baselines); no-surface rule of make_subsets.py once ≥ 30
   points. `--no-design` measures the regular layout through the same loop (control).
   CPU test (`--test`: 4 mm grid, 2 rounds of 12 points, 2 probes): loop runs end to end (3 min, 2.9 GB RAM); design
   objective decreases (3.58 → 3.08, 2.90 → 2.41), median move 6 mm, snap ≤ 2 mm. Design gradient vs finite differences
   (1 mm step): −29.1 / −25.9, −18.7 / −18.4, +24.3 / +21.7 (agree in sign, within about 10 %; the bilinear
   receiver sampling makes the objective piecewise smooth in position). First test showed the objective rising when the
   probes were redrawn every step; fixed by keeping them fixed within a round (as the paper).
6. x9b done (16 Nov 16:18): min h D1 3.23, D2 3.77, D3 2.34, D4 3.49 mm; no false spots; RMS vs v3 0.28 mm.
   x9a stopped at iteration 87 (its shell was ended with the previous Claude session, about 15:45); the saved result is
   iteration 80 (min h D1 3.46, D2 3.69, D3 2.65, D4 3.96 mm; RMS vs v3 0.38 mm). To be rerun on 17 Nov for 100 iterations.
7. OED design memory (16 Nov): first GPU benchmark failed with out-of-memory (142 GB requested): differentiating the
   illumination w.r.t. the point positions through all 48 (source, probe) sensitivities in one graph keeps every term.
   Fix as in the Oct 2026 report (Sec. 2, `src/remat/remat_hypergrad.py`): each (source, probe) term is wrapped in
   `jax.checkpoint` and mapped one at a time, so the backward pass recomputes a term instead of storing it. CPU check:
   same value and gradient as the one-graph version (an explicit term-by-term sum matched it to 1e-6). Probe batching
   (as in October) is not possible here because it would also batch the source position. Finite-difference check on the
   4 mm test grid: agreement at one point, small-gradient points disagree (bilinear receiver sampling makes the objective
   kinked in position); to be redone with smaller steps or on the 1 mm grid.
   Second GPU benchmark started 16:27 and was stopped at 16:29 (GPU handed back at 16:30): no out-of-memory in the first
   2 min (2.4 GB in use while compiling), no timing yet. Expected cost of one design step: 48 terms × (forward + adjoint
   + their recomputation and second derivative), about 10–15 min at full size; 5 steps × 5 rounds would be 4–6 h, so
   fewer design steps and/or probes per round are needed (decide after the benchmark).
8. 17 Nov: budgets extended to 18 mm (r18: 179 points, 176 usable, 2.8 %) and 24 mm (r24: 111 / 110, 1.7 %) with random
   counterparts (same seeds; existing subsets unchanged, md5 checked). Reason: at 9 mm (11 %) all layouts already match
   the full scan, so the OED is to be run where the baselines fail. New queue runner `src/ldv_pmma/run_queue.sh`
   (detached with setsid/nohup so a session end cannot kill it, as happened to x9a; skips finished jobs).
   08:09 started: bench_o9 (design timing/memory), r12, r18, r24.
9. 17 Nov queue finished 10:10. bench_o9: design value+grad 618 s (12 probes, 4 sources, full grid), FWI update
   19.8 s, peak 3.09 GiB. New script `src/ldv_pmma/score_inversions.py` (post-hoc scores, `scores.json`).
   r12 (406 pts): D1 2.98, D2 3.64, D3 2.27, D4 4.11 mm, no false cells, RMS vs v3 0.25 mm.
   r18 (176 pts): D3 recovered as a stiff plug (13.2 mm) in a thin ring, D4 weak (6.76), 11 false cells, RMS 0.98 mm.
   r24 (110 pts): D1 3.32, D2 3.70, D3 2.39, D4 5.09 mm, no false cells, RMS 0.37 mm.
   Not monotonic in the budget: grid alignment decides at 18-24 mm.
10. 17 Nov 10:45: subsets 36 mm (r36 50 points, 48 usable; x36a/b 48) and 48 mm (r48 32 points, 31 usable; x48a/b 31)
   added to make_subsets.py; existing subset files md5-identical after regeneration. Queue: r36, r48, x24a, x24b, x18a, x18b.
11. 17 Nov 10:50-11:10: faster design gradient in `oed_replay.py` (`--design-impl fast`): one forward run per source,
   the 12 probe VJPs batched (vmap) on it, one jax.checkpoint per source; `--src-batch` also vmaps the sources;
   `--design-ckpt` sets the solver checkpoint interval. CPU test (4 mm grid, 2 probes): value identical, gradient
   within 1.1e-6 (loop) / 1.7e-6 (src-batch) of the remat reference. GPU (RTX 2080 Ti, 11 GB), 12 probes, full grid,
   shared with the r36 inversion, memory capped at 50 %: fast + sources looped 395 s per design value+grad, peak
   2.96 GiB (reference remat 618 s unshared; r36 iterations slowed 20 -> 45 s meanwhile, so roughly 200 s unshared,
   to be confirmed on a free GPU). Sources batched: needs 11.6 GiB (ckpt 100) / 25.4 GiB (ckpt 250), does not fit
   the card. Log `ldv_pmma_results/bench_fast.log`.
12. 17 Nov 11:20: `--src-chunk K` in `oed_replay.py` (fast design: K sources vmapped together, chunks looped).
   CPU test: K = 2 value identical, gradient within 2.0e-6 of the remat reference (K = 1: 1.1e-6). Not yet run on GPU;
   expected ~6 GiB for K = 2 (half of the 11.6 GiB measured for K = 4), so only with the GPU free.
13. r36 (48 pts, 0.8 % of the scan) done 11:25: misfit 0.080 -> 0.030; D1 4.09, D2 3.56, D3 2.49, D4 4.97 mm (all holes
   found), but 88 false cells below 9 mm (min 7.5 mm) spread over the lower-right quarter around the unmodelled cube;
   RMS vs v3 1.31 mm. The failure mode at low budget is false structure, not missed holes.
14. r48 (31 pts, 0.5 %) done 11:58: misfit 0.073 -> 0.034; only D3 found (2.78 mm); D1 8.98 and D2 8.96 mm (faint broad
   blobs), D4 9.60 mm (missed); 51 false cells, thin artefacts south-west of M1 down to 5.3 mm; RMS vs v3 1.19 mm.
   Breaking point of the regular layout between 48 points (holes found, false structure) and 31 points (holes missed).
15. 17 Nov 12:30: queue changed: x24b, x18a, x18b dropped; after x24a the queue runs x48a, x48b, x36a, x36b
   (randoms at the breaking budgets). The OED goes to 31 points (o48) first.
16. x24a (random, 110 pts) done 12:31: D4 missed (9.57 mm, no point within 15.9 mm), D1 2.89, D2 3.48, D3 2.37 mm;
   61 false cells around M1 (min 4.4 mm); RMS vs v3 1.15 mm. Regular r24 at the same budget found all four holes.
17. x48a (random, 31 pts) done 13:04: all four holes found (D1 4.99, D2 3.63, D3 2.54, D4 4.80 mm) although the nearest
   points are 13 / 8 / 35 / 18 mm away; 22 false cells (min 5.9 mm, south-west of M1); RMS vs v3 1.46 mm. The regular
   r48 with the same count missed D1, D2, D4: at 31 points, irregular sampling does better than the regular grid
   (probably aliasing of the 48 mm grid), not proximity to the defects.
18. x48b (random, 31 pts) done 13:37: D1 2.81, D2 3.78, D3 2.59 mm found, D4 missed (9.55 mm); 468 false cells, a thin band
   along the bottom edge of the scan (min 4.4 mm); RMS vs v3 1.33 mm. Not a simple coverage gap: x48b has as many points
   below v = -60 mm (4) as r48, and x48a has 6. At 31 points: r48 finds 1 hole, x48a 4, x48b 3 (with large false
   structure): high spread between layouts with the same count, which is the regime for the OED comparison.
19. x36a (random, 48 pts, 47 used) done 14:09: D1 4.20, D2 3.64, D3 2.81 mm; D4 weak and shifted (8.12 mm within 8 mm,
   7.2 mm patch south-east of it); 97 false cells (M1 area, bottom edge); RMS vs v3 1.19 mm.
20. 17 Nov 14:30: `--move-all` in `oed_replay.py`: paper protocol (run_oed_multiseed_full_autodiff.py): all points start
   on the regular layout and all move every round (design Adam state carried across rounds), acquisition snaps each to
   the nearest scanned point (no duplicates within a round), each round's FWI uses only that round's points, model
   carried, FWI Adam reset. Logs distinct points over all rounds (lab cost). CPU test (4 mm grid, 24 points, 2 rounds)
   passed: 47 distinct points, 1 re-measured. Run o48m: 31 points from r48, 5 rounds x (1 design step, 8 mm; 20 FWI),
   12 probes, fast design with 2 sources at a time; starts after x36b.
21. x36b (random, 48 pts, 47 used) done 14:42: D1 3.17, D2 3.80, D3 2.26 mm, D4 missed (9.44 mm); no false cells (thin patch
   next to M1 only); RMS vs v3 0.66 mm. Queue finished; o48m started 14:42.
22. 17 Nov 14:52: `--fwi-first` in `oed_replay.py` (user): no design in round 0, so every design step uses a model that has
   seen data (round 0 = FWI on the starting regular layout; 5 FWI blocks, 4 moves). Differs from the paper's order
   (design first). CPU test passed. o48m (design first) stopped in round 0 and kept as `inversion/o48m_stopped`;
   o48f (move-all, FWI first, 31 points from r48, 12 probes, 8 mm step, 2 sources at a time) started 14:52.
   GPU memory with 2 sources at a time: 8.5 GB of 11 GB (seen during o48m).
23. 17 Nov 14:59: o48f with 2 sources at a time ran out of memory at the first design step (round 1): a single 5.83 GiB
   buffer on top of what the process held after the round-0 FWI (11 GB card). The 8.5 GB seen earlier was during
   compilation only. Automatic fallback: o48f restarted from round 0 with `--src-chunk 1` (about 3 GiB). Use
   `--src-chunk 1` on this card.
24. o48f round 1 (15:16): design step 181 s (1 source at a time, free GPU), every point moved exactly 11.3 mm
   = 8 mm in x and y. **To test later:** Adam's first step (and, with the state carried, later steps too) normalises
   each coordinate by its own gradient size, so every point moves ~lr per axis whatever its gradient strength: the
   design only picks directions (sign of the gradient), not how far. Same as the paper protocol (outer Adam lr 4, one
   step per round, state carried), so consistent with the synthetic study. Variants to test: plain gradient descent
   (step proportional to the gradient, needs a step scale), Adam with several steps per round, or a per-point
   step cap. Compare final maps and distinct-point counts against o48f.
25. o48f done 15:43 (44 min). Per round: design 166-181 s, FWI ~395 s. Moves (median) 11.3, 9.2, 7.3, 5.8 mm. 148 distinct
   points measured over 5 rounds (31 per round, 4 re-measured). Result: D2 3.48, D3 2.56 mm; D4 weak but in place
   (6.51 mm); D1 missed (8.61 mm); 64 false cells (thin streak at the right edge u ~ 140, v ~ 5 mm; M1 area); RMS vs v3
   0.82 mm. Nearest measured point to D1 per round: 22.5, 16.7, 17.7, 26.1, 26.8 mm (points never approached D1);
   D4: 16.1, 7.6, 10.2, 16.2, 19.5 mm. Compared with the start layout r48 (only D3), the moves add D2 and D4; compared
   with random x48a (all four, 31 distinct points) it is worse on holes, better on RMS. At equal distinct count
   (~110-150) r24 (110 regular) finds all four with no false cells: the paper protocol does not win on lab cost.
26. 17 Nov 16:00: new `src/ldv_pmma/plot_oed_trajectory.py` (trajectory.png: paths of the points over the final map,
   one panel per round). o48f: every point wanders locally (zig-zag, total excursion ~30-40 mm), direction often
   reverses between rounds (new probes each round -> noisy direction); no migration towards unexplored areas; round 1
   moves all diagonal (Adam sign step). Points starting near D1 moved away from it.
   Launched 15:55: o48b = rule (b) (31 distinct points, ~6 new per round, accumulated), `--fwi-first`, new
   `--design-opt gdnorm` (gradient descent, one scale per step: the point with the largest gradient moves
   `--design-move` 10 mm, the others in proportion), 3 design steps per round; then c48m = move-all, FWI first, no design
   (control for o48f). CPU test of gdnorm: moves median 2.8 / max 13.1 mm, objective decreased.
27. o48b done 17:04 (69 min): rule (b), 31 distinct points, gdnorm 3 steps x 10 mm per round. Result like its start r48:
   D3 2.77 mm only; D1 8.64 and D2 8.13 mm faint, D4 9.16 mm missed; 100 false cells; RMS vs v3 1.22 mm. Moves per
   round (median / max): 2.9/8.2, 9.4/12.5, 4.6/13.7, 6.4/10.3 mm: the points stay within ~10 mm of the regular
   start, several are pushed to the scan edges (u = +-150, v = +-95; probably the uniformity term lifting the weakly
   illuminated border). Nearest point to D1-D4: 16.5 / 21.9 / 24.9 / 24.8 mm. Conclusion: local gradient moves on the
   illumination criterion cannot bring points to where they matter; next (18 Nov): discrete candidate selection and an
   adaptive, model-targeted criterion. c48m started 17:04.
27. o48b done 17:04 (69 min; 3 gdnorm design steps ~530 s per round). Result no better than r48: D3 2.77 mm only;
   D1 8.64, D2 8.13 (faint), D4 9.16 (missed); 100 false cells; RMS vs v3 1.22 mm. Moves per round (median / max):
   2.9/8.2, 9.4/12.5, 4.6/13.7, 6.4/10.3 mm. Nearest points to D3 / D4 24.9 / 24.8 mm (r48: 13.6 / 16.1).
   Finding: the regular subsets use offset 0, so r48 puts 14 of its 32 points within 10 mm of the scan edge (x48a 4,
   x48b 2); o48b starts from r48 groups and keeps 13 of 31 at the edge. Part of "random beats regular" at 31 points
   may be this wasted edge coverage. To do: centred regular grids (offset s/2) as the fair regular baseline and start.
   Started c48m (move-all, FWI first, no design) 17:04.
28. c48m (control for o48f: move-all, FWI first, no design) done 17:38: identical to r48 within noise (D3 2.56 mm only,
   D1 8.73, D2 8.51, D4 9.47; 55 false cells; RMS vs v3 1.20 mm). Restarting the FWI Adam every 20 updates on the same
   points changes nothing; o48f's gains (D2, D4, RMS 0.82) come from the moves and the 148 distinct points measured.

### 18 Nov: decisive tests (scope limited because of the deadline)
29. Correction to item 27: counted on usable points, within 10 mm of the scan edge: r48 16 of 31, x48a 6, x48b 3, o48b 16
   (r36 14 of 48, x36a 8, x36b 7, r24 33 of 110). The corner-anchored r<s> grids waste points on the edge, so "random beats
   regular" at 31 points is not established. Two random seeds are also too few.
30. New code. `make_subsets_fair.py`: rc30, rc48 (cell-centred 6 x 5, 8 x 6), j30a/b (jittered, one random point per 6 x 5
   cell), x48c/d (random, seeds 3, 4); existing subsets unchanged. `oed_replay.py`: `--grad-check TAG` (full-size fast vs
   remat, finite differences on the model of run TAG); `--random-moves-from TAG` (o48f's move lengths, random directions);
   `--select coverage|targeted` with `--budget`: 6 start points (3 x 2 grid), then new points picked among ALL unmeasured
   scan points by greedy information gain (log-det, regularised) on linearised data changes for 12 smooth random model
   perturbations (one forward + tangent run per source and perturbation), relative to each candidate's trace level.
   targeted weights the perturbations by 0.1 + |current model| (normalised). Blind: uses only the current model.
   CPU tests (4 mm grid) passed; grad check there: fast = remat (6.6e-7), finite differences same sign, within 0-40 %.
31. Pass rule fixed BEFORE the runs. Budget 31 points. Baselines: rc30, j30a/b, x48a/b/c/d (not the corner-anchored r48).
   "Found" = min h < 7 mm within 8 mm of the hole. A selection run (s31t, s31c) passes if holes found >= the baseline
   median AND false cells (h < 9 mm, > 20 mm from every defect) <= the baseline median. One run per criterion, so a pass
   is only suggestive: confirm with a second seed before claiming it. Settings are not tuned on this plate afterwards.
   Queue `src/ldv_pmma/queue_18nov.sh`: gc_o48f, rc30, rc48, s31t, s31c, x48c, x48d, j30a, j30b, o48r.
32. gc_o48f done (full size, 12 probes, o48f final model, the 31 r48 points). Fast vs remat: max difference 4.5e-6 of max |g|,
   cosine 1.000000. Autodiff vs central finite differences (eps 4 / 2 mm), dominant axis of 5 points:
   pt 5: -289 vs -277 / -433; pt 15: +204 vs +96 / +136; pt 29: -139 vs -104 / -120; pt 3: -39.2 vs -8.9 / -9.7;
   pt 20: -36.7 vs -19.4 / -25.1. Sign correct in all 10 comparisons; FD / autodiff ratio 0.23 to 1.50 (mostly < 1).
   Reading: no sign error and the speed-up is exact, but the slope is only a local one; the objective is rough or curved
   at the 2-4 mm scale, while the design moves were 6-14 mm per round. Cause not tested (bilinear receiver sampling is one
   candidate). The poor o48f/o48b results are not explained by a wrong-signed gradient. Selection runs do not use this gradient.
33. rc30 (cell-centred 6 x 5 regular grid, 30 points, none within 10 mm of the scan edge) done 13:47: misfit 0.0717 -> 0.0301;
   D1 3.10, D2 3.86, D3 2.64 mm found; D4 not found by the rule (8.60 mm within 8 mm; 5.9 mm patch ~12 mm north of it); 89 false
   cells (min 6.9 mm); RMS vs v3 1.21 mm. Nearest points to D1-D4: 8.4 / 13.5 / 28.5 / 18.0 mm. The corner-anchored r48 with one
   more point found only D3, so its poor result was largely due to grid placement.
34. rc48 (cell-centred 8 x 6 regular grid, 48 points) done 14:21: misfit 0.076 -> 0.0304; D1 3.47, D2 3.41, D3 2.42 mm found;
   D4 not found by the rule (8.49 mm within 8 mm; 6.3 mm patch ~10 mm north-west of it); 12 false cells (min 8.4 mm); RMS vs v3
   0.55 mm. Compared at 47-48 points: r36 88 false cells / RMS 1.31, x36a 97 / 1.19, x36b 0 / 0.66. s31t started 14:21.
35. s31t (targeted selection, 31 points) done 15:04: only D3 found (3.43 mm); D1 9.42, D2 9.17, D4 9.15 mm; 209 false cells
   (min 5.1 mm); RMS vs v3 1.39 mm. Selection took 131-147 s per round. The selected points concentrate on the strongest model
   anomaly, around the unmodelled cube M1 (10 of 25 within 30 mm, nearest 0.3 mm), and form tight groups, leaving D1/D2 uncovered (nearest
   22.8 / 30.5 mm). Formal pass/fail waits for all baselines, but with rc30, x48a, x48b at 3, 4, 3 holes it is very unlikely to
   pass. No retuning (item 31). s31c started 15:04.
36. s31c (coverage selection, 31 points) done 15:47: only D3 found (3.96 mm); D1 8.94, D2 9.27, D4 8.69 mm; 136 false cells
   (min 5.4 mm); RMS vs v3 1.27 mm. 5 of 25 selected points within 30 mm of M1. Nearest points to D1 / D2 / D4 are 22.8 / 30.5 /
   13.2 mm, the same as in s31t: they are the 6 start points, so neither selection run placed a point closer to these holes.
   Neither run looks promising, so no second seed; o48r stays in the queue. Formal verdict after the last baselines (item 31).
37. Criterion check on existing layouts (CPU, 4 mm grid, 12 probes; `oed_replay.py --score-layouts`, `ldv_pmma_results/criterion_check/`).
   Seed 0, 7 layouts with 30-31 points: illumination on the background model vs holes found, Spearman +0.90 (false cells +0.39,
   RMS -0.21); log-det and any score on the v3 model show no useful link for holes. Exploratory: 12 correlations on 7 layouts.
   Probe seeds 1-3 (partial): probe noise sd 0.01-0.22 is as large as the differences between layouts (2.2-2.4), and r48 moves
   ahead of x48b. Prediction frozen in `prediction_18nov.json` before x48c, x48d, j30a, j30b were scored: best to worst
   x48c, j30a, j30b, x48d (mean illumination 2.284, 2.365, 2.412, 2.583; x48c has sd 0.22).
38. Criterion check, independent probe seeds 1-3 (mean), 7 known layouts with 30-31 points: illumination on the background model
   vs holes Spearman +0.60 (per seed 0.60 / 0.42 / 0.78; +0.52 without s31t), fewer false cells +0.71, lower RMS +0.29. Moderate
   and still exploratory. Other budgets agree only partly: the best-scoring layouts are not always those with most holes.
39. New test d30a: `--offline-steps 60`: all 30 points designed in simulation only, from rc30, background model (no data, so blind),
   4 mm grid on CPU, Adam 1.5 mm, 12 fresh probes per step; layouts snapped to scan points are compared on 4 separate validation
   probe sets and the best is kept. CPU trial: snapping can erase the gain (2.43 -> 2.67, 2 probes), hence judging after snapping.
   Pass rule, fixed now: same as item 31 (holes >= baseline median and false cells <= baseline median over rc30, j30a/b, x48a-d),
   and it must not be worse than its start rc30 on either. One run: suggestive only. Inversion runs after the 18 Nov queue.
40. a30 (adaptive, approved 18 Nov, runs overnight after the d30a inversion): rc30 split into 3 interleaved groups of 10. Round 0:
   measure group 0, 33 FWI updates. Rounds 1-2: on the CURRENT inverted model, 20 Adam steps (1.5 mm, 12 fresh probes per step) on
   the next group with measured points fixed; snapped candidates scored on 4 separate probe sets every 10 steps, best one measured;
   33 / 34 FWI updates on all measured points. 30 distinct points, 100 updates. CPU test passed (4 mm grid, 2 rounds).
   Rule fixed now: passes the item 31 rule against the baselines; adaptivity is supported only if a30 also beats d30a
   (more holes, or equal holes and fewer false cells). One run: suggestive, needs a second seed.
41. Baselines at 30-31 points complete (found = min h < 7 mm within 8 mm): rc30 3 holes / 89 false / RMS 1.21; j30a 3 / 91 / 1.10;
   j30b 2 / 55 / 1.53; x48a 4 / 22 / 1.46; x48b 3 / 468 / 1.33; x48c 3 / 64 / 1.41; x48d 3 / 78 / 1.30. Medians 3 holes, 78 false, RMS 1.33.
   Item 31 rule: s31t (1 / 209) and s31c (1 / 136) FAIL. d30a PASSES (item 39): D1 3.53, D2 3.37, D3 2.40, D4 5.46 mm, 4 holes, 38 false
   cells, no cell below 8.50 mm away from the defects, RMS 1.20; not worse than its start rc30 (3 / 89). Nearest points to D1-D4:
   21.1 / 14.5 / 21.8 / 28.4 mm. x48a is still better on false cells (22), so d30a matches the best baseline on holes, not beats it.
   One run only: needs repeats before any claim.
42. Frozen prediction (item 37) NOT confirmed: predicted order x48c, j30a, j30b, x48d; holes 3, 3, 2, 3; false 64, 91, 55, 78; Spearman with
   holes +0.26, false cells 0.00, RMS 0.00. x48d was predicted worst and was not (j30b had fewest holes). So the illumination score is not
   shown to predict layout quality, and d30a's result cannot yet be attributed to the lower score.
43. o48r (o48f's move lengths, random directions, 146 distinct points): 3 holes (D1 4.11, D2 3.53, D3 2.36; D4 8.98), 76 false, RMS 1.32.
   o48f: 3 holes (D2, D3, D4 6.51; D1 8.61), 64 false, RMS 0.82, 148 distinct. Same hole count; the design direction shows only a lower RMS.
44. Repeats of d30a queued after a30 (`src/ldv_pmma/queue_d30_repeats.sh`, GPU for design and inversion): d30b from j30a, d30c from j30b, same
   settings as d30a (4 mm design grid, 60 steps, Adam 1.5 mm, 12 fresh probes, 4 validation probe sets, snapped candidates). Rule fixed now:
   each must pass the item 31 rule against the same medians (holes >= 3, false cells <= 78) and be no worse than its own start. The d30a result
   counts as repeatable only if both pass.
45. Two-source study queued (`src/ldv_pmma/queue_s2.sh`, after the d30 repeats). The scan files are S1, S1+S2, S1+S3, S1+S4, so S1 is always driven: a true
   two-source set is file 1 plus one other file. Chosen without using defects: files 1 and 3 (S1, S3), S3 being the transducer farthest from S1 (338 mm).
   `LDV_FILES` selects files in `invert_thickness.py` and `oed_replay.py`; all four PZT blocks stay in the model as added mass, only the driven transducers
   are simulated. Default behaviour checked unchanged. Layouts `make_subsets_s2.py` (usable and no-surface from files 1, 3): s2rc30, s2j30a/b, s2x48a-d.
   Runs: s2full (all usable points), the 7 baselines, s2d30 (design from s2rc30, as d30a), s2a30 (adaptive, as a30). Rule fixed now: holes >= and false cells <=
   the medians of the 7 two-source baselines, no worse than s2rc30; adaptivity supported only if s2a30 also beats s2d30. If s2full misses holes, the comparison is weak.
46. Repeats of the simulation-only design (4 scans; rule of item 44). Validation after snapping: d30a 2.080, d30b 2.088, d30c 2.156.
   d30b (from j30a): D1 3.30, D2 3.57, D3 2.51, D4 9.46 mm, 3 holes, 83 false, RMS 1.43: FAILS (83 > 78), no worse than its start (3 / 91).
   d30c (from j30b, 29 points used): D1 3.18, D2 6.11, D3 2.54, D4 4.44 mm, 4 holes, 28 false, RMS 1.22: passes; start 2 / 55.
   Pre-set criterion "repeatable only if both repeats pass" is NOT met. Paired with their own starts, all three designs are at least as good on
   holes and better on false cells (rc30 3/89 -> 4/38, j30a 3/91 -> 3/83, j30b 2/55 -> 4/28): consistent direction, but 3 pairs only
   (one-sided sign test p = 0.125).
47. a30 (adaptive, 4 scans) done 22:07: D1 4.03, D2 3.99, D3 2.51, D4 6.57 mm, 4 holes, 62 false, RMS 1.25. Passes the item 31 rule; not better than
   d30a (4 / 38): adaptivity NOT supported (item 40). Round validation 5.171 -> 4.418 and 2.766 -> 2.664.
48. Two sources (files 1, 3 = S1, S3), item 45. s2full: 4 holes (D4 4.35), 0 false, RMS 0.80, so two sources suffice with the full scan. Baselines:
   s2rc30 3/66, s2j30a 3/476, s2j30b 1/138, s2x48a 1/42, s2x48b 3/698, s2x48c 3/570, s2x48d 1/323 (holes/false); medians 3 / 323.
   s2d30 (validation 2.797): 3 holes, 166 false, RMS 1.16; s2a30: 3 holes (D4 6.92), 94 false, RMS 1.37; in round 2 no snapped candidate beat
   the start, so the last 10 points were not moved. Both pass the medians but are WORSE than their start s2rc30 (66 false): fail item 45.
   s2a30 beats s2d30 (same holes, fewer false), but both are worse than their start, so this does not support adaptivity either.
   Note: D4 values 6.57 (a30) and 6.92 (s2a30) are just under the 7 mm threshold.
49. Illumination criterion check (19 Nov, GPU, 4 mm grid, background model, 4 probe seeds; `oed_replay.py --score-terms`, `analyse_criterion.py`,
   `ldv_pmma_results/criterion_check/terms*_p*.json`, `analysis.json`). Variance term (mean over layouts) 0.851 / 0.752 / 0.736 with 12 / 48 / 192 probes,
   sd between seeds 0.117 / 0.055 / 0.024; about 14 % of the 12-probe variance term is probe noise (extrapolated). The log-mean term varies little
   (sd 0.04 / 0.02 / 0.01), so beta between 0 and 0.2 barely changes the score; beta >= 1 trades holes against false cells.
   Ranking of real layouts (Spearman holes / fewer false / lower RMS): 8 simple baselines with 4 scans, 192 probes, beta 0.05: -0.05 / +0.36 / +0.33
   (signs change with the probe count: no reliable ranking); all 15 layouts incl. designed and selected: +0.59 / +0.55 / +0.30 (driven by the failed
   selection runs and the designed layouts); 2 sources, 9 layouts: -0.09 / +0.10 / +0.13. Conclusion: 12 probes are noisy (use >= 48 to compare
   layouts), but neither more probes nor another beta makes the criterion predict hole detection among simple layouts on this plate.
50. Fixed-sensor-count study (user, 19 Nov; `src/ldv_pmma/queue_pairs.sh`): from rc30, j30a, x48c, the paper workflow with all sensors moving each round
   (`--move-all --fwi-first`, 5 rounds x 20 FWI updates, 1 Adam step of 8 mm per round, state carried) but 48 probes (in groups of 12, `--probe-chunk`;
   CPU check: identical value and gradient), against random moves of the same lengths per point and round (`--random-moves-from`). The static control
   is each start's own baseline inversion. Rule fixed now: in all 3 pairs the adaptive run has lower RMS vs v3 than its random partner and is no
   worse on holes found and false cells; distinct positions are matched by construction (reported). Tags ad_<start>, rn_<start>.
51. 19 Nov 08:25: ad_rc30 ran out of memory at the first design step (36 GB request): grouping the probes with lax.map still stored every group's residuals for the backward pass. Fix: jax.checkpoint per probe group (CPU check: identical value and gradient). Queue restarted.
52. 19 Nov 11:05, pair 1 of 3 (rc30 start, 48 probes, 5 rounds, all 30 sensors move each round; scored post hoc):
    ad_rc30 (adaptive design) 4 holes (D1 3.45, D2 3.61, D3 2.31, D4 5.63), 101 false cells, RMS vs v3 1.28, 148 distinct points;
    rn_rc30 (random moves, same step lengths) 3 holes (D4 7.95), 75 false cells, RMS 1.41, 147 distinct points;
    static rc30: 3 holes, 89 false, RMS 1.21. Rule of item 50: adaptive better RMS (yes) and holes (yes) but more false cells (101 vs 75), so pair 1 fails the rule.
    Design step: each step moved the points 6-11 mm; the first step moved all of them by 11.3 mm (Adam first step = ±lr per coordinate).
53. 19 Nov 12:25, pair 2 of 3 (j30a start): ad_j30a 4 holes (D4 6.17), 3 false cells, RMS 0.893, 138 distinct points;
    rn_j30a 3 holes (D4 9.09), 72 false, RMS 1.022, 141 distinct; static j30a 3 holes, 91 false, RMS 1.096.
    Adaptive wins on all three measures, and beats the static start on all three. Pair 2 passes the item-50 rule. Running score: 1 of 2 pairs pass.
54. 19 Nov 15:20, pair 3 of 3 (x48c start): ad_x48c 4 holes (D4 3.91), 34 false cells, RMS 1.049, 149 distinct points;
    rn_x48c 3 holes (D4 8.73), 19 false, RMS 1.169, 142 distinct; static x48c 3 holes, 64 false, RMS 1.408.
    Fails the item-50 rule on false cells (34 vs 19). Verdict on the fixed rule: 1 of 3 pairs pass (j30a), so the claim "adaptive beats random moves" is NOT supported as pre-registered.
    Consistent across all 3 pairs: adaptive finds D4 (5.63 / 6.17 / 3.91 mm) and random never does (7.95 / 9.09 / 8.73); adaptive has lower RMS (1.28/0.89/1.05 vs 1.41/1.02/1.17); sign test 3 of 3, one-sided p = 0.125.
    Post hoc observation (not part of the rule): the shallowest false cell is always thinner in the random run (6.80/7.68/6.16 mm) than in the adaptive run (7.63/8.67/8.68 mm), i.e. adaptive false cells are milder.
    Against the static start, adaptive is better on holes in 3 of 3 and on RMS in 2 of 3 (rc30 static 1.21 beats both moving runs).
55. 19 Nov 15:30, fewer-shots study launched (user; `src/ldv_pmma/queue_fewshots.sh`, GPU, cores 62-63, nice 10). Same protocol as item 50 (48 probes, 5 rounds, all sensors move, random partner with the same move lengths), starts from the 2-shot subsets s2rc30, s2j30a, s2x48c, s2j30b, s2x48a.
    A: 1 shot = scan file 3 only (S1 + S3 firing together, LDV_FILES=3): full scan k1full and static starts k1<start>.
    B: 1-shot pairs a1_/r1_<start>; C: 2-shot pairs (files 1 and 3) a2_/r2_<start>; D: 2 extra pairs per setting (j30b, x48a) plus their 1-shot static runs. 2-shot static baselines exist (s2*, item 47).
    Assessment as in item 50 (user: keep it; rules are ways of assessing, not fixed): adaptive lower RMS vs v3 and no worse on holes and false cells than its random partner. Holes, RMS and false cells are also reported separately.
56. 19 Nov, M1 (glued steel cube) diagnostic (`src/ldv_pmma/diag_m1.py`, CPU, post hoc). The thickness-only model cannot represent a glued mass:
    v3 (the RMS reference) has |h - 10| up to 48.9 mm within 25 mm of M1 (no upper bound on h). Consequences for the scores:
    - RMS vs v3 is dominated by M1: excluding 40 mm around M1, RMS drops to 0.35-0.48 (s2full 0.80 -> 0.16). Outside that zone adaptive vs random is
      0.45 vs 0.39 (rc30), 0.35 vs 0.42 (j30a), 0.48 vs 0.41 (x48c): adaptive better in 1 of 3, not 3 of 3 (items 52-54 RMS wins were M1-driven).
    - Most false cells lie 20-50 mm from M1 (rc30 50/89, ad_rc30 80/101, rn_rc30 65/75, j30a 69/91, rn_j30a 69/72, x48c 63/64, rn_x48c 19/19).
    - The design is not attracted to M1: share of measured points within 30 mm of M1 ad 0.06/0.06/0.10, random 0.03/0.06/0.14, all scan points 0.05.
57. 19 Nov, scoring v2 (`src/ldv_pmma/score_inversions_v2.py` -> `ldv_pmma_results/inversion/scores_v2.json`; old scores.json kept). M1 found if max h > 13 mm within 8 mm
    (a glued mass appears as a thick, stiff plug: v3 58.9 mm); false thin (h < 9) and false thick (h > 11) cells counted outside 40 mm zones around M1 and D3;
    RMS vs v3 on all cells and outside the zones; RMS vs as built (10 mm, blind holes 4 mm) outside the zones (floor: full scans 0.35, resolution of the 3 mm TV map).
    Pairs, adaptive vs random (static start in brackets):
      rc30: holes 4 vs 3 (3); thin 21 vs 10 (44); thick 91 vs 32 (80); RMS v3 all 1.28 vs 1.41 (1.21); outside 0.425 vs 0.358 (0.444); built 0.507 vs 0.484 (0.521)
      j30a: holes 4 vs 3 (3); thin 1 vs 3 (35); thick 33 vs 43 (47); RMS v3 all 0.89 vs 1.02 (1.10); outside 0.300 vs 0.358 (0.395); built 0.429 vs 0.428 (0.499)
      x48c: holes 4 vs 3 (3); thin 30 vs 0 (5); thick 39 vs 78 (0); RMS v3 all 1.05 vs 1.17 (1.41); outside 0.471 vs 0.374 (0.385); built 0.542 vs 0.454 (0.448)
    Outside the M1/D3 zones adaptive beats random on RMS in 1 of 3 (j30a); holes 3 of 3 (D4). Item-50 rule with v2 measures: j30a passes, rc30 and x48c fail.
    4-shot fair baselines (r48, rc30, j30a, j30b, x48a-d) medians: RMS v3 outside 0.43, RMS built 0.51.
58. 19 Nov, scoring v2 adds RMS vs as built on all cells (D3 at h_min 1 mm, M1 at 10 mm): ad/rn/static rc30 0.795/1.001/0.743, j30a 0.801/0.777/0.791,
    x48c 1.117/1.199/1.434; full scans 1.0-1.16 (they reproduce the M1 plug, 59 mm in v3, which this measure penalises). Adaptive better than random in 2 of 3 (rc30, x48c).
    Correction: the k1rc30 numbers in item 57 came from an unfinished run (intermediate result.npz); removed, and the v2 scorer now skips runs without settings.json.
59. 19 Nov 16:10, scoring v3 (`src/ldv_pmma/score_inversions_v3.py` -> `scores_v3.json`; user: an unmodelled defect must at least be located and contained).
    Anomaly a = |h - 10|; found if a > 3 mm within 8 mm (all 5 features incl. M1); loc error = distance to the peak of a within 25 mm; false alarms a > 1 mm
    > 20 mm from every feature; background RMS of h - 10 there; spill = 90 % radius of a^2 within 60 mm of M1 / D3. Rank: found, false alarms, background RMS.
    Pairs (adaptive / random / static start): found 5/4/4 in all three pairs (random and static miss D4);
      rc30 false 240/149/233, bg RMS 0.535/0.475/0.550, spill M1 43/25/44 mm;
      j30a false 64/188/206, bg RMS 0.344/0.442/0.454, spill M1 19/30/45 mm;
      x48c false 146/179/161, bg RMS 0.577/0.534/0.759, spill M1 28/22/25 mm.
    Adaptive ranks above random in 3 of 3 (on found); fewer false alarms 2 of 3; lower bg RMS 1 of 3. Against its static start: better bg RMS 3 of 3, fewer false alarms 2 of 3.
    ad_j30a ranks first of all 29-31-point runs (5 found, 64 false alarms; d30a 102). Full scans: 3-34 false alarms, spill M1 10-16 mm. k1rc30 (1 shot, 30 static points, finished): 4 found, 267, 0.667.
60. 19 Nov, trajectory maps for the 3 pairs (`plot_oed_trajectory.py`, ldv_pmma_results/inversion/{ad,rn}_{rc30,j30a,x48c}/trajectory.png).
    Same path length per sensor (median 32-33 mm over rounds 1-4) but adaptive moves keep their direction (cos between consecutive moves 0.52/0.62/0.75)
    while random moves do not (-0.03/-0.08/-0.04): net displacement adaptive 24-29 mm vs random 13-14 mm. The random control is a random walk that stays
    near its start; the adaptive sensors sweep straight lines (Adam momentum carried over rounds). Confound for the D4 result: part of the gain may come from
    sweeping farther, not from the direction. A fairer control would keep a random heading per sensor (same path and net displacement). Not run.
    Static starts (v3, 4 shots): regular rc30 4/233/0.550, r48 2/131/0.536; jittered j30a 4/206/0.454, j30b 4/120/0.672; random x48a 5/97/0.608,
    x48b 4/520/0.799, x48c 4/161/0.759, x48d 4/155/0.574. 2 shots: s2rc30 4/234; s2j30a 4/548, s2j30b 4/301; s2x48a-d 2/165, 4/755, 4/677, 2/487.
    Decision (user): fixed-heading control not run; the confound is disclosed in the paper instead.
61. 19 Nov 16:45: with 1 shot (file 3 only) the s2 starts have fewer usable points: s2rc30 27, s2j30a 24, s2x48c 31, s2j30b 27, s2x48a 27 (of 30-31; the others
    are dropouts in file 3). The static k1 runs and the a1_/r1_ pairs use the same effective points, so comparisons within a start are fair; across starts
    they are not (k1j30a, 1 found of 5, had only 24 points). Part A finished 16:37; a1_rc30 round 0 FWI 203 s (half the 4-shot time).
62. 20 Nov 08:20, fewer-shots queue finished 03:08 with no errors (item 55). All runs scored with v3 (`scores_v3.json`); trajectory maps for all 20 a/r runs.
    Found of 5 / false alarms / background RMS, adaptive | random | static:
    1 shot:  rc30 3/337/0.594 | 4/59/0.346 | 4/267/0.667;  j30a 3/229/0.468 | 2/182/0.480 | 1/401/0.672;  x48c 4/149/0.454 | 4/339/0.775 | 4/239/0.773;
             j30b 4/88/0.371 | 4/147/0.484 | 2/123/0.670;  x48a 3/231/0.491 | 4/1638/1.555 | 2/217/0.573
    2 shots: rc30 4/182/0.442 | 4/245/0.497 | 4/234/0.557;  j30a 4/801/1.019 | 4/440/0.602 | 4/548/0.811;  x48c 5/568/0.775 | 4/449/0.677 | 4/677/0.922;
             j30b 4/217/0.529 | 3/839/0.881 | 4/301/0.780;  x48a 4/81/0.364 | 3/95/0.361 | 2/165/0.428
    Full scans: 1 shot 5/3/0.145, 2 shots 5/34/0.203.
    v3 ranking (found, false alarms, bg RMS), adaptive above random: 1 shot 3 of 5 (j30a, x48c, j30b), 2 shots 4 of 5 (all but j30a), 4 shots 3 of 3.
    New pairs only (rule fixed before they finished): 7 of 10, one-sided sign test p = 0.17; all 13 pairs 10 of 13 (p = 0.046, but the 4-shot pairs informed the rule).
    Item-50-style strict rule (lower bg RMS, no fewer found, no more false alarms): 1 shot 2 of 5 (x48c, j30b), 2 shots 2 of 5 (rc30, j30b).
    D4 with fewer shots: adaptive 2 of 10 (a1_x48c, a2_x48c), random 0 of 10, static 0 of 10. The 4-shot D4 advantage is much weaker with less data.
    Moving at all helps: against the static start, adaptive ranks higher in 8 of 10 and random in 6 of 10 (they measure about 4-5 times more distinct points).
    Failures: a2_j30a (801 false alarms, worst of its trio), a1_rc30 (misses D2 and D4, 337 false alarms).
63. 20 Nov: paper figure of all 13 pairs (`src/ldv_pmma/plot_pairs_summary.py` -> reports/2026-11_mssp_outline/fig_pairs_summary.{pdf,png}), v3 scores; storyline bullets added to storyline.md.
64. 20 Nov: tex outline (reports/2026-11_mssp_outline/main.tex) rewritten for the fixed-sensor-count claim, v3 scoring and 13 pairs; final-draft figures F7 (`plot_paper_fullscans.py`), F8 (`plot_pairs_summary.py`), F9 (`plot_paper_trajectories.py`, j30a start).
65. 20 Nov, waveform check LDV vs Mindlin vs Salvus 3-D (`src/ldv_pmma/salvus_asbuilt.py`, 8 GPU runs: pristine 9 min each, as built 20 min each
    (steel cube sets a smaller time step); `waveform_check.py`; figure `plot_waveform_check.py` -> F3_waveforms). Salvus: full 900 x 835 x 10 mm plate,
    free edges, PZT blocks 10 x 10 x 7 mm, as-built holes and cube, each scan's calibrated s, Q, sources; same source-signature fit for every model.
    Misfits (all files, all points; window 450-800 us): pristine scan Mindlin 0.028, Salvus 0.037; defect scan Mindlin pristine 0.082,
    Mindlin as built (holes as thinning on the 3 mm grid, D3 at h_min, rigid cube) 0.076, Salvus as built 0.043, Mindlin blind inversion v3 0.042.
    Reading: Mindlin is adequate for the intact plate; the as-built defects in Mindlin explain little (0.082 -> 0.076) while the same geometry in 3-D
    explains about as much as the blind inversion. The plate model's representation of the through-hole and the glued mass is the weak point.
    Colour maps for all paper figures now Crameri (vik for thickness, oslo otherwise; `src/ldv_pmma/paper_style.py`).
66. 20 Nov: paper colours switched to the categorical osloS (adaptive navy, random blue, static light; Mindlin deep blue, Salvus mid blue); F8 now written directly to figures/F8_pairs (the outline had a stale pre-oslo copy). F6 (calibration traces) taken out of the main body: the pristine row of F3 shows the same fit; calibration numbers go in a table.
67. 20 Nov: all paper-figure axes and text black (paper_style.py). F3 question "why does Mindlin miss the first arrival?": the small packet at
    570-605 us (pristine, S1, scan centre 173 mm from the source) is the S0 mode. Expected S0 arrival 508 + 173 mm / 2340 m/s (plate velocity of
    PMMA, nu 0.34) = 582 us; A0 peak about 635 us. RMS in 570-605 us relative to the A0 packet (620-660 us): data 0.108, Salvus 3-D 0.128, Mindlin 0.010.
    S0 reaches the LDV through the Poisson thickness change; the Mindlin plate is flexural only and has no S0. Its energy is about 1 % of the A0 packet,
    so it is a small part of the pristine misfit (0.028). Correction to the header line "S0/SH0 not above noise": S0 is visible in the 1.5 mm
    defect-scan and pristine data at about 10 % of the A0 amplitude.
68. 20 Nov: F1 = NeurIPS workflow schematic recoloured to oslo (reports/2026-11_mssp_outline/recolor_schematic.py; from figures/differentiable-oed-double-loop-draft.pptx -> figures/F1_loop.{pptx,pdf}): frames/arrows/text oslo blues, sources strong red, receivers gold with outline (outside the map), embedded velocity maps mapped to oslo, equations from PowerPoint's fallback pictures, Helvetica Neue -> Liberation Sans. F3: time axis on the pristine traces. Outline recompiled.
69. 20 Nov: F10 restyled (`src/ldv_pmma/plot_paper_remat.py`, osloS: workshop CPU light, this work navy, GPU limit dashed black). F1 loops in three hues on frame/title/arrows only, faint tinted fills: outer design loop oslo deep blue, lab/measurement Crameri cork green (#2A6D28), inner FWI loop neutral grey; sources red, receivers gold. Acoustic 2-D maps kept as the example (user). Outline recompiled.
70. 20 Nov: F10 now per design gradient for three settings (CPU workshop / CPU + remat / GPU + remat; src/remat/README.md numbers). Outline restructured: Part I numerical study (objectives + cost), Part II laboratory.
71. 20 Nov: F2 regenerated in osloS from the workshop runs (`src/ldv_pmma/plot_paper_numerical.py`, oed_multiseed_full_autodiff_results): 16.7 / 9.4 / 18.6 %, 10/10 each, as in the paper. Other workshop figures live on the author's computer (not on the server).
72. 20 Nov: workshop figures brought into the outline from the workshop data on the server (no new runs): F2 two panels (paired improvement; runtime vs improvement, results_per_run.csv), F2b maps (`plot_paper_numerical_maps.py`, oed_single_full_autodiff/*.npz = the NeurIPS example run, MSE 3449/3024/3040/2668; true model rebuilt from the config). The author's copies are in paper/figures and paper/reproducibility.
73. 20 Nov: F4 simplified (`reports/2026-11_specimen/make_drawing_paper.py`): front face only, single panel, no titles, osloS, PZT as 10 x 10 x 7 mm blocks, PZT not in the features list; the full as-built drawing moved to the appendix (A1). Note: the full drawing still says 'PZT disc 10 mm, 1 g' (from the 1 mm plate design); the PMMA plate has 10 x 10 x 7 mm blocks (5.3 g, geometry.json).
74. 20 Nov: F4 = (a) front-face drawing in x, y (mm) + (b) plate photo IMG_6330 resampled into the same frame (homography on D1-D4, M1, S1-S4; residuals <= 2 mm) and annotated (`make_drawing_paper.py`). Appendix A2 = labelled setup photo IMG_6341 (`make_setup_photo.py`). HEIC originals untouched; JPEG copies in LDV_photos/jpg/.
75. 20 Nov: F4 legend fixed (all 5 entries in one row under both panels, tight bbox). A2: illustrative green laser beams drawn from the three head apertures to one scan point (stated in the caption).
76. 20 Nov: A2 beams made realistic: thin core with a faint halo; exits at the visible aperture window between the heads and at the lower edge of the left and right heads' front faces.
77. 20 Nov: editable PowerPoint of A2 for the author (figures/source/A2_setup_editable.pptx, `reports/2026-11_specimen/make_setup_pptx.py`): photo locked as background; 3 beams, scan point, 5 labels and leader lines as movable shapes. To be re-imported after manual editing (soffice -> PDF, crop).
78. 20 Nov: A2 = the author's hand-edited PowerPoint (figures/source/A2_setup_editable_modified.pptx) exported with LibreOffice to figures/A2_setup_edited.pdf (single page = photo, no crop); outline recompiled. The scripted version (make_setup_photo.py -> A2_setup.pdf) is no longer used.
79. 20 Nov: A1 corrected (make_drawing_pmma_final.py): PZT sources are 10 x 10 x 7 mm blocks of 5.3 g (were drawn and listed as 10 mm discs of 1 g); square footprints, section and table updated. Terminology (user): "false alarms" -> "false indications" (NDT term, ASTM E1316: an indication caused by something other than a discontinuity) in all paper figures, the outline, the storyline and the v3 scorer description; the JSON key stays false_alarm.
80. 20 Nov: run overview (`src/ldv_pmma/run_overview.py` -> reports/2026-11_mssp_outline/run_overview.md) for Q1 (same budget, 30 points) and Q2 (gradient vs random moves). F8b rebuilt (`plot_matched_budget.py`): 4 and 2 shots, static vs adaptive (a30, s2a30) vs simulation and earlier designs; rows found / false indications / background RMS; F8 (13 pairs) kept for the supervisors' discussion.
81. 20 Nov, same-budget study (Q1) launched (user; `src/ldv_pmma/queue_q1.sh`, layouts `make_subsets_q1.py`). Budget 30 measured points.
    Adaptive = a30 protocol unchanged (3 interleaved groups of 10, group 0 as is, 20 Adam steps 1.5 mm per later group on the current model, 12 probes,
    validation every 10 steps on 4 probe sets, 33/33/34 FWI updates), starts from jittered layouts:
      1 shot (file 3, new 30-point layouts): q1a_rc30, q1a_j30a/b/c (+ static q1s_rc30, j30a-d, x30a/b on the same layouts; the old k1* statics measured 24-27 points)
      2 shots (files 1, 3): q2a_j30a/b/c/d (s2j30a/b existing, c/d new seeds 5, 6)
      4 shots: q4a_j30a/b/c/d (j30a/b existing, c/d new seeds 5, 6)
    Test fixed before launch: per shot setting, share of adaptive runs that find all 5 features (v3) vs share of static 30-point layouts
    (4 shots: rc30, r48, j30a, j30b, x48a-d; 2 shots: s2rc30, s2j30a/b, s2x48a-d; 1 shot: q1s_*), one-sided Fisher test; a30 and s2a30 count
    (same protocol). False indications and background RMS reported alongside, no pass rule. Expected about 26 h.

82. 2026-11-20 15:30 — queue_q1.sh first results scored (score_inversions_v3.py, all runs rescored). 1 shot, 30 points:
    q1a_rc30 (adaptive, 3 groups of 10, start f3rc30) found 3/5 (missing D1, D4), 452 false indications, bg RMS 0.830;
    validation before/after design: round 1 4.94 -> 4.48, round 2 9.32 -> 4.95 (20-point model uneven, min h 3.3 mm).
    Static q1s_rc30 5/5, 203 FI, 0.616; q1s_j30a 3/5 (D3, D4), 224, 0.521; q1s_j30b 4/5 (D4), 120, 0.609.
    Remaining in the queue: q1s_j30c/d, x30a/b, q1a_j30a/b/c, then the 2- and 4-shot adaptive blocks.

83. 2026-11-20 16:05 — q1s_j30c, q1s_j30d scored (v3) and added to F8b / run_overview (planned tags now appear automatically once
    scored; earlier k1* statics moved to "static, other budgets").

84. 2026-11-20 16:50 — q1s_x30a, q1s_x30b scored (v3); 1-shot static set complete (7 layouts). F8b and run_overview regenerated.

85. 2026-11-20 18:05 — q1a_j30a (1 shot, adaptive, start f3j30a) scored: 3/5 (missing D2, D3), 214 false indications, bg RMS 0.547;
    same-start static q1s_j30a 3/5 (D3, D4), 224, 0.521. Validation round 1 3.01 -> 2.69, round 2 6.76 -> 5.09 (again higher
    at the start of round 2). F8b regenerated.

86. 2026-11-20 20:40 — q1a_j30b (1 shot, adaptive, start f3j30b) scored: 3/5 (missing D3, D4), 209 false indications, bg RMS 0.767;
    same-start static q1s_j30b 4/5 (D4), 120, 0.609. 1 shot so far: adaptive 0/3 all-5 vs static 1/7. F8b regenerated.

87. 2026-11-20 21:25 — q1a_j30c scored: 2/5 (missing D1, D3, D4), 354 false indications, bg RMS 0.717; same-start static q1s_j30c
    3/5, 690, 1.222. 1-shot block complete: adaptive 0/4 find all five (found 3, 3, 3, 2; FI 452, 214, 209, 354) vs static
    1/7 (median 3 found, FI 221); one-sided Fisher p = 1.00 (pre-registered test, item 81: no gain at 1 shot). F8b regenerated.
    2-shot block started 20:56 (q2a_j30a).

88. 2026-11-21 09:15 — 2-shot block (q2a_j30a-d) and q4a_j30a/b scored (v3). Pre-registered test (item 81), adaptive only:
    4 shots a30 + q4a_j30a/b 1/3 all five vs static 1/8 (Fisher p = 0.49); 2 shots s2a30 + q2a_j30a-d 1/5 vs 0/7 (p = 0.42);
    1 shot 0/4 vs 1/7 (p = 1.00). Replicates: q4a_j30a 4/5 (D4) FI 567, q4a_j30b 4/5 (D4) 331; q2a_j30a 4/5 746, q2a_j30b/c/d
    3/5 (337, 379, 402). Same-start statics: j30a 4/5 206, j30b 4/5 120, s2j30a 4/5 548, s2j30b 4/5 301.
    Failure pattern: final-round validation at the start of round 2 jumps (up to 41 vs ~3 in round 1) with min h 1.6-2.0 mm in the
    20-point model: the last group is designed on artefacts of a weak inversion. a30 / s2a30 (5/5) both started from the regular
    rc30 groups; all jittered-start replicates fail to find D4. F8b regenerated.

89. 2026-11-21 ~10:00 — Diagnosis of the adaptive replicates (no new runs):
    - D4 is the discriminating feature (found in ~23 % of 30-point runs, all protocols). Its detection does not follow the distance
      to the nearest point (0.25 at 8-16 mm, 0.28 at 16-25 mm) nor the number of source-receiver paths crossing it.
    - In 4 of the 6 failed 2-/4-shot adaptive replicates the anomaly peak IS at D4 (1.3-1.8 mm off) but below the 3 mm threshold
      (D4_a 1.65, 1.38, 2.50, 2.51); static runs that miss D4 mostly have the peak 14-24 mm away. -> adaptive locates D4 but the
      progressive inversion (34 updates on all 30 points) leaves it weak. The full scans reach D4_a 5.5-6.2.
    - Round-2 designs on 20-point models with deep false thin spots (min h 1.6-2.0 mm; validation at start up to 9/17/41) end worst.
    Test queued (queue_reinv.sh, new; make_reinv_subsets.py, new): final 30 measured points of every adaptive run (a30, s2a30,
    q*a_*) inverted exactly like the static layouts (100 updates from calib_blind.json). Tags ri_<tag>. Starts after queue_q1.sh.
    Rule fixed now: same Fisher test as item 81 on ri_* vs static, per shot setting; FI and bg RMS reported alongside.

90. 2026-11-21 — Threshold sensitivity (post hoc, from stored peak anomalies; v3 standard stays 3 mm). Found = |h-10| > thr within
    8 mm; false indications counted at the SAME thr (bg cells > 20 mm from features). All-five, adaptive (a30 protocol) vs static:
    thr 3.0: 4 sh 1/3 vs 1/8, 2 sh 1/5 vs 0/7, 1 sh 0/4 vs 1/7;  thr 2.0: 1/3 vs 1/8, 2/5 vs 1/7, 0/4 vs 1/7;
    thr 1.5: 2/3 vs 1/8 (p 0.15), 3/5 vs 1/7 (p 0.15), 1/4 vs 1/7. Median FI at same thr, 4 shots: adaptive 17/31/55/86 vs static
    45/53/86/109 (thr 3/2.5/2/1.5); 2 shots 59/75/97/216 vs 36/56/119/222; 1 shot adaptive higher at all thr.

91. 2026-11-21 — F8c detection curves (plot_detection_curves.py, new): threshold t swept 1-5 mm, found (mean, of 5) vs false
    indications at the same t (median), per method, 4/2/1 shots; F8b kept unchanged. ri_* runs are added automatically when done.
    Reading: 4 shots adaptive above static at all t, simulation design above both; 2 shots crossing (adaptive better only at
    t < ~2 mm); 1 shot static better.

92. 2026-11-21 — Detection-radius sensitivity (post hoc; v3 keeps 8 mm). Radius 4/6/8/12/16/20 mm at t = 3 and 2 mm, mean found
    static vs adaptive: 4 shots t3 3.50/3.50/3.88/4.00/4.00/4.00 vs 4.33 at all radii; 2 shots t3 3.29-3.43 vs 3.20-3.60 (tie),
    t2 3.57-3.71 vs 4.00 at all radii (D4 0.14-0.29 vs 0.60); 1 shot static >= adaptive except t2 at 20 mm. Larger radii help
    static (mislocated D4 blobs 12-24 mm off), smaller radii help adaptive (D4 peak at 1-2 mm). Conclusions unchanged.

93. 2026-11-21 — Peak-robust illumination criterion (user: more weight on overall illumination, less on peaks). oed_replay.py gets
    --criterion logmean (default unchanged: var(h/mean) - 0.05 log mean): score = -mean(log h) = -mean(log(h/mean)) - log(mean),
    i.e. a uniformity term in which hot spots saturate and dark cells weigh, plus the overall illumination at weight 1 instead of
    0.05 (diagonal D-optimality). Motivation: item 89 (round-2 designs chase deep false thin spots; var term up to 41) and item 49
    (beta <= 0.2 has no effect; the mean term is ~20x smaller than the variance term).
    queue_logmean.sh (new; starts after queue_reinv.sh): q2l_j30a-d (2 shots), q4l_j30a-d (4 shots), same starts as q2a/q4a (paired),
    each followed by its re-inversion ri_q2l_* / ri_q4l_*. ~21 h, expected Sun 22 Nov ~17:00.
    Rules fixed now (v3, 3 mm / 8 mm): (i) paired with q2a/q4a at the same start: ranking (found, then FI, then bg RMS) sign test;
    (ii) vs static 30-point layouts: same Fisher test as item 81 and the F8c detection curve. Report all variants tried.

94. 2026-11-21 16:40 — Scored q4a_j30c (5/5, FI 292, D4_a 5.69), q4a_j30d (2/5 miss D1, D2, D4; FI 261); C3 complete. 4 shots adaptive
    (a30 protocol) all-five 2/5 vs static 1/8. First re-inversions (C4, item 89): ri_s2a30 4/5 (D4) FI 283; ri_q2a_j30a 4/5 717;
    ri_q2a_j30b 3/5 306; ri_q2a_j30c 4/5 (D2) 353; ri_q2a_j30d 4/5 277; ri_a30 3/5 (D2, D4) 144. Re-inversion lowers FI in 6/6, changes
    found by -1/0/0/+1/+1/-2; D4 amplitude moves erratically (0.6-5.5 mm), i.e. D4 at 30 points is marginal and depends on the
    inversion path. F8c: 2-shot re-inverted curve above static at every threshold (n = 5). F8b, F8c, run_overview regenerated.

95. 2026-11-21 18:15 — 4-shot re-inversions: ri_q4a_j30a 4/5 (D4) FI 150 (was 567); ri_q4a_j30b 4/5 (D2; D4 now found, 4.06) FI 164 (was 331);
    ri_q4a_j30c 5/5 FI 90 (was 292), bg 0.35. With ri_a30 (3/5, 144): FI lower in 10/10 re-inversions so far (2 and 4 shots).
    F8c regenerated (ri_q4a_j30d still running).

96. 2026-11-21 18:30 — Progressive static control (user): queue_progressive_static.sh (new). Every adaptive start measured as its 3
    interleaved groups of 10 with --no-design and the same inversion schedule (33/33/34, FWI first): s2n30, q2n_j30a-d (2 shots),
    n30, q4n_j30a-d (4 shots), q1n_rc30, q1n_j30a-c (1 shot). ~5.3 h, after queue_reinv.sh; queue_logmean.sh now waits for it
    (restarted, nothing had run). Rule fixed now: paired sign test, adaptive (q*a, a30, s2a30) vs no-design at the same start, on the
    v3 ranking (found, then FI, then bg RMS), per shot setting and pooled; F8c gets a "static, progressive" curve.

## 97. Re-inversion block (C4) complete; first progressive static control (21 Nov, 20:30)

Scored v3: ri_q4a_j30d 1/5 (D1, D2, D4, M1 missed) FI 387 (as run 2/5, 261); ri_q1a_rc30 4/5 (D4) FI 346 (as run 3/5, 452);
ri_q1a_j30a 4/5 (D2) FI 284 (3/5, 214); ri_q1a_j30b 4/5 (D4) FI 145 (3/5, 209); ri_q1a_j30c 3/5 (D3, D4) FI 254 (2/5, 354).
Over all 14 re-inversions: FI lower in 12/14, found higher in 6, equal in 5, lower in 3 (a30, ri_s2a30, ri_q4a_j30d).
At 1 shot re-inversion lifts every adaptive run by one feature (mean 2.75 -> 3.75, static 3.4 mean, 1/7 all five).
At 4 shots ri_q4a_j30d collapses, so the re-inverted mean at 3 mm drops below the adaptive-as-run and static curves.
Progressive static control s2n30 (pairs s2a30): 3/5 (D2, D4) FI 333 vs adaptive s2a30 5/5 FI 309 -> pair 1 of 14 to adaptive.
F8c regenerated (panel label of the progressive group fixed), main.tex recompiled.

## 98. Progressive static control (C4b) complete; 2-shot log-criterion block (C5) complete (22 Nov, 09:10)

C4b (no design, same 3 groups, same 33/33/34 inversion), v3 found/FI, paired with C3 adaptive at the same start:
- 4 shots: n30 4/232 vs a30 5/196 (A); q4n_j30a 4/167 vs 4/567 (N); q4n_j30b 4/208 vs 4/331 (N); q4n_j30c 3/403 vs 5/292 (A); q4n_j30d 3/470 vs 2/261 (N) -> adaptive 2/5
- 2 shots: s2n30 3/333 vs s2a30 5/309 (A); q2n_j30a 3/707 vs 4/746 (A); q2n_j30b 4/370 vs 3/337 (N); q2n_j30c 3/214 vs 3/379 (N); q2n_j30d 2/339 vs 3/402 (A) -> 3/5
- 1 shot: q1n_rc30 3/328 vs 3/452 (N); q1n_j30a 2/277 vs 3/214 (A); q1n_j30b 4/203 vs 3/209 (N); q1n_j30c 2/896 vs 2/354 (A) -> 2/4
Pre-registered paired sign test (item 96, ranking found then FI): pooled 7/14, p = 0.60 -> no effect of the design. All five found:
adaptive 3/14 vs progressive static 0/14 (one-sided Fisher p = 0.11). D4 missed in 11/14 controls.
C5 2-shot (log criterion), vs C3 at same start: q2l_j30a 3/666 vs 4/746 (old); q2l_j30b 3/301 vs 3/337 (new); q2l_j30c 4/351 vs 3/379
(new); q2l_j30d 3/276 vs 3/402 (new) -> new 3/4; all five 0/4. Vs progressive static at same start: log wins 3/4.
Re-inverted: ri_q2l_j30a 4/720, b 3/219, c 4/228, d 3/327. D4 missed in all 8.
Behaviour check: validation decreases monotonically in both design rounds, no jumps; round-2 points move (median 9-12 mm),
whereas the old criterion often rejected the round-2 design (q2a_j30a: 0 mm move); no drift towards the sources.
Added C4b and C5 tags to run_overview (excluded from the Q1 Fisher line) and a log-criterion curve to F8c; legend in 3 columns.

## 99. First 4-shot log-criterion run (22 Nov, 13:20)

q4l_j30a 4/5 (D4), FI 125, bg 0.43; ri_q4l_j30a 4/5 (D4), FI 122. Same start: old criterion q4a_j30a 4/5 FI 567,
progressive static q4n_j30a 4/5 FI 167 -> log criterion best of the three (same features, fewest false indications).
Validation decreases in both rounds, no jumps. Design is ~2x slower at 4 shots (4280 s per round): one pair (run + ri) takes ~3 h 30,
so q4l_j30b ~15:25, q4l_j30c ~19:00, q4l_j30d + ri ~22:45. F8c regenerated.

## 100. Second 4-shot log-criterion run (22 Nov, 15:20)

q4l_j30b 3/5 (D2, D4), FI 123, bg 0.52; ri_q4l_j30b 2/5 (D1, D2, D4), FI 169. Same start: old criterion q4a_j30b 4/5 FI 331,
progressive static q4n_j30b 4/5 FI 208 -> log criterion has the fewest false indications but loses D2, so it loses both pairs
on the found-first ranking. Running tally of log vs old criterion at the same start: 4/6. D4 missed in all 12 log runs (incl. ri).
Remaining: q4l_j30c + ri ~18:15, q4l_j30d + ri ~21:45. F8c regenerated.

## 101. Third 4-shot log-criterion run (22 Nov, 19:20)

q4l_j30c 5/5, FI 449, bg 0.85; ri_q4l_j30c 5/5, FI 267, bg 0.67. First log-criterion runs to find D4.
Same start: old criterion q4a_j30c 5/5 FI 292 (old wins on FI), progressive static q4n_j30c 3/5 FI 403 (log wins).
Tally log vs old at the same start: 4/7; log vs progressive static: 5/7. Validation decreased in both rounds, no jumps.
q4l_j30d started 18:55; with its re-inversion done ~22:30. F8c regenerated.

## 102. 1-shot log-criterion block queued (22 Nov, 19:40)

To have the log criterion at all three shot settings: queue_logmean_1shot.sh (waits for "logmean queue finished"), LDV_FILES=3,
same starts and groups as q1a_* / q1n_* (f3rc30, f3j30a-c), same protocol and --criterion logmean, each run followed by its
re-inversion. Tags q1l_rc30, q1l_j30a-c, ri_q1l_*. ~1 h 50 per pair -> done ~06:00 Mon. Not added: j30d at 1 shot, simulation
design at 1/2 shots. Tests on completion: log vs old (q1a) and vs progressive static (q1n), paired, as item 93/96.

## 103. All queues finished; final scores and pre-registered tests at 30 points (23 Nov, 09:50)

Last runs (v3 found/FI): q4l_j30d 2/401, ri_q4l_j30d 3/341; q1l_rc30 3/434, q1l_j30a 1/360, q1l_j30b 3/293, q1l_j30c 4/351;
ri_q1l_rc30 3/209, ri_q1l_j30a 1/299, ri_q1l_j30b 3/209, ri_q1l_j30c 3/267. Log-criterion designs behaved (validation decreases,
no jumps, median moves 6-12 mm, at most 1 point off-surface).
Summary per shot setting (all five found; mean found; median FI):
- 4 shots: static 1/8, 3.88, 158 | adaptive old 2/5, 4.00, 292 | log 1/4, 3.50, 263 | progressive static 0/5, 3.60, 232 | ri old 1/5, 3.40, 150
- 2 shots: static 0/7, 3.43, 487 | old 1/5, 3.60, 379 | log 0/4, 3.25, 326 | progressive 0/5, 3.00, 339 | ri old 0/5, 3.80, 306
- 1 shot:  static 1/7, 3.43, 221 | old 0/4, 2.75, 284 | log 0/4, 2.75, 355 | progressive 0/4, 2.75, 302 | ri old 0/4, 3.75, 269
Paired (found first, then FI), pooled over shots, one-sided sign tests:
- adaptive old vs progressive static (item 96): 7/14, p = 0.61
- log vs progressive static: 6/12, p = 0.61; log vs old (item 93): 6/12, p = 0.61
- ri old vs progressive static: 11/14, p = 0.029 -- but this mixes inversion schedules
- ri (old + log) vs the same start layout as a plain static (same 100-update inversion; starts with a 30-point static at
  that shot setting): adaptive points win 5/18 (old 3/10, log 2/8) -> the static start layout is at least as good.
Fisher (all five vs static) not significant anywhere (p >= 0.31).
Conclusion: at 30 points the illumination design gives no detectable gain over a regular/jittered layout with either criterion;
the progressive inversion costs about as much as the design gains (ri > progressive). D4 (deepest-off-axis blind hole) is the
limiting feature in almost every run. Overview, F8b, F8c regenerated; main.tex recompiled.

## 104. Second look at all results: where does design help? (23 Nov, 10:40; post hoc, exploratory)

Script: src/ldv_pmma/explore_views.py (reads scores/maps only). Views beyond v3:
1. Contrast of the weakest feature (min feature peak / 95th percentile of the map's own background), median per group:
   4 shots static 1.03, progressive 0.85, adaptive 1.25, re-inverted adaptive 3.14, simulation design 5.07, gradient moves 5.59,
   random moves 1.60; 2 shots static 0.31, adaptive 1.08, ri 0.95. D4 peak: 4 shots static 0.8 mm vs ri 2.8, gradient moves 4.4.
   -> designed point sets image D4 above their background, but below the absolute 3 mm v3 threshold (amplitude underestimated).
2. Detection at a fixed false-indication rate (threshold = 99th percentile of each map's background; CFAR-type), mean found:
   static 3.88 / 3.43 / 3.43 (4/2/1 shots); ri 4.20 / 4.00 / 4.00; gradient moves 5.00 / 4.40 / 4.20; random moves 4.33 / 3.80 / 3.40.
   All five: gradient moves 7/13 vs random moves 1/13. Paired pooled: ri vs progressive static more 10, same 3, fewer 1 (p = 0.006);
   ri vs same-start static more 5, same 4, fewer 1 (p = 0.11); adaptive-as-run vs progressive 5/6/3 (p = 0.36).
   At the 95th percentile: ri vs progressive 8/5/1 (p = 0.02), ri vs same-start static 4/6/0 (p = 0.06).
3. Map RMSE vs the full scan of the same shots: gradient moves lowest of the 30-point-final groups at every shot setting
   (1.05 / 1.08 / 0.92 mm vs static 1.31 / 1.25 / 1.09), but they visited 116-153 distinct positions (5 rounds of 30).
Reading: the design places points that make the weak feature stand out (contrast, CFAR detection), the progressive inversion
loses that, and the fixed 3 mm amplitude rule hides it. Not pre-registered: report as exploratory alongside items 93/96.
Moving runs: subsets ldv_work/prepared/subsets/{ad,rn,a2,r2,a1,r1}_*.npz keep 'ever' (all visited positions) -> joint re-inversion possible.

## 105. F8d (fixed false-indication rate) and proposal A queued (23 Nov, 14:10)

F8d: src/ldv_pmma/plot_fixed_rate.py -> figures/F8d_fixed_rate.{pdf,png,json}; per-run threshold = 99th / 95th percentile of the
map's own background anomaly (CFAR-type), all 30-point groups + moving arrays; added to main.tex after F8c (F8c caption updated
to the final pre-registered result). Exploratory, labelled as such.
Proposal A: make_ever_subsets.py -> ev_<run> (all distinct positions visited by moving runs ad_/rn_ at 4 shots, a2_/r2_ at
2 shots; 138-153 points) and mj4_/mj2_<start> (static jittered layout, n matched to the gradient run, grid 13x11 to 14x11,
seed 100+k). Random runs visited 1-9 fewer distinct positions (re-measured positions). queue_ever.sh: 24 inversions, 100 updates
from calib_blind, started 13:46, 4-shot block ~18:45, all ~23:15. Plan for scoring: v3 + fixed-rate (1 %, 5 %); paired
gradient vs random vs same-n static per start. 1-shot block not queued (could add 15 runs, ~4.5 h).

## 106. Regular reference for proposal A (23 Nov, 14:30)

make_regular143.py -> rg4_143 (files 1-4), rg2_143 (files 1, 3): cell-centred regular 13 x 11 grid, 143 points (within 10 of
every ev_/mj run). queue_regular143.sh waits for "ever queue finished"; ~50 min -> all done ~00:05. Added as a column of F8d.
Expectation stated beforehand: at ~145 points (4 shots) the regular grid is near saturation (r24, 110 points: 5/5, FI 26),
so A is read mainly as gradient-visited vs random-visited positions; the regular grid is the reference, not the target.

## 107. Robustness of the fixed-rate (CFAR) view (23 Nov, 15:00)

src/ldv_pmma/explore_blind_cfar.py. Item 104/105 set the threshold from background cells > 20 mm from the known features
(not possible blind). With the threshold from the whole ROI (blind) and rates 0.5-5 %, paired (more/same/fewer, sign p):
- ri vs same-start static (the clean design test, same inversion): feature-excluded 4/6/0 .. 5/4/1 (p 0.06-0.11, 0.64 at 0.5 %);
  blind 4/5/1 (5 %), 1/5/4 (2 %), 1/7/2 (1 %), 3/3/4 (0.5 %) -> no design effect once the threshold is blind.
- ri vs progressive static: 10/3/1 (p 0.006) feature-excluded at 1 %; blind 9/3/2 (5 %, p 0.03), 8/4/2 (1 %, p 0.055), 4/6/4 (0.5 %)
  -> mostly measures the cost of the progressive inversion, not the design.
- adaptive as run vs progressive: never significant (p >= 0.11).
- moving arrays, all five, gradient vs random (of 13): feature-excluded 7/7 (5 %), 7/1 (2 %), 7/1 (1 %), 6/0 (0.5 %);
  blind 7/2, 5/0, 2/0, 0/0 -> direction holds at every setting, size depends on the rate.
Conclusion: the 30-point "design helps" reading of item 104 does not survive a blind threshold; the gradient-vs-random
move result (Q2) does, in direction. F8d as built uses the feature-excluded background: switch to blind or drop.

## 108. Re-inverted adaptive points vs static 30-point layouts (23 Nov, 15:30; exploratory)

Decision (user): keep the feature-excluded background for the fixed-rate rule (in practice a defect-free scan gives the
background level). src/ldv_pmma/explore_ri_vs_static.py; found counts only; unpaired = Mann-Whitney on values centred on the
static mean of each shot setting; same start = paired with the static run of the same start layout (more/same/fewer).
- Old criterion (ri_a30, ri_q*a_*; 14 runs) vs static (22 runs): rate 1 %: 4.20/4.00/4.00 vs 3.88/3.43/3.43 (4/2/1 shots),
  all five 4/14 vs 2/22, MWU p = 0.022, same start 5/4/1 (p 0.11); rate 5 %: MWU p = 0.047, same start 4/6/0 (p 0.06);
  v3: 3.40/3.80/3.75 vs 3.88/3.43/3.43, MWU p = 0.24, same start 1/6/3.
- Log criterion (ri_q*l_*; 12 runs): no gain under any rule (MWU p >= 0.62; same start 0-1 more, 2-5 fewer).
- Both criteria pooled: MWU p 0.18-0.62.
Reading: the old-criterion design + re-inversion beats static 30-point layouts at a fixed false-indication rate (p ~0.02-0.05,
not corrected for the views tried), at 2 and 1 shots clearly, at 4 shots on all-five (3/5 vs 1/8) but not at 5 %; not under v3.
The log criterion does not reproduce it.

## 109. Adaptive as run vs progressive static: same scan cost and same inversion (23 Nov, 15:50; exploratory)

src/ldv_pmma/explore_adaptive_vs_prog.py (found counts; unpaired MWU centred per shot; paired by start).
Old criterion (14 pairs): all five 3/14 vs 0/14 under v3 and at 1 %; mean found higher at 4 and 2 shots under every rule
(e.g. 1 %: 4.20 vs 3.80, 3.60 vs 3.20), equal at 1 shot. MWU p = 0.21 (1 %), 0.032 (5 %), 0.13 (v3); paired 5/6/3, 8/3/3, 6/5/3
(p 0.36, 0.11, 0.25). Log criterion (12 pairs): no gain (MWU p 0.17-0.83, paired 3/3/6 to 5/4/3).
Note: the pre-registered item-96 test (7/14) ranks on found then false indications; on found alone v3 gives 6/5/3.

## 110. Relaxed case: equal inversion cost only, moving array vs static 30 points (23 Nov, 16:20; exploratory)

src/ldv_pmma/explore_moving_vs_static.py. Moving arrays: 5 rounds x 20 updates = 100 (as static), final 30 points, ~145 positions
measured (5x the scan). Paired with the static run of the same start (s2* at 2 shots, k1* at 1 shot).
Gradient moves (13 runs) vs static 30 (22): 1 %: 5.00/4.40/4.20 vs 3.88/3.43/3.43, all five 7/13 vs 2/22, MWU p = 0.001,
paired 11/2/0 (p < 0.001); 5 %: MWU p = 0.032, paired 6/6/1 (p 0.06); v3: 5.00/4.20/3.40 vs 3.88/3.43/3.43, all five 4/13 vs 2/22,
MWU p = 0.021, paired 8/4/1 (p 0.02). v3 FI median 146/217/229 vs 158/487/221.
Random moves (same 5x scan): 1 % MWU p = 0.18 (paired 5/8/0), 5 % p = 0.08, v3 p = 0.22 (paired 4/8/1); all five 1/13 (1 %),
0/13 (v3). -> the gain over static comes from the move direction, not from the extra measurements.
Storyline candidate order (user): high-level first; three cases: (i) equal inversion cost (this item), (ii) equal scan cost with
re-inversion (item 108), (iii) equal scan + inversion cost (item 109); plus Q2 gradient vs random (pre-registered).

## 111. Three cases registered in the storyline; case (i) vs dense static (23 Nov, 16:45)

storyline.md: new section "Lab result at small budgets: three cases" with tables (i) equal inversion cost, (ii) equal scan cost
with re-inversion, (iii) equal scan + inversion cost. ev_ad_rc30 (all 147 positions visited by ad_rc30, joint inversion): 5/5,
FI 91, D4 5.6 mm. Case (i) vs dense static, 4 shots: gradient final 30: 5/5 in 3/3, FI 64/146/240, D4 3.8-6.1; regular r24 (110)
5/5 FI 26 D4 4.9; regular r18 (176) 5/5 FI 52; random x24a (110) 4/5 FI 133 (D4 0.4, missed at 1 %). Matched ~145-point statics
(mj4_*, rg4_143) still running.

## 112. F10a (forest plot, three cases) and F10b (paired slopes) (23 Nov, 17:20)

src/ldv_pmma/plot_funnel.py -> figures/F10a_funnel, F10b_slopes; added to main.tex after F8d. Pooled gain (designed-n weighted
mean of per-shot mean differences, 95 % bootstrap within group and shot, 5000 draws), 1 % rule / v3:
(i) gradient moves vs static 30: interval excludes 0 under both rules; random-move control and (ii)/(iii) intervals include 0;
log-criterion rows negative. F10b pairs: (i) gradient 11/2/0, random 5/8/0; (ii) 5/4/1; (iii) 5/6/3.

## 113. F10a redrawn as two panels (23 Nov, 17:40)

Left 1 % rule (exploratory), right v3 (pre-registered), same rows; explicit legend (pooled gain + 95 % bootstrap interval; open
markers = gain at 4 / 2 / 1 shots on sub-rows). Both rules kept: (i) excludes 0 under both, (ii) only under 1 % (marginal),
(iii) under neither. Caption in main.tex updated.

## 114. Design controls queued (23 Nov, 15:35)

oed_replay.py: new opt-in flag --design-on-background (design every round on the initial background model, m = 0, instead
of the current inversion; default unchanged). make_ever_subsets.py: per-run mode (ev_<RUN> from the run's 'ever' mask and
active files). queue_design_controls.sh (waits for "regular143 queue finished"):
1. simulation design (as d30*/s2d30: 60 Adam steps of 1.5 mm on the background, then static inversion): s2d30b (s2j30a),
   s2d30c (s2j30b), q1d_rc30/j30a/j30b (f3*) -> 3 per shot setting with d30a-c / s2d30; for case (ii)/(iii) of F10a.
2. offline moving array: ad_/a2_/a1_ protocol with --design-on-background, same starts: od_{rc30,j30a,x48c} (4 shots),
   o2_{rc30,j30a,j30b,x48a,x48c} (2 shots), o1_{...} (1 shot) -> control for adaptivity in case (i).
3. ev_ of every offline moving run (joint inversion of all visited positions) -> case (iv).
Expected: A ~22:25, regular ~23:15, simulation block ~01:15, offline moving ~12:00 Tue, ev ~16:30 Tue.
User note: simulation design and offline moving array may be left out of the paper if the storyline gets too big.

## 115. Paper draft started (23 Nov, evening)

reports/2026-11_mssp_paper/: neutral article wrapper (main.tex) + one file per section in sections/ (no preamble, to be
\input by the thesis template or a journal class later; siunitx used if installed, stand-ins otherwise). Figures read
from ../2026-11_mssp_outline/figures. First section: sections/sec7_1_protocols.tex (replay of the dense scan, static
baselines, moving array + random control, 3x10 adaptive protocol, controls, larger budget / offline checks, Table of the
three cases, outcome measures incl. fixed false-indication rate, statistics and what was fixed in advance).
Order agreed with the user: 7.1, then 4, 5, 6, then 2, 3.

## 116. Section 7.1 restructured; F11 strategies schematic; Appendix A (23 Nov, evening)

User: 7.1 read like a list. New structure: replay in 3 sentences; three choices (what is limited: scan budget vs number of
sensors; who places the points: no design / designed beforehand / adaptive; how the map is made: one pass vs staged);
Table of strategies (2 x 3 matrix, run counts per shot); progressive static, re-inversion, all visited positions as variations
of the third choice; the three cases in one paragraph; scoring and pre-registration in 4 sentences. All settings, random-move
net-displacement caveat, larger-budget details, second criterion and statistics moved to sections/appA_protocols.tex.
F11 (src/ldv_pmma/plot_strategies.py -> figures/F11_strategies): (a) adaptive batches of a30, (b) gradient moves ad_j30a,
(c) random moves rn_j30a, real positions in the scan frame, feature outlines for orientation, step sequence (M/I/D) below.

## 117. All paper sections created as stubs with notes (23 Nov, evening)

reports/2026-11_mssp_paper/sections/: sec1_introduction, sec2_method (with Eq. illumination placeholder), sec3_numerical,
sec4_experiment, sec5_model, sec6_fullscan_scoring (+ subsection Scoring), sec7_1_protocols (written), sec7_2_case_i,
sec7_3_case_ii, sec7_4_case_iii, sec7_5_paths, sec8_discussion, sec9_conclusion, appA_protocols (written). Each stub has a
red "notes" block (environment in main.tex, drafting only) with the facts and log items it needs; e.g. the
defect-free-background argument for the fixed-rate rule is in sec6 (Scoring). All cross-references resolve.

## 118. Design controls and run A scored: adaptivity not supported; designed sweeps vs random (24 Nov, 18:30; exploratory)

All queues finished 16:45 (no failures). 40 runs scored (v3 + fixed rate). src/ldv_pmma/explore_controls.py.
Offline moves (--design-on-background) behaved: moves 11.3 / ~10 / 7-8 / 6-8 mm, 119-148 distinct positions.
Case (i), final 30 points, mean found 4 / 2 / 1 shots at 1 %: static 3.88 / 3.43 / 3.43; gradient 5.00 / 4.40 / 4.20;
offline 4.67 / 4.40 / 4.20; random 4.33 / 3.80 / 3.40. Paired: gradient vs offline 2/10/1 under every rule (no adaptivity
effect); offline vs random 7/5/1 (1 %, p 0.035), v3 7/4/2; offline vs static same start 11/2/0 (1 %), 7/5/1 (v3, p 0.035).
Sweep geometry (net displacement / path, median per sensor; cos between consecutive moves): gradient 23-27 / 32-35 mm, cos
0.71-0.95; offline 28-35 / 32-38 mm, cos 0.95-0.99; random 12-15 / 30-33 mm, cos 0. -> both designed variants sweep straight
and far, random moves wander; designed vs random therefore still confounds direction with sweep distance.
Case (ii), simulation design (3 per shot): 4.67 / 3.33 / 2.67 at 1 % (v3 4.67 / 3.33 / 3.00) vs static 3.88 / 3.43 / 3.43:
no gain (same-start 3/2/4, MWU p 0.34); good only with 4 shots. Re-inverted adaptive vs simulation, same start: 4/4/1 (1 %),
5/3/1 (5 %, p 0.11), 2/4/3 (v3): weak hint in favour of adapting at a fixed budget.
Case (iv), all visited positions (~145, one pass), 4 / 2 shots, v3 mean found (FI median): gradient 5.00 / 4.60 (66 / 137);
offline 4.67 / 4.60 (45 / 214); random 4.67 / 4.20 (92 / 317); static jittered same size 4.33 / 4.80 (32 / 54); regular 143
5 / 5 (90 / 116). Gradient-visited vs static same size: 2/5/1 (v3), 0/6/2 (1 %); FI higher in 7 of 8 starts. -> with all
data inverted at once, a uniform layout of the same size is as good or better: the walk's advantage exists only when each
stage sees the current 30 sensors.
1-shot offline-visited: 4.40 v3 (2/5 all five), FI median 20.

## 119. Framing agreed; storyline, paper notes and F10a updated (25 Nov)

User: keep the three-case storyline; adaptive is justified because a design planned beforehand needs a calibrated model and
a design run per specimen/geometry (our offline control had a best-case model calibrated on the full defect scan); random
moves sweeping less is a point for the design; no fixed-heading test (keeps the paper focused); irregular geometries as a
discussion point. storyline.md: section "Framing agreed 25 Nov". Paper notes updated (sec1, sec7_2, sec7_3, sec8).
F10a (plot_funnel.py): rows (i) gradient / planned beforehand (blue) / random; (ii) re-inverted / designed beforehand (blue);
(iii); (iv) all visited by gradient / random moves vs same-size static (4 and 2 shots). Log-criterion rows moved to the
appendix. Outline caption updated. Deadlines: supervisor draft Fri 27 Nov; poster ~9 Dec.

## 120. One paper directory; figures placed; cost figure dropped (25 Nov)

reports/2026-11_mssp_outline merged into reports/2026-11_mssp_paper and removed: figures/ (all plotting scripts in
src/ldv_pmma and reports/2026-11_specimen now write there), LDV_photos/, notes/ (storyline.md, run_list.md,
run_overview.md; run_overview.py writes to notes/), scripts/ (recolor_schematic.py; make_schematics.py marked obsolete),
outline/ (archived outline_main.tex/.pdf, not edited). Earlier log items keep the old paths (historical).
Figures placed in the sections (captions from the outline, F-numbers removed): loop (sec2), numerical + example (sec3),
specimen (sec4), forward check (sec5), full scans (sec6), strategies (7.1), funnel + slopes (7.2), paths 4 shots (7.5);
Appendix B (sections/appB_figures.tex): 13 pairs, paths 2/1 shots, fixed-budget figures, drawing, setup photo.
Cost figure (F10) and CPU/GPU benchmark dropped (user: the NeurIPS implementation did not use remat); remat kept as 2-3
sentences in sec2 (142 GB -> ~3 GB on the plate). Draft: 21 pages, compiles.

## 121. Main figure set implemented in the paper (25 Nov)

Main text (in order, Fig. 1-9): design loop (sec2); numerical study = objective comparison (a, b) + example run (c-g) merged
into one figure (sec3; plot_paper_numerical_maps.py panels re-lettered c-g); specimen (sec4); forward check (sec5); full-scan
maps (sec6); acquisition strategies (7.1); funnel (7.2); paired slopes (7.2, chosen over the 13-pair figure); sensor paths
4 shots (7.5). Appendix B: 13 pairs, paths 2 / 1 shots, fixed-budget figures (matched budget, detection curves, fixed rate),
specimen drawing, setup photo. Cost figure dropped (item 120). Compiles, no undefined references.

## 122. Unused figures removed from the paper (25 Nov)

Appendix B now holds only the 2- and 1-shot sensor paths (Figs. 10, 11; referenced from 7.5), the specimen drawing and the
setup photo. Removed (files kept in figures/): F8_pairs (13 pairs), F8b, F8c, F8d (fixed-budget figures).


## 123. Handoff note for a local session (25 Nov)

reports/2026-11_mssp_paper/notes/handoff.md: where everything is, agreed storyline and wording, figure set with
generating scripts, status, next steps and deadlines (supervisor Fri 27 Nov, poster ~9 Dec), writing conventions, what
needs eegdevtest, git workflow on branch mssp-paper. Paper .gitignore re-includes figures/ (root .gitignore drops *.pdf, *.png).
