# Paper storyline and decisions (living note)

Started 16 Nov 2026. Companion to `main.tex` (outline with figure drafts). Update this note whenever a decision changes.

## The claim

For a laser-vibrometer (LDV) scan of a plate, an adaptive design picks where to measure while full-waveform inversion
runs. It finds the defects from a fraction of the dense scan's points, without knowing where the defects are.

**Cost = number of measured points** (scan time is about 4 s per point; the dense defect scan has 6418 points, about 7 h).

## Storyline

1. **Problem: scan time.** A dense LDV scan takes hours, and which points matter is unknown in advance.
2. **Method: design while inverting** (workshop paper, extended). Rounds of measure → a few FWI updates → move the
   points with a design objective that needs no true model. Main objective: the stochastic illumination criterion of
   the paper (Rademacher probes, var(h/mean) − β·log(mean)).
3. **Synthetic evidence** (controlled, many seeds): the method works when the model is exact. See "Synthetic studies" below.
4. **Forward model is cheap and good enough.** Damped Mindlin plate (A0) on GPU with rematerialisation. It matches
   Salvus 3-D, and after calibration it fits the pristine PMMA scan (correlation 0.98, misfit 0.041).
5. **Lab data, fully blind pipeline.** 10 mm PMMA plate, 4 PZT sources, holes D1–D4, steel cube M1 (unknown target).
   - Background calibration on the defect scan itself, blind: robust reweighting (IRLS) down-weights high-residual
     points from the data alone. "No surface" points (beam inside a through-hole) are found from amplitude alone.
   - Full-scan blind inversion v3: D1, D2, D4 found (min thickness 3.0 / 3.7 / 3.8 mm; truth 4 mm left), D3 through-hole
     shows as a thinning (2.5 mm), cube found as a stiff/heavy target, no false spots. Misfit 0.088 → 0.047.
   - It is as good as the oracle run v2 (calibration with known defects, cube modelled), so no prior knowledge is needed.
   - A pristine scan is not required: the pristine-only background (v4) finds the same defects but adds false thin
     bands at the edges. Pristine scan = optional extra, not part of the method.
6. **Main result: fewer points, same defects.** Replay the dense scan: regular, random and adaptive OED layouts with the
   same number of measured points (≈ 6 %, 11 %, 25 % of the scan). Headline = the share of points at which the
   adaptive layout matches the full-scan v3 map, and how much regular/random lose there.
7. **Limits stated openly.** No plate edges in the model (data after ≈ 800 µs unusable), only vz used, thickness is the
   only unknown (the cube appears as a thickness artefact), sources are bonded blocks modelled as added mass.
8. **(Optional) live closed loop** with the LDV driven by the optimiser.

## Decisions and why

| Decision | Why |
|---|---|
| Every result-producing step is blind to the defects; oracle runs only as an upper bound | real-world use: defect positions are unknown |
| Background from blind self-calibration on the defect scan (`calib_blind.json`), same for all layouts | v4 showed the pristine values leave false anomalies; same background keeps the layout comparison fair |
| Cube treated as an unknown target, not modelled | blind story; it still shows up |
| Time window 450–800 µs, 30 µs taper | edge reflections arrive after ≈ 800 µs, edges not modelled |
| Thickness h = 1 + 9·exp(m) mm (floor 1 mm) | hole depth unknown, through-holes allowed |
| vz only | in-plane components not used yet (possible extension) |
| 100 inversion iterations for the reduced-scan runs | v3 converged by 75–100 iterations |
| OED counting rule (b): N/5 new points per round, all points kept | every method gets exactly N measured points; pending the memory/speed test |

## Synthetic studies

- **Workshop result (exists):** 2-D acoustic, 10 seeds, 30 receivers, 5 design rounds, methods fixed / illumination /
  waveform misfit / oracle (`oed_multiseed_full_autodiff_results/scientific`, about 4.2 h GPU in total). Protocol: the
  30 receivers move every round, FWI uses only the current round's data, so the OED measures up to 150 positions against
  30 for the fixed baseline. Fair for "30 sensors at a time", not for "points measured".
- **Needed for consistency with the LDV study (if rule (b) is adopted):** new script (paper scripts stay frozen) with
  N/5 new receivers per round and accumulation, rerun illumination + fixed baseline (+ random) at equal measured points.
  Estimate 1–2 h GPU. Keep the workshop result as the "fixed sensor count" case.
- **Lab-model checks (exist):** Mindlin vs Salvus 3-D, and the earlier OED lab comparison (Salvus / Mindlin / real-LDV
  replay, `oed_ldv_results/oed_ldv_summary.png`, earlier cube scan). The accumulated variant there (150 points) was the
  best one, which supports rule (b). Candidate for the paper's numerical-verification section or supplement.

## Where things are

- Processing record: `reports/2026-11_specimen/ldv_processing_log.md`
- Reduced-scan run table and schedule: `ldv_pmma_results/reduced_scan_runs.md`
- Outline with figure drafts: `reports/2026-11_mssp_outline/main.tex`

## Section key points (DRAFT 25 Nov, for user review; then copied into the section notes of the paper)

Thread: an illumination-based design places sensors well in simulation (S3); on real LDV data with model error (S4-6) it
beats static and random placement when sensors are few and movable, matches a best-case advance plan without needing one,
and its advantage shrinks as scan and inversion budgets are equalised (S7); dense uniform scanning is the ceiling when
scanning is cheap (S8).
- S1 Intro: dense LDV scans take hours; where to measure matters; classical design is done beforehand (model + design per
  specimen/geometry); adaptive differentiable design designs on the model being inverted (shown in simulation); open: real
  data, model error, which constraints; contributions: method on plate FWI with lab LDV, blind pipeline + scoring, three
  comparisons relaxed -> strict with controls.
- S2 Method: nested loop; illumination needs no true model, no extra data; continuous positions, gradients through the
  solver; checkpointing fits one GPU. -> which objective for the lab?
- S3 Numerical: illumination 16.7 % vs oracle 18.6 % -> taken to the lab; example paths; most favourable setting (no model
  error, movable array = case (i) analogue). -> does it survive real data and stricter budgets?
- S4 Experiment: five features of graded difficulty (D4 hardest; through-hole and glued cube not representable); one dense
  scan replayed for every strategy; 4/2/1 shots; defect positions for scoring only.
- S5 Model: Mindlin + blind calibration fits the intact plate; model error at through-hole and cube (Salvus); same
  background for all strategies.
- S6 Full scan + scoring: full scans find all five (information is in the data); plugs -> detect-locate-contain; two rules
  (fixed amplitude, pre-registered; fixed false-indication rate with background from a defect-free reference); D4 separates.
- S7.1 Strategies: three choices, matrix, cases relaxed -> strict.
- S7.2 Case (i): ~+1 feature over static 30; random gains little (coherent sweeps); planned beforehand (best case) matches;
  ~145 points competitive with uniform (D4 better at 4 shots, noisier background).
- S7.3 Case (ii): re-inverted adaptive +0.3-0.6 at fixed rate; designed beforehand no gain (except 4 shots).
- S7.4 Case (iii): direction holds at 4 and 2 shots, not significant; only adaptive finds all five (3/14 vs 0/14); staged
  inversion costs part of the gain.
- S7.5 Paths: coherent sweeps, not drawn to the cube; D4 follows illumination, not proximity.
- S8 Discussion: gain shrinks with constraints; where adaptivity pays (few movable sensors, costly measurements, irregular
  geometry); adaptive vs beforehand; staged inversion and re-inversion; dense uniform ceiling; limitations.
- S9 Conclusion.

## Framing agreed 25 Nov (after the design controls of 24 Nov; log item 118)

Keep the three cases from relaxed to strict; the controls add rows, they do not replace the story.
- Wording: "the adaptive design works, and matches a best-case design planned beforehand without needing one"
  (not: "adapting to the data adds detection").
- Why adaptive rather than beforehand (user): a design planned beforehand needs a calibrated model of each specimen and
  geometry and a design run per setup; the adaptive design uses the model being inverted anyway. Our offline control used a
  background calibrated on the full defect scan, i.e. a best case nobody has before inspecting.
- Case (i): offline moves match gradient moves (2/10/1); both beat static 30 and random moves. Designed moves sweep coherently
  (net 23-35 mm vs random 12-15 mm at equal path length): a property of the design; disclose that sweep distance and
  direction are not separated (no fixed-heading test, user decision 25 Nov).
- Case (ii): a 30-point design made beforehand gives no gain (except 4 shots); adaptive re-inverted does somewhat better.
- Case (iv), ~145 points, one-pass inversion on both sides: positions visited by the gradient walk are competitive with a
  uniform layout of the same size (detection tie, D4 better at 4 shots: 3/3 vs 1/3 all five; background noisier: FI 66/137 vs
  32/54). Uniform dense scanning is the honest ceiling when scanning is cheap.
- Discussion (user): irregular geometries where a regular grid does not fit (edges, stiffeners, unreachable areas) are where
  an illumination-based adaptive design may have an edge; untested here (rectangular scan area).
- Log criterion moved to the appendix (no gain anywhere).
- Deliverables: draft for the supervisor meeting Fri 27 Nov; poster in about 2 weeks (~9 Dec).

## Lab result at small budgets: three cases, from relaxed to strict (registered 23 Nov)

Order of presentation: (i) -> (ii) -> (iii); each step adds a constraint and loses some strength. Baseline in every case:
static 30-point layouts (regular, jittered, random). Rules: features found (of 5) at a fixed false-indication rate of 1 %
(threshold = 99th percentile of the background anomaly; in practice set from a scan of a defect-free plate; exploratory,
chosen after the pre-registered tests) and v3 (fixed 3 mm, pre-registered). Mean found per shot setting (4 / 2 / 1 shots).
Log items 108-110; scripts src/ldv_pmma/explore_{moving_vs_static,ri_vs_static,adaptive_vs_prog}.py.

### (i) Equal inversion cost: moving array (30 sensors, 5 rounds, gradient moves) vs static 30 points

Scan cost 5x (~145 positions measured), 100 inversion updates on both sides. Control: random moves (same scan, same inversion).

| | 1 % rate | v3 | all five (1 % / v3) | test |
|---|---|---|---|---|
| static 30 (22 runs) | 3.88 / 3.43 / 3.43 | 3.88 / 3.43 / 3.43 | 2/22 / 2/22 | |
| gradient moves (13) | **5.00 / 4.40 / 4.20** | **5.00 / 4.20** / 3.40 | **7/13 / 4/13** | 1 %: MWU p = 0.001, paired 11/2/0; v3: p = 0.021, paired 8/4/1 |
| random moves (13) | 4.33 / 3.80 / 3.40 | 4.00 / 3.60 / 3.60 | 1/13 / 0/13 | 1 %: p = 0.18; v3: p = 0.22 |

Gain comes from the move direction, not the extra measurements. Q2 (pre-registered): gradient beats random in 10/13 pairs.
Reference, 4 shots, dense static: regular r24 (110 pts) 5/5 FI 26, regular r18 (176) 5/5 FI 52, random x24a (110) 4/5 FI 133;
gradient moves 5/5 FI 64-240. Matched static (~145 pts: mj*, rg*_143) and joint inversion of all visited positions (ev_*) in run A.

### (ii) Equal scan cost (30 points), re-inversion allowed: adaptive points (3 groups of 10, illumination-variance criterion) inverted in one pass vs static 30 points

| | 1 % rate | v3 | all five (1 %) | test |
|---|---|---|---|---|
| static 30 (22 runs) | 3.88 / 3.43 / 3.43 | 3.88 / 3.43 / 3.43 | 2/22 | |
| adaptive, re-inverted (14) | **4.20 / 4.00 / 4.00** | 3.40 / 3.80 / 3.75 | 4/14 | 1 %: MWU p = 0.022, paired with same start 5/4/1 (p 0.11); v3: n.s. |

Log criterion: no gain (p >= 0.62).

### (iii) Equal scan and inversion cost: adaptive as run vs the same protocol without design (progressive static, 33/33/34)

| | 1 % rate | v3 | all five (1 % / v3) | test |
|---|---|---|---|---|
| progressive static (14) | 3.80 / 3.20 / 3.00 | 3.60 / 3.00 / 2.75 | 0/14 / 0/14 | |
| adaptive as run (14) | **4.20 / 3.60** / 3.00 | **4.00 / 3.60** / 2.75 | **3/14 / 3/14** | n.s. (MWU p 0.13-0.21, paired 5/6/3 to 6/5/3); pre-registered (found, then FI): 7/14 |

Direction holds at 4 and 2 shots; not significant; adaptive has more false indications under v3 (median 292 vs 232 at 4 shots).

## Decision needed (23 Nov): Q1 result at 30 points is negative

All 30-point runs are done (log item 103). Neither criterion beats a regular or jittered static layout on the same budget:
adaptive vs progressive static 7/14 pairs, log vs old criterion 6/12, re-inverted adaptive points vs the same-start static
layout 5/18; no Fisher test significant. The progressive inversion is a handicap of similar size to the design's gain.
Q2 (gradient vs random moves, 10/13 pairs) stays the positive lab result. Options: report Q1 as a negative/limit result
(30 points is below what the design needs: D4 marginal everywhere), or drop Q1 from the main story and keep F8c as supplementary.

## Open decisions

- Counting rule: confirm (b) after the OED replay test.
- Rerun the synthetic study with rule (b)? (Recommended, 1–2 h GPU.)
- Waveform-misfit objective in the lab: needs data at unmeasured points, so it is not usable blind in a real scan
  (only through the model-based extrapolation of `src/oed_ldv/problem.py`). Include or not?
- Live closed loop: do it, or state it as a limitation.

- **Open (17 Nov): design step size.** With Adam and one step per round (paper protocol), every point moves ~lr per
  axis regardless of gradient strength (o48f: all points moved exactly 11.3 mm in round 1). Test plain gradient descent
  or several Adam steps per round before the final LDV runs; see processing log item 24.

## NeurIPS workshop reviews (Kqqf 5/conf 3, pPMi 4/conf 3, Xw48 5/conf 4) -> to address in the MSSP paper
Files: paper/neurips_review/Untitled*.pdf (screenshots).
- Xw48 ticked "Yes, I flag this paper for an ethics review" while writing "no unaddressed ethical considerations": likely a
  misclick, but check with the chairs / answer it if a camera-ready or discussion is possible.
- Gain may come from leaving a degenerate start (all receivers on a line), not from the design (Xw48); no random /
  predetermined-movement control (pPMi). LDV 17 Nov supports the concern: at 31 points random x48a beats regular r48.
  -> start from regular and random layouts; add a random-move control with the same acquisition budget.
- Acquisition budget not matched: moving receivers re-measure every round (pPMi); compute not matched, fixed layout gets
  ~half the runtime (Kqqf). -> report distinct measured points and wall time; compute-matched baseline.
- How d_obs(p) is differentiated w.r.t. receiver positions (pPMi): synthetic generator vs physical data. Illumination
  needs no observed data (answer: works with LDV); waveform misfit needs spatial derivatives of measured data.
- Inverse crime: no noise, same solver for data and inversion (Kqqf, pPMi). -> LDV data answers it; add noise and a
  different generator (Salvus) in the synthetic study.
- One true model only (all three). -> several targets / defect layouts in the synthetic study.
- Unexplained high variance of illumination (Kqqf); beta not discussed; diagonal of J^T J ignores off-diagonal structure;
  no comparison with SOTA greedy (spectral / D-, A-optimal) sensor selection (Xw48). -> beta sweep, variance analysis,
  greedy baseline.
- Clarity: move Eq. (6) (illumination on the inherited model m_0^t, constant in p, hence low memory) to the main text.
- No scaling beyond a small 2-D case (Kqqf) -> the LDV plate (620 x 520 grid, real data) is the scaling answer.

- **Open (17 Nov): regular baselines are edge-heavy.** make_subsets.py uses offset 0, so r48 has 14 of 32 points within
  10 mm of the scan edge (random 2-4). Rerun regular baselines with a centred grid (offset s/2) before claiming
  "random beats regular"; OED runs starting from the regular groups inherit the same bias.

- **18 Nov evidence:** simulation-only illumination design d30a (30 points, measured once) passes the fixed rule: 4 holes, 38 false cells against baseline medians of 3 and 78. One run; x48a is as good on holes and better on false cells, and the illumination score failed its frozen prediction. Adaptive test a30 is running; needs repeats before it enters the claims.
- **19 Nov results:** simulation-only illumination design improved its starting layout in 3 of 3 pairs with 4 scans (d30a, d30b, d30c), but only
  2 of 3 pass the fixed rule, so the repeatability criterion is not met. Adaptive design (a30, s2a30) did not beat the non-adaptive design with
  4 scans and was worse than its start with 2 sources. Two sources with the full scan still recover all holes (s2full).

### 19 Nov: adaptive vs random moves at a fixed sensor count (48 probes, 3 pairs)
- Fixed rule (lower RMS and no worse holes/false cells, in all 3 pairs): **not met**, 1 of 3 pairs pass (j30a).
- In all 3 pairs the adaptive run finds the fourth hole (D4) and random moves do not. ~~The adaptive run is closer to the full scan (RMS vs v3)~~: withdrawn, the RMS vs v3 difference came from the M1 plug (see below). n = 3, so p = 0.125 at best.
- Possible paper framing (to decide with the user): "with the same number of sensors and the same move lengths, gradient-directed moves consistently recovered the hardest defect", reported with the failed pre-set rule.

### 19 Nov: the glued cube M1 and the scoring (v3)
- The thickness-only Mindlin model cannot represent a glued mass (h moves mass, rotary inertia and stiffness together). The inversion
  renders M1 as a stiff, heavy plug (v3: 59 mm; reduced runs 14-29 mm) with a thin ring, the same signature as the through-hole D3.
  The ring spilled into the old "false cells", and the plug dominated RMS vs v3. The design is not attracted to M1 (3-14 % of measured
  points within 30 mm, 5 % for the whole scan).
- Paper philosophy (user): a defect the model cannot represent must at least be **located** and **contained**, so it does not contaminate
  the rest of the map. Scoring v3 (`score_inversions_v3.py`, processing log item 59): anomaly a = |h - 10|; found if a > 3 mm within 8 mm,
  for all five features (D1-D4, M1); localisation error; false indications a > 1 mm farther than 20 mm (about half an A0 wavelength) from
  every feature; background RMS vs 10 mm there; spill radius of M1 and D3. Rank: found, then false indications, then background RMS.
  RMS vs v3 is dropped (its reference carries the plug).
- Results with v3, adaptive / random / static start (found of 5, false indications, background RMS mm):
  rc30 5/240/0.535, 4/149/0.475, 4/233/0.550; j30a 5/64/0.344, 4/188/0.442, 4/206/0.454; x48c 5/146/0.577, 4/179/0.534, 4/161/0.759.
  Adaptive ranks first in 3 of 3 pairs (only run to find D4), fewer false indications than random in 2 of 3, lower background RMS than
  random in 1 of 3 and than its static start in 3 of 3. ad_j30a is the best 30-point run of the study (5 found, 64 false indications;
  best simulated design d30a 102). Full scans: 3-34 false indications, M1 spill 10-16 mm; sparse runs spill 19-45 mm.
- Candidate claim: "at a fixed sensor count, gradient-directed moves located every feature, including the hardest hole and an
  unmodelled glued mass, while random moves of the same lengths never found the hardest hole; containment of the unmodelled
  feature remains the open weakness of sparse layouts". Running (19-20 Nov): the same pairs with 1 shot and 2 shots (item 55).
- Decision (user, 19 Nov): no fixed-heading control; keep the storyline closed. Disclose in the paper that the random-move control has the
  same path length but a smaller net displacement (13-14 vs 24-29 mm), so part of the D4 gain may come from sweeping farther (processing log item 60).

### 20 Nov: fewer shots (1 shot = file 3; 2 shots = files 1 and 3), 5 pairs each, v3 scoring (processing log item 62)
- Adaptive ranks above random moves in 3 of 5 pairs (1 shot) and 4 of 5 (2 shots); with the 4-shot pairs 10 of 13. The 10 new pairs alone:
  7 of 10, p = 0.17. Strict rule (better on everything): 2 of 5 for each setting.
- The D4 advantage seen with 4 shots mostly disappears with fewer shots (adaptive 2 of 10, random 0 of 10).
- Both moving strategies beat the static start most of the time (adaptive 8 of 10, random 6 of 10): they measure 4-5 times more points.
- Honest reading: a consistent tendency in favour of gradient-directed moves at a fixed sensor count, not a clean win; failure cases exist
  (a2_j30a, a1_rc30). The full 1-shot scan finds all 5 features with 3 false indications, so the information is in the data.

## Paper storyline (draft bullets, 20 Nov)
1. **Problem.** Full-field LDV scans of a plate take hours; we want fewer measurement points, placed where they help the inversion most.
   Question: does differentiable adaptive design place a fixed number of sensors better than random or regular placement on real data?
2. **Method.** Mindlin plate full-waveform inversion of thickness (JAX, 1 mm grid, 620 x 520), illumination criterion from Rademacher
   probes and per-source sensitivities, sensor positions moved by its gradient between inversion rounds (the NeurIPS workflow, now on
   real data with probe grouping and checkpointing to fit an 11 GB GPU).
3. **Specimen and data.** 10 mm PMMA plate, 3-D LDV, four blind holes (one through), one glued steel cube; scans with 1, 2 and 4 shots.
   Every result-producing step is blind to the defects; they are used only for scoring.
4. **Model error as part of the story.** A thickness-only plate model cannot represent a glued mass: it renders it as a stiff, heavy
   plug with a thin ring (and the through-hole as a plug too). We therefore score detection, localisation and containment of every
   feature through the anomaly |h - 10 mm|, not agreement with a reference inversion (scoring v3).
5. **Main experiment.** 30-31 sensors, 5 rounds, every sensor moves each round; adaptive (gradient direction) vs random moves with the
   same step lengths, from regular, jittered and random starting layouts; 13 pairs over 4, 2 and 1 shots (fig_pairs_summary).
6. **Result.** Adaptive ranks above random in 10 of 13 pairs (4 shots 3/3, 2 shots 4/5, 1 shot 3/5); with 4 shots it is the only
   strategy that finds the shallowest small hole (D4), and the best 30-point map of the study comes from it (ad_j30a: 5 of 5 found,
   64 false indications). Better on every measure at once in only 5 of 13 pairs; clear failures exist (a2_j30a, a1_rc30).
7. **Both moving strategies beat the static layout** most of the time (more distinct points measured), so the fair comparison is
   at equal sensor count and equal move lengths, as done here.
8. **Limitations to state openly.** n = 13 pairs (7 of 10 on the pairs scored after the rule was fixed, p = 0.17); random moves end
   closer to their start than adaptive ones (13-14 vs 24-29 mm net), so part of the gain may come from sweeping farther; the illumination
   score did not rank simple static layouts reliably; unmodelled features spill (sparse runs 19-45 mm vs 10-16 mm for full scans).
9. **Take-home.** On real LDV data, gradient-directed sensor moves are a consistent, not guaranteed, improvement over random moves at a
   fixed sensor count; the full 1-shot scan still finds every feature, so better design criteria (and models that contain unmodelled
   features) are where the remaining gap lies.
- 20 Nov, forward-model check (F3, processing log item 65): Salvus 3-D of the as-built plate fits the defect scan with misfit 0.043, as well as the
  blind Mindlin inversion (0.042), while Mindlin with the as-built defects reaches only 0.076 (pristine plate 0.082). Mindlin is adequate for the
  intact plate (0.028 vs Salvus 0.037). Message for the paper: the plate model's defect representation (through-hole, glued mass) is the main model
  error, and the inversion compensates for it with plugs; this supports the detect-locate-contain scoring.

### 20 Nov: paper structure (user)
- Part I, numerical study (kept from the workshop, with its CPU numbers): test the method and compare the design objectives (oracle 18.6 %,
  illumination 16.7 %, waveform misfit 9.4 % over the fixed layout, 10/10 seeds), plus cost: CPU workshop vs CPU + remat vs GPU + remat
  (F10, one design gradient: remat cuts memory 4-15x at +20 % to 2.7x CPU time, identical gradients; GPU runs the full study in 22 min vs 4.2 h).
  Remat is the enabler (fits 11 GB), the GPU is the accelerator; conclusions of the workshop unchanged.
- Part II, laboratory: Mindlin plate and its check (F3), blind processing, scoring, the 13 LDV pairs (F7-F9).
