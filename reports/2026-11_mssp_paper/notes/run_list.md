# LDV runs: what was run, why, and where it stands (21 Nov 2026, 10:10)

Scores: v3 (found = |h − 10| > 3 mm within 8 mm; FI = false indications, background cells > 1 mm).
Shots: 4 = files 1–4 (sources S1–S4); 2 = files 1, 3; 1 = file 3 only (S1 + S3 firing together).
Full detail per run: `run_overview.md`; every step: `reports/2026-11_specimen/ldv_processing_log.md`.

## A. Reference: full scans (6418 points, ~7 h scanning)

| tag | shots | result |
|---|---|---|
| v3 | 4 | 5/5, FI 29 |
| s2full | 2 | 5/5, FI 34 |
| k1full | 1 | 5/5, FI 3 |

## B. Q2 — gradient moves vs random moves (same sensors, same move lengths) → F8, F9

Move-all protocol: 5 rounds, every sensor moves one 8 mm Adam step per round; the random partner moves the same lengths in random directions.
Tags: `ad_` / `rn_` (4 shots), `a2_` / `r2_` (2 shots), `a1_` / `r1_` (1 shot), then the starting layout.
**Done. Gradient ranks above random in 10 of 13 pairs** (4 shots 3/3, 2 shots 4/5, 1 shot 3/5).

| start | 4 shots: gradient / random | 2 shots: gradient / random | 1 shot: gradient / random |
|---|---|---|---|
| rc30 | 5/5 FI 240 / 4/5 FI 149 | 4/5 182 / 4/5 245 | 3/5 337 / 4/5 59 |
| j30a | 5/5 FI 64 / 4/5 FI 188 | 4/5 801 / 4/5 440 | 3/5 229 / 2/5 182 |
| x48c | 5/5 FI 146 / 4/5 FI 179 | 5/5 568 / 4/5 449 | 4/5 149 / 4/5 339 |
| j30b | – | 4/5 217 / 3/5 839 | 4/5 88 / 4/5 147 |
| x48a | – | 4/5 81 / 3/5 95 | 3/5 231 / 4/5 1638 |

## C. Q1 — same scan budget: 30 measured points (~2 min scanning) → F8b, F8c

### C1. Static layouts (the baseline) — done

| shots | tags | all 5 found |
|---|---|---|
| 4 | rc30, r48, j30a, j30b, x48a–d | 1/8 |
| 2 | s2rc30, s2j30a, s2j30b, s2x48a–d | 0/7 |
| 1 | q1s_rc30, q1s_j30a–d, q1s_x30a, q1s_x30b | 1/7 |

(rc = regular, j = jittered regular, x = random.)

### C2. Simulation design (designed once on the defect-free model, then measured) — done

| tag | shots | result |
|---|---|---|
| d30a / d30b / d30c | 4 | 5/5 FI 102 / 4/5 FI 206 / 5/5 FI 109 |
| s2d30 | 2 | 4/5, FI 311 |

### C3. Adaptive design, criterion var(I/Ī) − 0.05 log Ī ("a30 protocol") — done except q4a_j30c/d

Protocol: 3 interleaved groups of 10 points; group 1 measured as is; each later group designed on the current inverted model
(20 Adam steps); inversion 33 + 33 + 34 updates.

| shots | tag (start) | result | status |
|---|---|---|---|
| 4 | a30 (rc30) | 5/5, FI 196 | done (earlier) |
| 4 | q4a_j30a, q4a_j30b | 4/5 (D4), FI 567 / 331 | done |
| 4 | q4a_j30c, q4a_j30d | – | **running** (until ~14:30) |
| 2 | s2a30 (s2rc30) | 5/5, FI 309 | done (earlier) |
| 2 | q2a_j30a / b / c / d | 4/5 746 / 3/5 337 / 3/5 379 / 3/5 402 | done |
| 1 | q1a_rc30 / j30a / j30b / j30c | 3/5 452 / 3/5 214 / 3/5 209 / 2/5 354 | done |

Diagnosis (log 89): D4 is the only feature that separates methods; adaptive puts it in the right place but too weak; round-2
designs chase false thin spots of the 20-point model.

### C4. Re-inversion of the adaptive points — done (20:00 Sat)

Same final 30 points as each C3 run, but inverted like a static layout (100 updates from the background).
Question: is the progressive 33/33/34 inversion the handicap, rather than where the points are?

| tags | shots |
|---|---|
| ri_s2a30, ri_q2a_j30a–d | 2 |
| ri_a30, ri_q4a_j30a–d | 4 |
| ri_q1a_rc30, ri_q1a_j30a–c | 1 |

### C4b. Progressive static control — done (01:14 Sun); adaptive wins 7/14 pairs

Each adaptive start measured as its 3 groups of 10 **without design**, with exactly the adaptive inversion (33 + 33 + 34 updates).
Paired with C3 at the same start: the only difference is the design.

| tags | shots | pairs with |
|---|---|---|
| s2n30, q2n_j30a–d | 2 | s2a30, q2a_j30a–d |
| n30, q4n_j30a–d | 4 | a30, q4a_j30a–d |
| q1n_rc30, q1n_j30a–c | 1 | q1a_rc30, q1a_j30a–c |

### C5. Adaptive design, new criterion −mean(log I) — done (05:14 Mon); no gain over the old criterion (6/12)

Same protocol and same starts as C3, only the criterion changes (more weight on overall illumination, hot spots saturate).
Each run is followed by its re-inversion.

| tags | shots |
|---|---|
| q2l_j30a–d, then ri_q2l_j30a–d | 2 |
| q4l_j30a–d, then ri_q4l_j30a–d | 4 |
| q1l_rc30, q1l_j30a–c, then ri_q1l_* | 1 (added 22 Nov, queue_logmean_1shot.sh, ~22:30 Sun → ~06:00 Mon) |

## D. Older runs (kept for completeness, not in the main story)

- Static at other budgets (4 shots): x36a/b, r36, rc48, r24, x24a, r18; dense static r12, r9, x9b.
- Earlier 1-shot statics with 24–27 points: k1rc30, k1j30a, k1j30b, k1x48a, k1x48c (replaced by q1s_*).
- Greedy selection s31t / s31c and early adaptive settings o48b, o48f / o48r ("earlier designs" in F8b).

## Timeline

| when | what |
|---|---|
| Sat ~14:30 | q4a_j30c/d done → C3 complete |
| Sat ~20:00 | re-inversions (C4) done |
| Sun ~01:00 | progressive static controls (C4b) done |
| Sun ~02:30 | first new-criterion run (q2l_j30a) done → check that the criterion behaves |
| Sun ~22:30 | 2- and 4-shot new-criterion blocks (C5) done |
| Mon ~06:00 | 1-shot new-criterion block done → final F8b / F8c and tests |

## E. Proposal A — all visited positions of the moving arrays, inverted jointly (queued 23 Nov 13:46 → ~23:15)

Same data as B, but every distinct position each moving array visited (~145) is inverted at once (100 updates, like static),
against a static jittered layout with the same number of points. Question: at ~145 points, are gradient-walked positions better?

| tags | shots |
|---|---|
| ev_ad_{rc30,j30a,x48c}, ev_rn_{…}, mj4_{…} | 4 |
| ev_a2_{rc30,j30a,j30b,x48a,x48c}, ev_r2_{…}, mj2_{…} | 2 |

Regular reference: rg4_143, rg2_143 (13 x 11, 143 points), queued after A (~00:05).
Scored v3 and at a fixed false-indication rate (F8d, exploratory).

## F. Design controls (queued 23 Nov, after E; done ~16:30 Tue)

| block | tags | role |
|---|---|---|
| simulation design | s2d30b, s2d30c (2 shots); q1d_rc30, q1d_j30a, q1d_j30b (1 shot) | design before measuring, one-pass inversion: cases (ii)/(iii) |
| offline moving array | od_{rc30,j30a,x48c}; o2_{rc30,j30a,j30b,x48a,x48c}; o1_{same} | moves computed on the defect-free model (--design-on-background): adaptivity control for case (i) |
| all visited, offline | ev_od_*, ev_o2_*, ev_o1_* | case (iv) |

Possibly left out of the paper if the storyline gets too big (user, 23 Nov).

## Timeline (23-24 Nov)

| when | what |
|---|---|
| Mon ~22:25 | proposal A (ev_*, mj*) done |
| Mon ~23:15 | regular 143-point grids done |
| Tue ~01:15 | simulation designs done |
| Tue ~12:00 | offline moving arrays done |
| Tue ~16:30 | all visited (offline) done -> assess, F10a with blocks (iv) and controls, false-indication panel |
