# LDV inversions: overview for the storyline (v3 scores)

Budget = distinct measured LDV points (about 3.8 s each; full scan 6418 points, about 7 h). found = features of 5 (D1-D4, M1) with |h - 10 mm| > 3 mm within 8 mm; false ind. = cells (3 mm) with |h - 10| > 1 mm farther than 20 mm from every feature; bg RMS = RMS of h - 10 mm there.

## 4 shots

| run | layout | points | found | missing | false ind. | bg RMS |
|---|---|---|---|---|---|---|
| r12 | full / dense static | 406 | 5 | - | 29 | 0.180 |
| r9 | full / dense static | 708 | 5 | - | 28 | 0.191 |
| x9b | full / dense static | 706 | 5 | - | 31 | 0.190 |
| v3 | full / dense static | all | 5 | - | 29 | 0.186 |
| x36a | static, other budgets | 47 | 3 | D4,M1 | 318 | 0.396 |
| x36b | static, other budgets | 47 | 4 | D4 | 37 | 0.332 |
| r36 | static, other budgets | 48 | 5 | - | 145 | 0.414 |
| rc48 | static, other budgets | 48 | 4 | D4 | 72 | 0.346 |
| r24 | static, other budgets | 110 | 5 | - | 26 | 0.198 |
| x24a | static, other budgets | 110 | 4 | D4 | 133 | 0.551 |
| r18 | static, other budgets | 176 | 5 | - | 52 | 0.242 |
| rc30 | static, 30 points | 30 | 4 | D4 | 233 | 0.550 |
| r48 | static, 30 points | 31 | 2 | D1,D2,D4 | 131 | 0.536 |
| j30a | static, 30 points | 30 | 4 | D4 | 206 | 0.454 |
| j30b | static, 30 points | 30 | 4 | D4 | 120 | 0.672 |
| x48a | static, 30 points | 31 | 5 | - | 97 | 0.608 |
| x48b | static, 30 points | 31 | 4 | D4 | 520 | 0.799 |
| x48c | static, 30 points | 31 | 4 | D4 | 161 | 0.759 |
| x48d | static, 30 points | 30 | 4 | D4 | 155 | 0.574 |
| d30a | simulation design | 30 | 5 | - | 102 | 0.327 |
| d30b | simulation design | 30 | 4 | D4 | 206 | 0.564 |
| d30c | simulation design | 29 | 5 | - | 109 | 0.518 |
| a30 | adaptive, 3 groups of 10 | 30 | 5 | - | 196 | 0.473 |
| o48b | adaptive, points added per round (early settings) | 31 | 2 | D1,D2,D4 | 232 | 0.526 |
| s31t | greedy selection (targeted) | 31 | 2 | D1,D2,D4 | 509 | 0.833 |
| s31c | greedy selection (coverage) | 31 | 2 | D1,D2,D4 | 318 | 0.647 |
| q4a_j30a | adaptive, 3 groups of 10 | 30 | 4 | D4 | 567 | 0.624 |
| q4a_j30b | adaptive, 3 groups of 10 | 30 | 4 | D4 | 331 | 0.562 |
| q4a_j30c | adaptive, 3 groups of 10 | 30 | 5 | - | 292 | 0.597 |
| q4a_j30d | adaptive, 3 groups of 10 | 30 | 2 | D1,D2,D4 | 261 | 0.570 |
| q4l_j30a | adaptive (log criterion), 3 groups of 10 | 30 | 4 | D4 | 125 | 0.429 |
| q4l_j30b | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D2,D4 | 123 | 0.518 |
| q4l_j30c | adaptive (log criterion), 3 groups of 10 | 30 | 5 | - | 449 | 0.851 |
| q4l_j30d | adaptive (log criterion), 3 groups of 10 | 30 | 2 | D1,D2,D4 | 401 | 0.603 |
| n30 | static, progressive (3 groups, no design) | 30 | 4 | D4 | 232 | 0.541 |
| q4n_j30a | static, progressive (3 groups, no design) | 30 | 4 | D4 | 167 | 0.427 |
| q4n_j30b | static, progressive (3 groups, no design) | 30 | 4 | D4 | 208 | 0.692 |
| q4n_j30c | static, progressive (3 groups, no design) | 30 | 3 | D1,M1 | 403 | 0.805 |
| q4n_j30d | static, progressive (3 groups, no design) | 30 | 3 | D1,D4 | 470 | 0.624 |
| ad_rc30 | moving, gradient (start rc30) | 148 | 5 | - | 240 | 0.535 |
| rn_rc30 | moving, random (start rc30) | 147 | 4 | D4 | 149 | 0.475 |
| ad_j30a | moving, gradient (start j30a) | 141 | 5 | - | 64 | 0.344 |
| rn_j30a | moving, random (start j30a) | 138 | 4 | D4 | 188 | 0.442 |
| ad_x48c | moving, gradient (start x48c) | 149 | 5 | - | 146 | 0.577 |
| rn_x48c | moving, random (start x48c) | 142 | 4 | D4 | 179 | 0.534 |
| o48f | moving, gradient (start r48 (12 probes, early)) | 148 | 4 | D1 | 192 | 0.436 |
| o48r | moving, random (start r48 (12 probes, early)) | 146 | 4 | D4 | 209 | 0.556 |

## 2 shots

| run | layout | points | found | missing | false ind. | bg RMS |
|---|---|---|---|---|---|---|
| s2full | full / dense static | all | 5 | - | 34 | 0.203 |
| s2rc30 | static, 30 points | 30 | 4 | D4 | 234 | 0.557 |
| s2j30a | static, 30 points | 30 | 4 | D4 | 548 | 0.811 |
| s2j30b | static, 30 points | 30 | 4 | D4 | 301 | 0.780 |
| s2x48a | static, 30 points | 29 | 2 | D1,D2,D4 | 165 | 0.428 |
| s2x48b | static, 30 points | 31 | 4 | D1 | 755 | 1.392 |
| s2x48c | static, 30 points | 31 | 4 | D4 | 677 | 0.922 |
| s2x48d | static, 30 points | 31 | 2 | D1,D2,D4 | 487 | 0.724 |
| s2d30 | simulation design | 30 | 4 | D4 | 311 | 0.562 |
| s2a30 | adaptive, 3 groups of 10 | 30 | 5 | - | 309 | 0.603 |
| q2a_j30a | adaptive, 3 groups of 10 | 30 | 4 | D4 | 746 | 0.970 |
| q2a_j30b | adaptive, 3 groups of 10 | 30 | 3 | D2,D4 | 337 | 0.748 |
| q2a_j30c | adaptive, 3 groups of 10 | 30 | 3 | D2,D4 | 379 | 0.823 |
| q2a_j30d | adaptive, 3 groups of 10 | 30 | 3 | D1,D4 | 402 | 0.669 |
| q2l_j30a | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D1,D4 | 666 | 0.823 |
| q2l_j30b | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D2,D4 | 301 | 0.677 |
| q2l_j30c | adaptive (log criterion), 3 groups of 10 | 30 | 4 | D4 | 351 | 0.631 |
| q2l_j30d | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D1,D4 | 276 | 0.538 |
| s2n30 | static, progressive (3 groups, no design) | 30 | 3 | D2,D4 | 333 | 0.620 |
| q2n_j30a | static, progressive (3 groups, no design) | 30 | 3 | D1,D4 | 707 | 0.807 |
| q2n_j30b | static, progressive (3 groups, no design) | 30 | 4 | D4 | 370 | 0.817 |
| q2n_j30c | static, progressive (3 groups, no design) | 30 | 3 | D1,D4 | 214 | 0.564 |
| q2n_j30d | static, progressive (3 groups, no design) | 30 | 2 | D1,D2,D4 | 339 | 0.757 |
| a2_rc30 | moving, gradient (start s2rc30) | 145 | 4 | D4 | 182 | 0.442 |
| r2_rc30 | moving, random (start s2rc30) | 144 | 4 | D4 | 245 | 0.497 |
| a2_j30a | moving, gradient (start s2j30a) | 148 | 4 | D4 | 801 | 1.019 |
| r2_j30a | moving, random (start s2j30a) | 141 | 4 | D4 | 440 | 0.602 |
| a2_x48c | moving, gradient (start s2x48c) | 153 | 5 | - | 568 | 0.775 |
| r2_x48c | moving, random (start s2x48c) | 144 | 4 | D4 | 449 | 0.677 |
| a2_j30b | moving, gradient (start s2j30b) | 146 | 4 | D4 | 217 | 0.529 |
| r2_j30b | moving, random (start s2j30b) | 144 | 3 | D1,D4 | 839 | 0.881 |
| a2_x48a | moving, gradient (start s2x48a) | 150 | 4 | D4 | 81 | 0.364 |
| r2_x48a | moving, random (start s2x48a) | 146 | 3 | D1,D4 | 95 | 0.361 |

## 1 shot

| run | layout | points | found | missing | false ind. | bg RMS |
|---|---|---|---|---|---|---|
| k1full | full / dense static | all | 5 | - | 3 | 0.145 |
| k1rc30 | static, other budgets | 30 | 4 | D4 | 267 | 0.667 |
| k1j30a | static, other budgets | 30 | 1 | D2,D3,D4,M1 | 401 | 0.672 |
| k1j30b | static, other budgets | 30 | 2 | D2,D3,D4 | 123 | 0.670 |
| k1x48a | static, other budgets | 29 | 2 | D2,D4,M1 | 217 | 0.573 |
| k1x48c | static, other budgets | 31 | 4 | D4 | 239 | 0.773 |
| q1s_rc30 | static, 30 points | 30 | 5 | - | 203 | 0.616 |
| q1s_j30a | static, 30 points | 30 | 3 | D3,D4 | 224 | 0.521 |
| q1s_j30b | static, 30 points | 30 | 4 | D4 | 120 | 0.609 |
| q1s_j30c | static, 30 points | 30 | 3 | D3,D4 | 690 | 1.222 |
| q1s_j30d | static, 30 points | 30 | 3 | D1,M1 | 221 | 0.494 |
| q1s_x30a | static, 30 points | 30 | 3 | D1,D4 | 851 | 1.455 |
| q1s_x30b | static, 30 points | 30 | 3 | D1,D4 | 211 | 0.601 |
| q1a_rc30 | adaptive, 3 groups of 10 | 30 | 3 | D1,D4 | 452 | 0.830 |
| q1a_j30a | adaptive, 3 groups of 10 | 30 | 3 | D2,D3 | 214 | 0.547 |
| q1a_j30b | adaptive, 3 groups of 10 | 30 | 3 | D3,D4 | 209 | 0.767 |
| q1a_j30c | adaptive, 3 groups of 10 | 30 | 2 | D1,D3,D4 | 354 | 0.717 |
| q1l_rc30 | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D2,D4 | 434 | 0.886 |
| q1l_j30a | adaptive (log criterion), 3 groups of 10 | 30 | 1 | D2,D3,D4,M1 | 360 | 0.559 |
| q1l_j30b | adaptive (log criterion), 3 groups of 10 | 30 | 3 | D3,D4 | 293 | 0.783 |
| q1l_j30c | adaptive (log criterion), 3 groups of 10 | 30 | 4 | D3 | 351 | 0.771 |
| q1n_rc30 | static, progressive (3 groups, no design) | 30 | 3 | D3,D4 | 328 | 0.759 |
| q1n_j30a | static, progressive (3 groups, no design) | 30 | 2 | D1,D3,M1 | 277 | 0.569 |
| q1n_j30b | static, progressive (3 groups, no design) | 30 | 4 | D4 | 203 | 0.684 |
| q1n_j30c | static, progressive (3 groups, no design) | 30 | 2 | D1,D3,D4 | 896 | 1.041 |
| a1_rc30 | moving, gradient (start k1rc30) | 132 | 3 | D2,D4 | 337 | 0.594 |
| r1_rc30 | moving, random (start k1rc30) | 127 | 4 | D4 | 59 | 0.346 |
| a1_j30a | moving, gradient (start k1j30a) | 116 | 3 | D2,D4 | 229 | 0.468 |
| r1_j30a | moving, random (start k1j30a) | 113 | 2 | D2,D3,D4 | 182 | 0.480 |
| a1_x48c | moving, gradient (start k1x48c) | 150 | 4 | M1 | 149 | 0.454 |
| r1_x48c | moving, random (start k1x48c) | 144 | 4 | D4 | 339 | 0.775 |
| a1_j30b | moving, gradient (start k1j30b) | 133 | 4 | D4 | 88 | 0.371 |
| r1_j30b | moving, random (start k1j30b) | 125 | 4 | D4 | 147 | 0.484 |
| a1_x48a | moving, gradient (start k1x48a) | 131 | 3 | D4,M1 | 231 | 0.491 |
| r1_x48a | moving, random (start k1x48a) | 127 | 4 | D4 | 1638 | 1.555 |

## Summary

- Q1, 4 shot(s), 30 points: all 5 found in 4/8 illumination designs (simulation or adaptive) vs 1/8 static layouts (one-sided Fisher p = 0.141); false indications designs [102, 206, 109, 196, 567, 331, 292, 261] vs static median 158
- Q2, 4 shot(s): gradient ranks above random in 3/3 pairs (sign test p = 0.125); more features 3, fewer 0; fewer false indications 2/3; lower bg RMS 1/3
- Q1, 2 shot(s), 30 points: all 5 found in 1/6 illumination designs (simulation or adaptive) vs 0/7 static layouts (one-sided Fisher p = 0.462); false indications designs [311, 309, 746, 337, 379, 402] vs static median 487
- Q2, 2 shot(s): gradient ranks above random in 4/5 pairs (sign test p = 0.188); more features 3, fewer 0; fewer false indications 3/5; lower bg RMS 2/5
- Q1, 1 shot(s), 30 points: all 5 found in 0/4 illumination designs (simulation or adaptive) vs 1/7 static layouts (one-sided Fisher p = 1.000); false indications designs [452, 214, 209, 354] vs static median 221
- Q2, 1 shot(s): gradient ranks above random in 3/5 pairs (sign test p = 0.500); more features 1, fewer 2; fewer false indications 3/5; lower bg RMS 4/5
