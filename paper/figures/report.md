# Multi-seed online receiver-design report

Mode: **scientific**. Configuration fingerprint: `ca40761cc1be56b0`.

## Experimental protocol

The study requested 10 paired repetitions. Each optimized run in this mode uses 5 receiver-design updates followed or accompanied, according to the objective ordering, by 20 carried-model FWI updates per round (100 total). The paired fixed-layout baseline uses one uninterrupted 100-update inversion. No extra final inversion is performed.

Across repetitions, the initial diagonal receiver layout is independently jittered, the paired FWI source schedule changes, and illumination uses a new probe realization. Within a repetition, all methods share exactly the same initial receiver layout and FWI source schedule. Illumination probes are fixed across every design evaluation in that repetition.

The true model and inclusion, four active source positions, 30 receivers, solver grid, CPML, wavelet, time sampling, trace-by-trace normalization, inner SGD and outer Adam learning rates, illumination probe count, and update counts are held fixed.

Illumination differentiates its stochastic Gauss--Newton illumination criterion with respect to receiver coordinates at the current reconstruction, updates the geometry, and then performs the round's FWI updates without differentiating through them. Oracle and moving-geometry waveform objectives differentiate through the round's unrolled SGD inversion, update the receiver coordinates, and carry the reconstructed model. Waveform observations and residuals are both evaluated at the current candidate geometry. SPSA is not used.

## Completion

Completed runs: **40/40**. Recorded failed runs: **0**. Runs written by a resumed invocation: **20**.

## Descriptive results

### Fixed receiver layout

Runs: 10. Final model MSE: mean 3.36e+03, sample SD 65.5, median 3.37e+03, IQR [3.32e+03, 3.4e+03]. Improvement versus paired baseline: mean 0%, sample SD not defined, median 0%, IQR [0, 0]%. Improved repetitions: 0/10. Total runtime: mean 166 s, sample SD 0.86 s, median 166 s, IQR [166, 166] s.

### Stochastic illumination

Runs: 10. Final model MSE: mean 2.8e+03, sample SD 275, median 2.87e+03, IQR [2.63e+03, 2.97e+03]. Improvement versus paired baseline: mean 16.7%, sample SD 8.57%, median 14.2%, IQR [11.5, 19.8]%. Improved repetitions: 10/10. Total runtime: mean 357 s, sample SD 0.798 s, median 357 s, IQR [356, 357] s.

### Waveform misfit

Runs: 10. Final model MSE: mean 3.04e+03, sample SD 125, median 3.06e+03, IQR [2.93e+03, 3.12e+03]. Improvement versus paired baseline: mean 9.41%, sample SD 4.34%, median 8.62%, IQR [5.88, 12.7]%. Improved repetitions: 10/10. Total runtime: mean 503 s, sample SD 0.687 s, median 503 s, IQR [503, 504] s.

### Oracle model error

Runs: 10. Final model MSE: mean 2.73e+03, sample SD 242, median 2.68e+03, IQR [2.56e+03, 2.95e+03]. Improvement versus paired baseline: mean 18.6%, sample SD 7.06%, median 18.7%, IQR [11.9, 22.9]%. Improved repetitions: 10/10. Total runtime: mean 479 s, sample SD 1.16 s, median 479 s, IQR [479, 480] s.

## Runtime accounting

`total_fwi_update_count` and `candidate_fwi_update_evaluations` both count the 100 SGD updates in the reconstruction carried forward. For oracle and waveform objectives, nested FWI and receiver-hypergradient time are compiled together and cannot be separated exactly; illumination records its design and subsequent FWI timings separately.

## Limitations

With only 10 repetitions, estimates of variability and especially tails are unstable (the default five-run study is particularly limited). The report therefore emphasizes paired changes and every individual observation. No claim of statistical significance is made, and no inferential test is performed.

## Reproducibility notes

The experiment generates a repetition-specific FWI source schedule and holds it paired across methods. Illumination receives a repetition-specific probe realization held fixed throughout that repetition. The reconstructed model is carried between rounds while plain SGD has no moment state to carry. Scientific and smoke-test outputs are kept in separate directories.
