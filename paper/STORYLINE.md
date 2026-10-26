# Paper storyline and workshop decision

## Recommendation

Target the **Sim2Science NeurIPS 2026 workshop** with a five-page workshop paper.

- Deadline: **29 August 2026, 23:59 AoE**.
- Review: double blind; NeurIPS 2026 LaTeX style; references excluded from the five-page limit.
- Status: non-archival; early research and synthetic scientific-simulator applications are explicitly welcome.
- Important administrative item: nominate an eligible reciprocal reviewer at submission time; that person may receive two workshop papers to review.

The Representations for the Physical Sciences workshop is a weaker fit. Its sampling theme is relevant, but the call is specifically organized around representation learning, self-supervision, transfer, sampling for learned representations, and tokenization. This project currently has no learned representation as a scientific contribution.

Sim2Science's exact theme is **“ML with Imperfect Scientific Models.”** Its listed topics include closed-loop and experiment-in-the-loop workflows and differentiable frameworks, which directly match this project. The current successful experiment is still matched-physics synthetic data, however. If the existing Salvus-to-acoustic pipeline can be rerun reliably within the one-day budget, a compact matched-versus-mismatched-physics result would make the venue fit substantially stronger. A negative result is scientifically useful here: the EEG deck already suggests that optimizing a differentiable acoustic surrogate can improve its outer loss while worsening reconstruction of 3D elastic data.

## Recommended thesis

> Differentiable wave physics turns acquisition design into a programmable, task-aware optimization problem: continuous receiver positions can be optimized for the actual reconstruction or for deployable proxies, instead of selecting measurements from a fixed candidate survey using one prescribed Jacobian criterion.

The paper is **not** a catalogue of cases A/B/C/D. It is a controlled study of three outer objectives in one continuous design framework.

## Story arc

1. **Problem.** FWI is powerful but expensive; acquisition geometry controls reconstruction quality.
2. **Limitation of the usual formulation.** Classical FWI OED selects rows/sources/receivers from a comprehensive survey and uses a local linear-information criterion. This is discrete and may not align with the nonlinear reconstruction actually run.
3. **Mechanism.** A JAX time-domain solver and an unrolled FWI optimizer form one differentiable program. Receiver coordinates become continuous optimization variables.
4. **Scientific question.** Once the workflow is differentiable, which end objective should design the experiment?
5. **Objective ladder.** Compare:
   - true-model error: ideal but privileged oracle;
   - waveform misfit: an online physical-data objective with no true-model error in the outer loss;
   - stochastic illumination: the criterion does not differentiate through FWI; 20 FWI updates still advance the model after each design update.
6. **Matched reconstruction budget.** All optimized layouts use five acquisition/design rounds with 20 FWI updates per round; the fixed-layout baseline uses one uninterrupted run of $5\times20=100$ FWI updates. For illumination, each sensitivity-based layout update is followed by 20 FWI updates, but the illumination criterion is not differentiated through them. Every displayed reconstruction therefore uses 100 FWI updates.
7. **Result.** In the full-autodiff synthetic NDT case, the fixed-layout model MSE of 3449 falls to 2668 (oracle), 3040 (waveform misfit), and 3024 (illumination), or improvements of 22.6%, 11.9%, and 12.3%.
8. **Takeaway.** The deployable objectives recover most of the oracle benefit on this proof of concept; differentiability provides flexibility, while objective validity and robustness remain the key research questions. Applying the loop to elastic LDV measurements remains an open scaling question, motivating future learned-surrogate or simpler plate-physics design models rather than constituting a result of this paper.

## Contribution claims we can defend

1. A time-domain automatic-differentiation implementation that optimizes continuous receiver coordinates through a finite FWI procedure.
2. A single framework comparing privileged, ground-truth-free data, and stochastic illumination objectives.
3. A synthetic NDT demonstration in which data and illumination objectives approach the reconstruction benefit of an oracle objective.
4. An online/acquisition-in-the-loop interpretation for repeatable movable-sensor experiments.

## Claims to avoid

- “The first differentiable/bilevel OED method for FWI.” Downing et al. already optimize FWI sensor locations by supervised bilevel learning.
- “The first continuous differentiable sensor-placement method.” PIED, NODE, DSPO, and related work already do this for other inverse problems or neural reconstructors.
- “Experimental validation.” The latest successful evidence is synthetic and acoustically simplified.
- “Generalizes across models” or “robust to noise” until those experiments exist.
- A quantitative illumination speed advantage before the matched wall-clock and hardware analysis is complete.

## Closest literature and differentiation

| Work | Reconstruction/design mechanism | Key difference from this paper |
|---|---|---|
| Maurer et al. (2017); Krampe et al. (2021) | Jacobian/Gauss--Newton information criteria and subset selection | Classical motivation and baseline family; usually needs a comprehensive candidate survey |
| Mercier et al. (2025) | Wavelet/Jacobian criterion with source selection and compressed model space | Earlier discrete, information-proxy route; this paper optimizes continuous coordinates through reconstruction |
| Downing et al. (2026) | Supervised bilevel sensor optimization for Helmholtz FWI; implicit/adjoint upper gradient | Closest work. This paper uses finite unrolled time-domain FWI and emphasizes waveform-misfit and cheap illumination objectives |
| PIED (Hemachandra et al., 2025) | Continuous design through PINN training dynamics | Neural/PINN inverse solver and one-shot design rather than conventional time-domain FWI |
| NODE (Darges et al., 2025) | Joint neural reconstructor and continuous fixed-budget design | Avoids classical bilevel inversion by learning a reconstructor |
| DSPO (Liu et al., 2025) | Bilevel differentiable placement for neural field reconstruction | Sparse-field neural reconstruction, not wave-equation FWI |
| j-Wave (Stanziola et al., 2023) | Differentiable wave simulation and FWI prototyping | Enabling software; it does not establish the objective comparison here |

## One-day GPU experiment plan

Do not begin the larger elastic/laboratory extension for this submission. Run the following in order and stop when the compute budget is exhausted.

### Priority 0: required to make the current claim credible

1. **Matched-budget, multi-start comparison.** At least three, preferably five, receiver-layout initializations. Run fixed, oracle, waveform misfit, illumination, and a random-motion/random-layout control. Use identical total reconstruction FWI updates and report mean, standard deviation, and every individual point.
2. **Cost table.** For each objective report wall-clock time per outer step, number of wave solves or VJPs, peak GPU memory, number of outer steps, and total FWI iterations.
3. **Proxy--oracle agreement.** Evaluate all three objective values on a shared collection of layouts (saved trajectory points plus small random perturbations). Plot waveform-misfit and illumination value against final model error; report Spearman rank correlation. This experiment directly supports the story and may reuse existing checkpoints.

### Priority 1: if time remains

4. **Controlled simulator-mismatch check.** Compare matched acoustic observations with one deliberately mismatched setting that reuses existing assets: Salvus 3D elastic observations sampled at movable receiver positions, or a cheaper controlled dispersion/density mismatch. Report both the optimized outer loss and true reconstruction error. This directly fits Sim2Science and can turn the earlier failed source-optimization run into an informative limitation rather than hidden history.
5. **Probe-count ablation.** For illumination, use a small set such as 1, 4, 8, and 16 probes. Report gradient/objective variance and final error, not only the best hyperparameter.

### Explicitly defer

- Full 3D elastic/lab-data matching.
- A learned surrogate or neural reconstruction component.
- Large OpenFWI benchmarking.
- A complete implementation of the implicit-adjoint method from Downing et al.

## Five-page allocation

- Abstract: 0.25 page.
- Introduction and contributions: 0.8 page.
- Related work folded into introduction: 0.35 page.
- Method and three objectives: 1.4 pages.
- Experimental setup: 0.6 page.
- Results: 1.1 pages, ideally one workflow figure + one result/correlation figure + one compact table.
- Limitations and conclusion: 0.5 page.

## Figure plan

1. **Main workflow figure:** acquisition coordinates → differentiable wave solver → K-step FWI → interchangeable outer objective → coordinate update. Show the oracle/data/illumination branches in one visual.
2. **Main result figure:** initial versus optimized receiver positions and corresponding reconstructions for fixed/data/illumination, using a shared color scale.
3. **Evidence figure or table:** proxy--oracle correlation and runtime, or a compact multi-seed error plot.

## Open technical facts to resolve before prose is treated as final

- Exact illumination scalarization, normalization, random-probe distribution, probe count, and `illum_beta` meaning.
- Exact source/receiver counts and which coordinates move in the Imperial experiment.
- Confirm the exact full-autodiff result files and environment used for every displayed run.
- For a physical online implementation, state how many receiver reacquisitions are required per design update.
- Exact hardware and matched timing and peak-memory measurements.
