# NeurIPS 2026 and Sim2Science writing guide

**Read this file before every substantial paper edit.** Workshop rules can change; recheck the linked call before submission.

## Submission contract

- Venue: Sim2Science: ML with Imperfect Scientific Models, NeurIPS 2026 workshop.
- Deadline: 29 August 2026, 23:59 Anywhere on Earth.
- Main-paper limit: 5 pages, excluding references.
- Appendix: unlimited, but optional reading; the five-page paper must support every central claim by itself.
- Review: double blind. Anonymize the PDF, source-visible links, code, data, acknowledgments, and self-references.
- Template: official NeurIPS 2026 LaTeX style, workshop double-blind option.
- Checklist: include the standard NeurIPS reproducibility checklist after the references and before the appendix. Missing or incomplete checklists may lead to desk rejection.
- Submission type: non-archival workshop paper. Do not submit the same paper concurrently to both workshops merely to improve acceptance odds; Sim2Science discourages this.
- Reciprocal review: nominate one eligible author at submission time; failure to complete assigned reviews can cause desk rejection.

Sources:

- <https://www.sim2science.com/cfp.html>
- <https://neurips.cc/Conferences/2026/MainTrackHandbook>
- <https://neurips.cc/Conferences/2026/CallForPapers>

## LaTeX rules

- Use `\usepackage[dblblindworkshop]{neurips_2026}` for submission.
- Set both `\title{...}` and `\workshoptitle{Sim2Science}`.
- Do not use `final` or `preprint` for the anonymous submission.
- Never modify `neurips_2026.sty`, margins, font sizes, line spacing, or title spacing.
- Paper size must be US Letter.
- The style uses a 5.5-inch by 9-inch text block, 10-point type with 11-point leading, and Times-family fonts.
- First-level headings are 12 point; lower-level headings are 10 point. Use sentence case: capitalize the first word and proper nouns only.
- Use LaTeX display environments such as `equation`, `equation*`, or `align`; never use bare `$$ ... $$`.
- Use `graphicx` and size figures relative to `\linewidth`; do not position figures by hand.
- Ensure all fonts in the final PDF are embedded Type 1 or TrueType fonts.
- Page numbers and submission line numbers are expected; the style handles them.

## Figures and tables

- Figure captions go **below** figures; table captions go **above** tables.
- Captions must explain the encoding and state the takeaway, not merely name the plotted quantities.
- Use a common color scale for reconstructions that are compared visually.
- Every plot needs labeled axes, units, legible type at final printed size, and a color-blind-safe palette.
- Results must remain interpretable in grayscale where practical.
- Use `booktabs`; never use vertical table rules.
- State sample counts and uncertainty directly in the figure/table or caption.

## Citations and anonymity

- Use one citation style consistently; this project uses numeric, sorted, compressed `natbib` citations.
- Cite primary papers and official software papers rather than surveys when making novelty or method claims.
- Discuss the closest work explicitly, especially Downing et al. (2026). Do not hide it in a long citation list.
- During double-blind review, describe published self-work in the third person: “Mercier et al. show ...”, not “our previous work ...”.
- Do not include identifying acknowledgments or non-anonymous repository links in the submission.
- Verify every bibliography entry against the paper or publisher. Hallucinated citations violate research-integrity expectations.

## Writing standard for this paper

- Use consistent US English (`optimization`, `modeling`) unless the authors deliberately switch the whole paper to UK English.
- Lead each section and paragraph with its claim, then give evidence or mechanism.
- Define every acronym at first use; use `FWI` and `OED` consistently afterward.
- Prefer precise subjects and verbs: “The illumination objective reduces model error” rather than “It can be seen that ...”.
- Separate facts from interpretation. Use “suggests” for the current one-model evidence, not “demonstrates generality”.
- Report absolute values and relative improvements together.
- State what was held constant in every comparison, especially total FWI steps and acquisition budget.
- Do not advertise JAX itself as the novelty. The contribution is the scientific formulation and objective comparison enabled by differentiability.
- Use “automatic differentiation,” not “auto-diff,” in formal prose.
- Use “conditioning,” “literature,” “experiment,” “receiver,” and “source” with careful spelling.

## Claim discipline

The submission may claim:

- continuous coordinate optimization through an unrolled time-domain FWI program;
- objective flexibility in one differentiable workflow;
- a waveform-misfit objective that does not use true-model
  error in the outer loss;
- a stochastic illumination proxy that is empirically cheaper, once timing is measured;
- preliminary synthetic NDT improvements under a matched protocol.

The submission must not claim without new evidence:

- first differentiable or bilevel OED for FWI;
- superiority over classical OED or Downing et al.;
- experimental/laboratory validation;
- robustness or generalization across models;
- real-time or scalable 3D performance.

## Reproducibility checklist for the experiments

Before declaring a result final, record:

- complete forward-model grid, boundary conditions, wavelet, time step, and recording time;
- material-model parameterization and starting model;
- number and initial coordinates of sources and receivers;
- interpolation used for continuous receiver sampling;
- FWI optimizer, learning rates, regularization, smoothing, and number of steps;
- outer optimizer, learning rate, constraints, regularization, and number of steps;
- exact definition and scaling of all three objectives;
- probe distribution, count, seed handling, and scalarization for illumination;
- all random seeds and the number of independent layout initializations;
- fixed versus optimized compute and acquisition budgets;
- hardware, software versions, wall-clock time, and peak memory;
- uncertainty (individual runs plus mean and standard deviation or confidence interval);
- failure cases, unstable hyperparameters, and negative findings;
- whether code/data can be released anonymously and under which license.

## AI-assistance policy

NeurIPS permits writing, editing, grammar, and basic code assistance without a special disclosure. Authors remain responsible for all text, figures, results, and references; agents or language models cannot be authors. If an agent/LLM is an important or non-standard component of the research method, disclose it in the experimental setup. Never add prompt-injection text intended to manipulate reviewing.

## Final pre-submission checks

1. Revisit the workshop call and replace the template/checklist if organizers publish workshop-specific files.
2. Build in anonymous mode and inspect every page at final size.
3. Confirm five content pages maximum; references and allowed appendix/checklist follow afterward.
4. Confirm the title, abstract, introduction, figures, and conclusion all express the same narrow claim.
5. Confirm every central claim has evidence in the main five pages.
6. Check anonymity in PDF metadata, links, file paths visible in figures, code snippets, acknowledgments, and self-citations.
7. Run a spelling/grammar pass and a separate scientific-fact/citation pass.
8. Check fonts, page size, file size, figure resolution, and OpenReview PDF rendering.
