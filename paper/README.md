# Hybrid OED workshop paper

This folder contains the working paper for the Sim2Science NeurIPS 2026 workshop.

## Start writing

Open `hybrid-oed-paper.code-workspace` in VS Code. Install the **LaTeX Workshop** extension if it is not already installed, then open and save `main.tex`. The configured recipe builds the PDF into `build/main.pdf`.

From a terminal in this folder:

```sh
make
```

Use `make clean` to remove LaTeX intermediate files.

## Files

- `main.tex`: entry point and draft abstract.
- `sections/`: manuscript sections.
- `references.bib`: checked starting bibliography.
- `STORYLINE.md`: venue recommendation, narrative, novelty boundaries, and short experiment plan.
- `NEURIPS_STYLE_GUIDE.md`: mandatory writing and formatting checklist.
- `neurips_2026.sty` and `checklist.tex`: official NeurIPS 2026 files downloaded from the conference template on 3 August 2026.

## Template status

Sim2Science currently says its own link will appear when the NeurIPS 2026 template is published. This project therefore uses the official NeurIPS 2026 `dblblindworkshop` option. Recheck the workshop call before submission and replace these files only if the organizers provide a workshop-specific package or checklist.
