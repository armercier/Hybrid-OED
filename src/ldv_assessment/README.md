# LDV data assessment (grid_scan_..._scatter.mat)

Run in order (hybridOED env): `ldv_analysis.py` (scan plane, rotation to plate frame,
noise), `ldv_modes.py` (band-pass, SNR, arrival-time source/speed fits), `ldv_fig.py`
(figure `ldv_assessment/ldv_components.png`). Intermediate files go to `ldv_assessment/`.

Findings:
* scan: 0.33 m x 0.11 m patch, 1.5 mm grid, plane ~1 deg from the scanner xy plane;
* out-of-plane component: SNR ~90 after 10-45 kHz band-pass; one dispersive packet,
  energy at ~1320 m/s = A0 group velocity of 10 mm PMMA at ~39 kHz (spectral peak);
* in-plane components: SNR 5-9, same ring pattern and sign as the out-of-plane field
  (cross-talk from the normal component, not radial in-plane motion) plus dropout spikes:
  no usable S0/SH0 signal;
* trigger delay ~200 us; broadband excitation (-6 dB up to ~50 kHz), not a 25 kHz Ricker.

## Mindlin reconstruction of the out-of-plane field

Solver: `src/elastic/mindlin.py`, validated against Salvus 3-D normal-force events
(`src/salvus_plate/flexural_events.py`, `src/elastic/compare_mindlin.py`): with the shear
factor matched to exact A0 dispersion (kappa ~ 0.89 instead of pi^2/12) and a 0.5 mm grid,
misfit 0.16-0.20 (total) and 0.07-0.20 (scattered) for a mid-plane force.

LDV fit (`fit_mindlin_ldv.py`, figure `plot_mindlin_fit.py`): open plate (sponge), PMMA
vs 1340 m/s (vp/vs 2.037), h = 10 mm (best of a vs x h grid), normal point source at
(-0.097, -0.005) m in the scan frame, one deconvolution filter (wavelet + ~170 us trigger
delay) estimated on half of the scan points. Held-out misfit by region:

| region | misfit | share of residual energy |
|---|---|---|
| all held-out points | 0.56 | 100 % |
| near source (r < 4 cm; finite transducer, near field) | 0.60 | 73 % |
| scan row through the source (LDV dropouts) | 0.88 | 64 % |
| clean propagation region between source and scatterer | 0.34 | 8 % |
| around / behind the scatterer (not in the homogeneous model) | 0.60 | 15 % |

The amplitude-ratio map (`amplitude_maps.png`) shows a sharp shadow cone starting at about
(-0.02, -0.005) m: a strong scatterer ~7.5 cm from the source. A parametric circular
thickness change (`fit_scatterer_ldv.py`) barely improves the fit (0.556 -> 0.551); the
scatterer needs an inversion rather than a guessed shape.

## Salvus 3-D model of the experiment: 10 mm PMMA plate + bonded 10 mm steel cube

`salvus_ldv_cube.py` (Salvus venv, GPU, ~80 s per run): free PMMA plate extending 0.35 m
beyond the scan, 5 mm order-4 hexahedra, steel cube (vp 5900, vs 3200, rho 7850) as a
2x2x2-element block on top, normal top-surface source at the transducer position, vz at
the scan points. Optional constant-Q PMMA (`--q`, QKAPPA = 2 QMU, linear solids attached
with Salvus' private `_attach_linear_solids`). `compare_salvus_ldv.py` scores against the
LDV data (same deconvolution / held-out pipeline), `plot_salvus_ldv.py` -> `salvus_ldv_cube.png`.

| Salvus model | misfit behind cube | misfit before cube | on-axis obs/model RMS, 5 -> 25 cm |
|---|---|---|---|
| plate only | 0.557 | 0.382 | 1.32 ... 0.51 (shadow missing) |
| cube at x = -25 mm | 0.430 | 0.431 | 1.16 ... 0.74 |
| cube at x = -20 mm | 0.354 | 0.344 | 1.14 ... 0.71 (smooth decay: attenuation) |
| cube at x = -15 mm | 0.408 | 0.379 | 1.16 ... 0.72 |
| cube at -20 mm, PMMA Q = 25 | 0.330 | 0.334 | 1.05 ... 1.13 (over-attenuated) |
| cube at -20 mm, PMMA Q = 40 | 0.330 | 0.334 | 1.07 ... 0.96 (flat) |

The cube centre (scan frame) is at about (-0.020, -0.005) m, 7.7 cm from the source.

## Mindlin equivalent of the experiment (`mindlin_equivalent.py`)

`src/elastic/mindlin.py` gained `mass_scale`, `inertia_scale`, `stiffness_scale` (per-node
factors on rho h, rho I, D and kappa G h) and `damping` (velocity damping rate, Q ~ omega / a).
The steel cube becomes a 10 mm x 10 mm patch with the physical mass (x7.65) and rotary
inertia about the plate mid-plane (x87.5); stiffness is fitted (x10), damping 6000 / s
(~ Q = 40 at 39 kHz). Calibrated in stages against the noise-free Salvus models
(same deconvolution / held-out pipeline):

| stage | comparison | behind cube | before cube |
|---|---|---|---|
| A | Mindlin plate vs Salvus plate only | 0.14 | 0.14 |
| A' | Mindlin plate vs Salvus + cube | 0.42 | - |
| B | + patch: mass only | 0.48 | 0.25 |
| B | + patch: mass, rotary inertia, stiffness x10 | 0.31 | 0.25 |
| C | + damping 6000 / s vs Salvus + cube, Q = 40 | 0.28 | 0.20 |
| D | Mindlin equivalent vs LDV | 0.41 | 0.39 |
| D | plain Mindlin plate vs LDV | 0.55 | 0.37 |
| D | Salvus 3-D (cube, Q = 40) vs LDV | 0.33 | 0.33 |

## Rigid-cube Mindlin (`mindlin_equivalent.py --rigid`)

`mindlin_plate(..., rigid_mask, rigid_added_mass, rigid_added_inertia)` constrains a
footprint to rigid-body motion (deflection + two rotations), projecting the footprint
impulse onto these modes with plate + cube mass (7.85 g) and rotary inertia about the
mid-plane (M (a^2/12 + d^2), d = 10 mm). No patch stiffness, so no internal resonance.
Sweep of the effective rigid footprint (full inertia, damping 6000 / s), scored behind/before:

| footprint | vs Salvus cube Q = 40 | vs LDV |
|---|---|---|
| 10 mm | 0.275 / 0.233 | 0.399 / 0.410 |
| 8 mm | 0.194 / 0.156 | 0.342 / 0.363 |
| **6 mm (best on Salvus)** | **0.158 / 0.112** | **0.308 / 0.327** |
| 5 mm | 0.168 / 0.118 | 0.304 / 0.317 |
| 4 mm | 0.194 / 0.140 | 0.313 / 0.314 |
| 2 mm | 0.280 / 0.191 | 0.368 / 0.326 |

Reference: Mindlin plate vs Salvus plate only 0.14; Salvus 3-D vs LDV 0.330 / 0.334.
Halving the coupled rotary inertia makes every footprint worse.
