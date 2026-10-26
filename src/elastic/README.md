# Plane-stress elastic forward model vs Salvus 3-D plate data

`plane_stress.py` is a 2-D velocity-stress staggered-grid solver with CPML (thin-plate
membrane model: S0 + SH0). `compare_salvus.py` runs it for the 12 Salvus events and
scores it against `src/scripts/extracted_velocity_wavefields_S0/`. Nothing existing is
modified; the CPML profile helper is imported from `hybridoed.forward`.

```bash
python src/elastic/compare_salvus.py --dx 1e-3 --events 1 4 [--with-acoustic]
```

The wavelet comes from `INTERNAL/ENTITIES/SIMULATION_CONFIGURATIONS/test_25kHz.json`
(Ricker, 25 kHz, `time_shift_in_seconds: null` -> peak at t = 0, simulation starting
at the data's first sample, -62.4 us). The material comes from the INTERNAL mesh
(PMMA vp 2730, vs 1340, rho 1180; soft core vs 740 / vp 1330 at the interior GLL
levels), averaged through the thickness with GLL weights (effective vs 820, vp 1520).

## Findings (events 1 and 4, misfit = relative L2 of trace-normalised data)

| model | misfit | lag |
|---|---|---|
| SH0 notebook acoustic set-up (c = 2100) | 1.31 | +29-33 us |
| plane stress, homogeneous PMMA | 0.60 (S0 sector 0.39) | 0 |
| plane stress + inclusion from the INTERNAL mesh | 0.71-0.77 | 0 |
| plane stress, homogeneous, + one source-wavelet filter per event (held-out receivers) | 0.35 | 0 |

* Timing and wave speeds match (zero lag; SH0 1340 m/s, S0 about 2300 m/s).
* The remaining waveform difference travels with SH0 at constant size from 0.12 to
  0.79 m, so it is a source-side difference, largely removed by source-wavelet
  estimation; it does not come from FD resolution (1 mm and 2 mm give the same misfit).
* The Salvus data contain 4-6x less inclusion scattering (mirror-antisymmetry test) than
  the INTERNAL mesh's inclusion produces. The `.nc` files were produced by Salvus run
  `c3/3d/5ffd368d9368`, while the INTERNAL configuration hashes to `c7af5b077022621e`:
  the mesh here is probably not the model behind the extracted data.

## Free-plate benchmark (src/salvus_plate, all parameters known)

`plane_stress_free` (traction-free edges, stress imaging) vs the Salvus 3-D free plate,
all 12 events, in-plane velocity at 3481 top-surface receivers, one amplitude factor per
event (2-D forces are per unit thickness). JAX: 1 mm grid, 1.4 s per solve; converged
(1 mm vs 0.5 mm: 6 %).

```bash
python src/elastic/compare_plate.py --dx 1e-3 [--s0-correction]
```

| mean relative L2 misfit | t < 300 us | t < 500 us | t < 750 us | t < 1 ms |
|---|---|---|---|---|
| total field, plane stress | 0.23 | 0.28 | 0.33 | 0.39 |
| scattered field, plane stress | 0.21 | 0.31 | 0.40 | 0.47 |
| total field, + S0 correction | 0.18 | 0.20 | 0.24 | 0.27 |
| scattered field, + S0 correction | 0.16 | 0.23 | 0.30 | 0.35 |

The remaining error grows with time: plane stress ignores S0 dispersion (exact S0 at
25 kHz in a 10 mm plate is 0.53 % slower; 2.5 % at 50 kHz), which accumulates over the
edge reverberations. `--s0-correction` sets each material's vp so that the plane-stress
plate velocity equals the exact Rayleigh-Lamb S0 phase velocity at f0 (vs unchanged,
SH0 is exact).

## FWI on the Salvus benchmark with plane stress (src/elastic/invert_plate.py)

Events 1, 4, 7, 10, in-plane velocity at all 3481 receivers, vs on a 5 mm grid
(start: homogeneous PMMA, true vs RMSE 23.7 m/s), Adam lr 0.01, 60-80 iterations.
`python src/elastic/plot_regularisation.py` summarises the variants:

| variant | vs RMSE at early stop | vs RMSE final | inclusion mean vs (true 1108) | background std | s/iteration |
|---|---|---|---|---|---|
| baseline, 500 us | 17.5 (it 46) | 20.5 | 1124 | 17.2 | 10.5 |
| + TV 1e-4 | 9.6 | 7.7 | 1127 | 8.7 | 10.7 |
| + TV 3e-4 | 8.7 | 6.7 | 1131 | 7.5 | 10.7 |
| 300 us + edge taper | 17.4 | 18.1 | 1125 | 16.9 | 7.0 |
| 300 us + taper + TV 1e-4 | 8.6 | 6.8 | 1121 | 7.3 | 7.0 |
| 300 us + taper + TV 3e-4 | 8.9 | 6.8 | 1126 | 7.7 | 7.0 |
| inverse crime (plane-stress data) | - | 5.3 | 1116 | 4.5 | 10.5 |

Early stopping rule: misfit decrease < 2 % over 5 iterations (data only).
