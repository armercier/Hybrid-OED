# Salvus 3-D plate benchmark (every parameter known)

A free PMMA plate simulated with Salvus 3-D elastic, used as "real" data for the 2-D
JAX solvers. The full definition is written to `salvus_plate/benchmark_definition.json`.

| | value |
|---|---|
| plate | 0.6 m x 0.6 m x 10 mm, x, y in [0, 0.6] m, **free on all faces** (no absorbing layers) |
| material | PMMA vp 2730, vs 1340 m/s, rho 1180 kg/m^3 |
| inclusion | 60 mm square through the thickness, centre (0.25, 0.35) m; vs 1100, vp 2240, same rho; edges on element boundaries |
| mesh | 60 x 60 x 1 hexahedra (10 mm), spectral order 4 |
| sources | 12 in-plane point forces at mid-plane (S0 + SH0 only), transverse to their side, sides 0.2 m from the centre, offsets -0.1/0/+0.1 m |
| wavelet | Ricker 25 kHz, Salvus default (peak at t = 0, start -62.4 us), end 1 ms |
| receivers | 59 x 59 grid, 10 mm, top surface, 3-component velocity, 1.19 us sampling |
| runs | `plate_inclusion` and `plate_homogeneous`, 12 events each; about 6-9 s per run on the RTX 2080 Ti |

```bash
V=/home/merciera/salvus_data/venv-salvus-2025.1.2/bin/python
$V src/salvus_plate/build_and_run.py     # build the project, run 24 simulations on site local_gpu
$V src/salvus_plate/export_data.py       # -> salvus_plate/plate_benchmark.npz
```

Set-up notes:
* The solver binary in `~/Salvus` is 2025.1.2 while the `salvus` conda env has the
  2025.1.1 package; Salvus refuses mismatched versions. The venv above is built on the
  `salvus` env (`--system-site-packages`) with the matching wheel from
  `~/Salvus/python_packages`; the conda env itself is unchanged.
* The site `local_gpu` (same binary and directories as `local`, with
  `use_cuda_capable_gpus = true`) was added to `~/.config/SalvusFlow/1.0/salvus-flow-config.toml`;
  a backup of the previous file sits next to it.
