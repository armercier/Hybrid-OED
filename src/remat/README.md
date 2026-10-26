# Rematerialised hypergradients (`jax.checkpoint`)

The paper scripts are imported unchanged. `make_remat_class()` in `remat_hypergrad.py`
returns a subclass of the paper `Simulation` that differs only in:

* `fwi_core`: each unrolled SGD FWI update is wrapped in `jax.checkpoint`, so only the
  carried model is stored per update and the forward and adjoint wave simulations are
  recomputed in the outer backward pass (oracle and waveform-misfit objectives).
* `illumination_loss`: sources are handled one at a time and probes in batches
  (`lax.map`), each checkpointed, instead of one `vmap` over all sources and probes.
  The probes and averaging are the same.

Environment: `/home/merciera/miniconda3/envs/hybridOED/bin/python`.

```bash
# one outer hypergradient, paper vs remat (gradient, time, memory)
python src/remat/remat_hypergrad.py --method oracle --remat none --device cpu --updates 20
python src/remat/remat_hypergrad.py --method oracle --remat step --device gpu --updates 20
python src/remat/remat_hypergrad.py --method illumination --remat step --device gpu --probe-batch 12

# full paper protocol (paper seeds) on GPU, one method per process
python src/remat/run_multiseed_remat.py --num-repetitions 10 --master-seed 20250813 \
    --output-dir remat_results/multiseed_gpu
```

Use `--device auto` (the wrapper's default) for GPU: in this environment the paper flag
`--device gpu` sets `JAX_PLATFORMS=gpu`, which tries ROCm and fails.

## One hypergradient (K = 20)

| objective | setting | peak memory | time | rel. grad diff vs paper |
|---|---|---|---|---|
| oracle | CPU, paper | 64.2 GiB | 95 s | – |
| oracle | CPU, remat | 4.3 GiB | 116 s | 2.8e-6 |
| oracle | GPU, remat | 3.1 GiB | 7.1 s | 2.5e-6 |
| waveform | CPU, paper | 66.0 GiB | 99 s | – |
| waveform | CPU, remat | 4.8 GiB | 120 s | 6.3e-6 |
| waveform | GPU, remat | 5.9 GiB | 7.4 s | 6.4e-6 |
| illumination | CPU, paper | 9.7 GiB | 36 s | – |
| illumination | CPU, remat | 2.4 GiB | 99 s | 4.7e-6 |
| illumination | GPU, remat (probe batch 4 / 12) | 2.4 / 2.8 GiB | 4.2 / 3.6 s | – |

The paper versions of all three objectives exceed the 11 GB GPU (OOM).

## Full multi-seed experiment on GPU vs paper (CPU), same seeds

| method | MSE paper | MSE GPU+remat | improvement paper / GPU | max per-run MSE diff | runtime paper / GPU | GPU peak |
|---|---|---|---|---|---|---|
| fixed layout | 3358.9 ± 65.5 | 3358.9 ± 65.5 | – | 0.000 % | 166 s / 9.8 s (17×) | 1.2 GiB |
| illumination | 2796.2 ± 274.6 | 2796.2 ± 274.7 | 16.7 / 16.7 %, 10/10 | 0.005 % | 357 s / 37 s (9.6×) | 2.4 GiB |
| waveform misfit | 3041.6 ± 125.4 | 3041.3 ± 125.9 | 9.4 / 9.4 %, 10/10 | 0.30 % | 503 s / 43 s (11.6×) | 5.9 GiB |
| oracle | 2732.3 ± 242.4 | 2732.3 ± 242.4 | 18.6 / 18.6 %, 10/10 | 0.004 % | 479 s / 41 s (11.8×) | 3.1 GiB |

The whole experiment takes about 22 min on the RTX 2080 Ti, against about 4.2 h on the
Threadripper. Runtimes include JAX compilation, as in the paper.
