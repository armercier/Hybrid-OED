# PINN / FBPINN feasibility (step 1)

Can a scattered-field (FB)PINN reproduce the acoustic wavefield of the paper set-up
accurately and cheaply enough to replace FD-FWI inside the OED loop?
Nothing here modifies the paper scripts; the paper `Simulation` is imported read-only.

Environment: `/home/merciera/miniconda3/envs/hybridOED/bin/python` (JAX 0.4.35, CUDA).

```bash
python src/pinn_feasibility/fd_reference.py                  # FD r1/r2/r4 references (~1 min)
python src/pinn_feasibility/scattered_pinn.py --tag fbpinn_s0                       # joint training
python src/pinn_feasibility/scattered_pinn.py --march-steps 1000 --steps 5000 \
       --tag fbpinn_march_s0                                 # causal time-marching
python src/pinn_feasibility/scattered_pinn.py --fit data --points 256 --steps 20000 \
       --tag fbpinn_datafit_s0                               # physics-free fit (control)
python src/pinn_feasibility/plot_run.py fbpinn_s0 fbpinn_march_s0 fbpinn_datafit_s0
```

Outputs go to `pinn_feasibility_results/` (reference npz ~700 MB, untracked).

Metric: relative L2 error of the scattered traces at 78 receivers (rings around the
inclusion + the nominal diagonal), each trace normalised by its total-field maximum
as in FWI, against the converged r4 FD solution.

## Results, source 0 (RTX 2080 Ti)

| | trace error | field error | PDE loss | time |
|---|---|---|---|---|
| FD r1 (paper grid) | 0.47 | – | – | 0.03 s |
| FD r2 | 0.09 | – | – | 0.10 s |
| FBPINN, joint, 20k steps | 0.78 | 1.22 | ~1e-4 | 1066 s |
| FBPINN, time-marching, 20k + 5k steps | 0.62 | 0.93 | 8e-5 | 1026 s |
| FBPINN, data fit (no physics), 20k steps | 0.025 | 0.042 | 1.2e-3 | 311 s |

Sanity check: the r4 FD field satisfies the scattered-field PDE used by the PINN to
0.8 % relative residual, so the formulation is consistent; the gap is training.

The data-fit control shows the architecture represents the solution well (2.5 %),
yet that accurate field has a ~15x *higher* PDE loss than the wrong physics-trained
fields: the residual loss does not single out the correct solution in this setting,
so the failure is the physics-loss optimisation, not capacity.  (Caveat: part of the
data-fit residual comes from regressing on a trilinearly interpolated target, whose
kinks inflate second derivatives; the comparison is indicative, not exact.)  Verdict: no-go for a
time-domain (FB)PINN as the FWI forward model at this problem size.
