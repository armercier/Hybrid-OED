#!/bin/bash
# Adaptive run a30 on the GPU, after the d30a inversion. CPU use is limited: 4 cores, reduced priority, 2 BLAS/OpenMP threads.
cd "$(dirname "$0")/../.." || exit 1
until grep -qE "end d30a|d30a equals rc30|d30a design FAILED" ldv_pmma_results/queue.log; do sleep 60; done
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="nice -n 10 taskset -c 60-63 /home/merciera/miniconda3/envs/hybridOED/bin/python"
T=a30; O=ldv_pmma_results/inversion/$T; mkdir -p $O
echo "$(date '+%F %T') start $T"
$P src/ldv_pmma/oed_replay.py --tag $T --spacing 48 --layout-groups rc30 --rounds 3 --fwi-per-round 33 --fwi-total 100 --fwi-first \
  --design-steps 20 --design-lr 1.5e-3 --design-fresh-probes --design-valid-every 10 --probes 12 --design-impl fast --src-chunk 1 > /dev/null 2> $O/stderr.txt
rc=$?
[ -f $O/settings.json ] && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $T > /dev/null
echo "$(date '+%F %T') end $T (exit $rc)"
