#!/bin/bash
# Simulation-only illumination design of a 30-point layout (CPU, 4 mm grid), then its inversion on the GPU after the 18 Nov queue.
cd "$(dirname "$0")/../.." || exit 1
P=/home/merciera/miniconda3/envs/hybridOED/bin/python
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false $P src/ldv_pmma/oed_replay.py --tag d30a --test --spacing 48 --probes 12 --design-impl fast --design-lr 1.5e-3 --offline-steps 60 --offline-start rc30 > /dev/null 2> ldv_pmma_results/d30a_design.stderr
L=ldv_pmma_results/inversion/d30a/run.log
grep -q "offline design done" $L || { echo "$(date '+%F %T') d30a design FAILED"; exit 1; }
echo "$(date '+%F %T') d30a design finished: $(tail -1 $L)"
if grep -q " 0 of 30 points differ" $L; then echo "$(date '+%F %T') d30a equals rc30, no inversion"; exit 0; fi
until grep -q "18nov queue finished" ldv_pmma_results/queue.log; do sleep 30; done
mv $L ldv_pmma_results/inversion/d30a/design.log
bash src/ldv_pmma/run_queue.sh d30a
