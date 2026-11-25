#!/bin/bash
# Illumination criterion check: score terms vs number of probes, all scored layouts, background model, 4 mm grid, GPU.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="nice -n 10 taskset -c 62-63 /home/merciera/miniconda3/envs/hybridOED/bin/python"
L4="r48 rc30 j30a j30b x48a x48b x48c x48d o48b s31t s31c d30a d30b d30c a30 r36 rc48 x36a x36b r24 x24a"
L2="s2rc30 s2j30a s2j30b s2x48a s2x48b s2x48c s2x48d s2d30 s2a30"
for p in 12 48 192; do
  $P src/ldv_pmma/oed_replay.py --tag terms4_p$p --test --spacing 48 --probes $p --design-impl fast --score-terms $L4 > /dev/null 2> ldv_pmma_results/criterion_check/terms4_p$p.stderr
  echo "$(date '+%F %T') terms4_p$p $(tail -1 ldv_pmma_results/inversion/terms4_p$p/run.log)"
  LDV_FILES=1,3 $P src/ldv_pmma/oed_replay.py --tag terms2_p$p --test --spacing 48 --probes $p --design-impl fast --score-terms $L2 > /dev/null 2> ldv_pmma_results/criterion_check/terms2_p$p.stderr
  echo "$(date '+%F %T') terms2_p$p $(tail -1 ldv_pmma_results/inversion/terms2_p$p/run.log)"
done
echo "$(date '+%F %T') terms finished"
