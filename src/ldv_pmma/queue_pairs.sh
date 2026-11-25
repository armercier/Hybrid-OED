#!/bin/bash
# Fixed-sensor-count study: adaptive design (paper workflow, 48 probes) vs random moves of the same lengths, from 3 starts. GPU; 2 CPU cores.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
BASE="--spacing 48 --move-all --fwi-first --rounds 5 --fwi-per-round 20 --design-impl fast --src-chunk 1"
run(){ T=$1; shift; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T ($*)"
  $P src/ldv_pmma/oed_replay.py --tag $T $BASE "$@" > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"
  [ -f $O/settings.json ] && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $T > /dev/null; }
for s in rc30 j30a x48c; do
  run ad_$s --start-subset $s --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 12
  if [ ! -f $I/ad_$s/settings.json ] && grep -q RESOURCE_EXHAUSTED $I/ad_$s/stderr.txt; then
    mv $I/ad_$s/stderr.txt $I/ad_$s/stderr_chunk12.txt; run ad_$s --start-subset $s --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 6; fi
  [ -f $I/ad_$s/settings.json ] && run rn_$s --start-subset $s --random-moves-from ad_$s
done
echo "$(date '+%F %T') pairs finished"
