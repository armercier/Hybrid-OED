#!/bin/bash
# Proposal A, regular reference (23 Nov): rg4_143 / rg2_143, cell-centred 13 x 11 (143 points), 100 updates from the blind
# background like every static layout. Waits for "ever queue finished". GPU; 2 CPU cores, nice 10.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
until grep -q "ever queue finished" ldv_pmma_results/queue.log; do sleep 300; done
inv(){ T=$1; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, regular 13 x 11)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$T.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"
  [ -f $O/settings.json ] && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $T > /dev/null; }
unset LDV_FILES; inv rg4_143
export LDV_FILES=1,3; inv rg2_143
echo "$(date '+%F %T') regular143 queue finished"
