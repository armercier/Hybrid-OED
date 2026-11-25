#!/bin/bash
# Re-inversion test (21 Nov): final 30 measured points of every adaptive run (a30 protocol) inverted like a static layout
# (100 updates from calib_blind.json, same command as the static runs). Tags ri_<adaptive tag>. Waits for queue_q1.sh to finish.
# GPU; 2 CPU cores, nice 10.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
until grep -q "same-budget queue finished" ldv_pmma_results/queue.log; do sleep 300; done
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
reinv(){ A=$1; T=ri_$A; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  [ -f $I/$A/result.npz ] || { echo "$(date '+%F %T') missing $A"; return; }
  $P src/ldv_pmma/make_reinv_subsets.py $A > /dev/null
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, static on final points of $A)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$T.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
export LDV_FILES=1,3
for t in s2a30 q2a_j30a q2a_j30b q2a_j30c q2a_j30d; do reinv $t; done
unset LDV_FILES
for t in a30 q4a_j30a q4a_j30b q4a_j30c q4a_j30d; do reinv $t; done
export LDV_FILES=3
for t in q1a_rc30 q1a_j30a q1a_j30b q1a_j30c; do reinv $t; done
echo "$(date '+%F %T') reinversion queue finished"
