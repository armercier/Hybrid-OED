#!/bin/bash
# Proposal A (23 Nov): joint inversion of all positions visited by the moving-array runs (gradient ad_/a2_, random rn_/r2_)
# and of a static jittered layout with the same number of points (mj4_*/mj2_*). 100 updates from the blind background,
# exactly like the static layouts. Subsets from make_ever_subsets.py. GPU; 2 CPU cores, nice 10.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
inv(){ T=$1; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, joint static inversion)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$T.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
unset LDV_FILES
for s in rc30 j30a x48c; do inv ev_ad_$s; inv ev_rn_$s; inv mj4_$s; done
echo "$(date '+%F %T') ever 4-shot block finished"
export LDV_FILES=1,3
for s in rc30 j30a j30b x48a x48c; do inv ev_a2_$s; inv ev_r2_$s; inv mj2_$s; done
echo "$(date '+%F %T') ever queue finished"
