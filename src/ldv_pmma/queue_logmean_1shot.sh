#!/bin/bash
# 1-shot block of the log criterion (22 Nov): same as queue_logmean.sh (a30 protocol, --criterion logmean, each run followed by
# its re-inversion), file 3 only, same starts as q1a_* / q1n_* (f3rc30, f3j30a-c). Tags q1l_*, ri_q1l_*.
# Waits for "logmean queue finished". GPU; 2 CPU cores, nice 10.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
until grep -q "logmean queue finished" ldv_pmma_results/queue.log; do sleep 300; done
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
adaptive(){ T=$1; L=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, groups from $L, criterion logmean)"
  $P src/ldv_pmma/oed_replay.py --tag $T --spacing 48 --layout-groups $L --rounds 3 --fwi-per-round 33 --fwi-total 100 --fwi-first \
    --design-steps 20 --design-lr 1.5e-3 --design-fresh-probes --design-valid-every 10 --probes 12 --design-impl fast --src-chunk 1 \
    --criterion logmean > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
reinv(){ A=$1; T=ri_$A; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  [ -f $I/$A/result.npz ] || { echo "$(date '+%F %T') missing $A"; return; }
  $P src/ldv_pmma/make_reinv_subsets.py $A > /dev/null
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, static on final points of $A)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$T.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
export LDV_FILES=3
adaptive q1l_rc30 f3rc30; reinv q1l_rc30
for s in j30a j30b j30c; do adaptive q1l_$s f3$s; reinv q1l_$s; done
echo "$(date '+%F %T') logmean 1-shot queue finished"
