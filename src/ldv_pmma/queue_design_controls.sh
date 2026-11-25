#!/bin/bash
# Design controls (23 Nov), after queue_regular143.sh. GPU; 2 CPU cores, nice 10.
#  1. Simulation design (designed once on the defect-free model, 60 Adam steps of 1.5 mm, then measured and inverted like a
#     static layout), as d30*/s2d30: 2 shots s2d30b (from s2j30a), s2d30c (s2j30b); 1 shot q1d_rc30/j30a/j30b (from f3*).
#  2. Offline moving array: the move-all protocol of ad_/a2_/a1_ (same starts, 5 rounds, one 8 mm Adam step, 48 probes,
#     5 x 20 staged FWI) with --design-on-background (moves computed on the background model, not the current inversion).
#     Tags od_ (4 shots), o2_ (2 shots), o1_ (1 shot).
#  3. Joint inversion of all positions visited by each offline moving run (ev_od_*, ev_o2_*, ev_o1_*), as proposal A.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
until grep -q "regular143 queue finished" ldv_pmma_results/queue.log; do sleep 300; done
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
inv(){ T=$1; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, static inversion of subset $T)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$T.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
sim(){ T=$1; S=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T simulation design from $S (LDV_FILES=$LDV_FILES)"
  $P src/ldv_pmma/oed_replay.py --tag $T --test --spacing 48 --probes 12 --design-impl fast --design-lr 1.5e-3 --offline-steps 60 \
    --offline-start $S > /dev/null 2> $O/design_stderr.txt
  if ! grep -q "offline design done" $O/run.log; then echo "$(date '+%F %T') $T design FAILED"; return; fi
  echo "$(date '+%F %T') $T design finished: $(tail -1 $O/run.log)"; mv $O/run.log $O/design.log
  inv $T; }
BASE="--spacing 48 --move-all --fwi-first --rounds 5 --fwi-per-round 20 --design-impl fast --src-chunk 1"
mov(){ T=$1; S=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, offline moving array from $S)"
  $P src/ldv_pmma/oed_replay.py --tag $T $BASE --start-subset $S --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 12 \
    --design-on-background > /dev/null 2> $O/stderr.txt
  if [ ! -f $O/settings.json ] && grep -q RESOURCE_EXHAUSTED $O/stderr.txt; then mv $O/stderr.txt $O/stderr_chunk12.txt
    $P src/ldv_pmma/oed_replay.py --tag $T $BASE --start-subset $S --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 6 \
      --design-on-background > /dev/null 2> $O/stderr.txt; fi
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
ev(){ [ -f $I/$1/settings.json ] || { echo "$(date '+%F %T') missing $1"; return; }
  $P src/ldv_pmma/make_ever_subsets.py $1 > /dev/null; inv ev_$1; }
# 1. simulation design
export LDV_FILES=1,3; sim s2d30b s2j30a; sim s2d30c s2j30b
export LDV_FILES=3;   for s in rc30 j30a j30b; do sim q1d_$s f3$s; done
echo "$(date '+%F %T') simulation design block finished"
# 2. offline moving array
unset LDV_FILES;      for s in rc30 j30a x48c; do mov od_$s $s; done
export LDV_FILES=1,3; for s in rc30 j30a j30b x48a x48c; do mov o2_$s s2$s; done
export LDV_FILES=3;   for s in rc30 j30a j30b x48a x48c; do mov o1_$s s2$s; done
echo "$(date '+%F %T') offline moving block finished"
# 3. all visited positions of the offline moving runs
unset LDV_FILES;      for s in rc30 j30a x48c; do ev od_$s; done
export LDV_FILES=1,3; for s in rc30 j30a j30b x48a x48c; do ev o2_$s; done
export LDV_FILES=3;   for s in rc30 j30a j30b x48a x48c; do ev o1_$s; done
echo "$(date '+%F %T') design controls queue finished"
