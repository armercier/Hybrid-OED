#!/bin/bash
# Fewer-shots study (19 Nov): adaptive design (paper workflow, 48 probes) vs random moves of the same lengths,
# with 1 shot (scan file 3 only: S1 + S3 firing together) and 2 shots (files 1 and 3). GPU; 2 CPU cores, reduced priority.
#   A: 1-shot full scan + static starts   B: 1-shot pairs   C: 2-shot pairs   D: 2 more pairs per setting
# Tags: k1<start> static 1-shot, a1_/r1_<start> 1-shot pairs, a2_/r2_<start> 2-shot pairs (starts: s2* subsets).
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
BASE="--spacing 48 --move-all --fwi-first --rounds 5 --fwi-per-round 20 --design-impl fast --src-chunk 1"
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
static(){ T=$1; O=$I/$T; shift
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES $*)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --iters 100 "$@" > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
run(){ T=$1; shift; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES $*)"
  $P src/ldv_pmma/oed_replay.py --tag $T $BASE "$@" > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
pair(){ k=$1; s=$2
  run a${k}_$s --start-subset s2$s --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 12
  if [ ! -f $I/a${k}_$s/settings.json ] && grep -q RESOURCE_EXHAUSTED $I/a${k}_$s/stderr.txt; then
    mv $I/a${k}_$s/stderr.txt $I/a${k}_$s/stderr_chunk12.txt; run a${k}_$s --start-subset s2$s --design-steps 1 --design-lr 8e-3 --probes 48 --probe-chunk 6; fi
  [ -f $I/a${k}_$s/settings.json ] && run r${k}_$s --start-subset s2$s --random-moves-from a${k}_$s; }
export LDV_FILES=3
static k1full --nosurface
for s in rc30 j30a x48c; do static k1$s --subset ldv_work/prepared/subsets/s2$s.npz; done
echo "$(date '+%F %T') A finished"
for s in rc30 j30a x48c; do pair 1 $s; done; echo "$(date '+%F %T') B finished"
export LDV_FILES=1,3
for s in rc30 j30a x48c; do pair 2 $s; done; echo "$(date '+%F %T') C finished"
export LDV_FILES=3
for s in j30b x48a; do static k1$s --subset ldv_work/prepared/subsets/s2$s.npz; pair 1 $s; done
export LDV_FILES=1,3
for s in j30b x48a; do pair 2 $s; done
echo "$(date '+%F %T') fewshots finished"
