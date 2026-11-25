#!/bin/bash
# Same-budget study (Q1, 20 Nov): adaptive illumination design at 30 measured points (a30 protocol: 3 interleaved groups of 10,
# group 0 measured as is, 20 Adam steps of 1.5 mm per later group on the current model, 12 probes, validation every 10 steps,
# 33/33/34 FWI updates) for 1, 2 and 4 shots, plus 1-shot static runs on the new 30-point layouts. GPU; 2 CPU cores, nice 10.
# Tags: q1a_* / q2a_* / q4a_* adaptive (1/2/4 shots), q1s_* static 1 shot. Layouts: make_subsets_q1.py (+ existing j30a/b, s2j30a/b).
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
C=ldv_pmma_results/calib/calib_blind.json
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
adaptive(){ T=$1; L=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, groups from $L)"
  $P src/ldv_pmma/oed_replay.py --tag $T --spacing 48 --layout-groups $L --rounds 3 --fwi-per-round 33 --fwi-total 100 --fwi-first \
    --design-steps 20 --design-lr 1.5e-3 --design-fresh-probes --design-valid-every 10 --probes 12 --design-impl fast --src-chunk 1 > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
static(){ T=$1; L=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, static $L)"
  $P src/ldv_pmma/invert_thickness.py --tag $T --background $C --no-cube --subset ldv_work/prepared/subsets/$L.npz --iters 100 > $O/run.log 2>&1
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
export LDV_FILES=3
adaptive q1a_rc30 f3rc30
for s in rc30 j30a j30b j30c j30d x30a x30b; do static q1s_$s f3$s; done
for s in j30a j30b j30c; do adaptive q1a_$s f3$s; done
echo "$(date '+%F %T') q1 block finished"
export LDV_FILES=1,3
for s in j30a j30b j30c j30d; do adaptive q2a_$s s2$s; done
echo "$(date '+%F %T') q2 block finished"
unset LDV_FILES
for s in j30a j30b j30c j30d; do adaptive q4a_$s $s; done
echo "$(date '+%F %T') same-budget queue finished"
