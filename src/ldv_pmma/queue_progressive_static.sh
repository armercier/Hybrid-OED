#!/bin/bash
# Progressive static control (21 Nov): every adaptive start layout measured as its 3 interleaved groups of 10 WITHOUT design
# (--no-design), with exactly the adaptive inversion schedule (33/33/34 FWI updates, FWI first). Paired with the adaptive runs at
# the same start, so the only difference is the design. Tags: n30 (rc30, pairs a30), s2n30 (s2rc30, pairs s2a30),
# q4n_* / q2n_* / q1n_* (pair q4a_* / q2a_* / q1a_*). Waits for queue_reinv.sh. GPU; 2 CPU cores, nice 10.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; I=ldv_pmma_results/inversion
until grep -q "reinversion queue finished" ldv_pmma_results/queue.log; do sleep 300; done
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag $1 > /dev/null; }
prog(){ T=$1; L=$2; O=$I/$T
  if [ -f $O/settings.json ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p $O; echo "$(date '+%F %T') start $T (LDV_FILES=$LDV_FILES, groups from $L, no design)"
  $P src/ldv_pmma/oed_replay.py --tag $T --spacing 48 --layout-groups $L --rounds 3 --fwi-per-round 33 --fwi-total 100 --fwi-first \
    --no-design --probes 12 --design-impl fast --src-chunk 1 > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; }
export LDV_FILES=1,3
prog s2n30 s2rc30; for s in j30a j30b j30c j30d; do prog q2n_$s s2$s; done
unset LDV_FILES
prog n30 rc30; for s in j30a j30b j30c j30d; do prog q4n_$s $s; done
export LDV_FILES=3
prog q1n_rc30 f3rc30; for s in j30a j30b j30c; do prog q1n_$s f3$s; done
echo "$(date '+%F %T') progressive static queue finished"
