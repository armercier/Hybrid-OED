#!/bin/bash
# Two-source study (scan files 1 and 3: S1 and S3), GPU, after the d30 repeats. CPU use kept low: 2 cores, reduced priority.
cd "$(dirname "$0")/../.." || exit 1
until grep -q "d30 repeats finished" ldv_pmma_results/queue.log; do sleep 60; done
export LDV_FILES=1,3 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/merciera/miniconda3/envs/hybridOED/bin/python; P="nice -n 10 taskset -c 62-63 $PY"; C=ldv_pmma_results/calib/calib_blind.json; I=ldv_pmma_results/inversion
plot(){ CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $P src/ldv_pmma/plot_inversion.py --tag "$1" > /dev/null; }
if [ ! -f $I/s2full/settings.json ]; then mkdir -p $I/s2full; echo "$(date '+%F %T') start s2full"
  $P src/ldv_pmma/invert_thickness.py --tag s2full --background $C --no-cube --nosurface --iters 100 > $I/s2full/run.log 2>&1; echo "$(date '+%F %T') end s2full (exit $?)"; plot s2full; fi
nice -n 10 taskset -c 62-63 bash src/ldv_pmma/run_queue.sh s2rc30 s2j30a s2j30b s2x48a s2x48b s2x48c s2x48d
T=s2d30; O=$I/$T
if [ ! -f $O/settings.json ]; then mkdir -p $O; echo "$(date '+%F %T') start $T design"
  $P src/ldv_pmma/oed_replay.py --tag $T --test --spacing 48 --probes 12 --design-impl fast --design-lr 1.5e-3 --offline-steps 60 --offline-start s2rc30 > /dev/null 2> $O/design_stderr.txt
  if grep -q "offline design done" $O/run.log; then echo "$(date '+%F %T') $T design finished: $(tail -1 $O/run.log)"; mv $O/run.log $O/design.log
    grep -q " 0 of 30 points differ" $O/design.log || nice -n 10 taskset -c 62-63 bash src/ldv_pmma/run_queue.sh $T
  else echo "$(date '+%F %T') $T design FAILED"; fi; fi
T=s2a30; O=$I/$T
if [ ! -f $O/settings.json ]; then mkdir -p $O; echo "$(date '+%F %T') start $T"
  $P src/ldv_pmma/oed_replay.py --tag $T --spacing 48 --layout-groups s2rc30 --rounds 3 --fwi-per-round 33 --fwi-total 100 --fwi-first --design-steps 20 --design-lr 1.5e-3 --design-fresh-probes --design-valid-every 10 --probes 12 --design-impl fast --src-chunk 1 > /dev/null 2> $O/stderr.txt
  echo "$(date '+%F %T') end $T (exit $?)"; [ -f $O/settings.json ] && plot $T; fi
echo "$(date '+%F %T') s2 queue finished"
