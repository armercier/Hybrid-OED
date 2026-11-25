#!/bin/bash
# Repeats of the simulation-only design (as d30a, 4 mm grid) from two other starts, on the GPU, then their inversions. Starts after a30.
cd "$(dirname "$0")/../.." || exit 1
until grep -q "end a30" ldv_pmma_results/queue.log; do sleep 60; done
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
P="nice -n 10 taskset -c 60-63 /home/merciera/miniconda3/envs/hybridOED/bin/python"
for pair in d30b:j30a d30c:j30b; do
  T=${pair%%:*}; S=${pair##*:}; O=ldv_pmma_results/inversion/$T; mkdir -p $O
  echo "$(date '+%F %T') start $T design from $S"
  $P src/ldv_pmma/oed_replay.py --tag $T --test --spacing 48 --probes 12 --design-impl fast --design-lr 1.5e-3 --offline-steps 60 --offline-start $S > /dev/null 2> $O/design_stderr.txt
  if ! grep -q "offline design done" $O/run.log; then echo "$(date '+%F %T') $T design FAILED"; continue; fi
  echo "$(date '+%F %T') $T design finished: $(tail -1 $O/run.log)"
  mv $O/run.log $O/design.log
  if grep -q " 0 of 30 points differ" $O/design.log; then echo "$(date '+%F %T') $T equals $S, no inversion"; continue; fi
  taskset -c 60-63 bash src/ldv_pmma/run_queue.sh $T
done
echo "$(date '+%F %T') d30 repeats finished"
