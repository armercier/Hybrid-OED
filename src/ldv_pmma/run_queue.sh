#!/bin/bash
# Run reduced-scan jobs one after another on the GPU, detached from the terminal / Claude session.
# Skips a job whose settings.json already exists (finished), so the queue can be restarted at any time.
#   setsid nohup bash src/ldv_pmma/run_queue.sh bench_o9 r12 r18 r24 > ldv_pmma_results/queue.log 2>&1 &
# Job names: bench_o9 (OED design benchmark), r<s>/x<s>a/x<s>b (subset inversions), o<s> (OED replay).
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
P=/home/merciera/miniconda3/envs/hybridOED/bin/python
C=ldv_pmma_results/calib/calib_blind.json
for id in "$@"; do
  out=ldv_pmma_results/inversion/$id
  if [ -f "$out/settings.json" ]; then echo "$(date '+%F %T') skip $id (done)"; continue; fi
  mkdir -p "$out"
  echo "$(date '+%F %T') start $id"
  case $id in
    bench_*) $P src/ldv_pmma/oed_replay.py --tag "$id" --spacing 9 --bench ;;
    o*)      $P src/ldv_pmma/oed_replay.py --tag "$id" --spacing "${id#o}" ${OED_ARGS} ;;
    *)       $P src/ldv_pmma/invert_thickness.py --tag "$id" --background $C --no-cube \
               --subset ldv_work/prepared/subsets/$id.npz --iters 100 > "$out/run.log" 2>&1 ;;
  esac
  rc=$?
  [ -f "$out/settings.json" ] && CUDA_VISIBLE_DEVICES="" $P src/ldv_pmma/plot_inversion.py --tag "$id" > /dev/null
  echo "$(date '+%F %T') end $id (exit $rc)"
done
echo "$(date '+%F %T') queue finished"
