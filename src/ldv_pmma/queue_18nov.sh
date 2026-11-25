#!/bin/bash
# 18 Nov GPU queue, detached:  setsid nohup bash src/ldv_pmma/queue_18nov.sh >> ldv_pmma_results/queue.log 2>&1 < /dev/null &
# Finished runs (settings.json present) are skipped, so it can be restarted.
cd "$(dirname "$0")/../.." || exit 1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
P=/home/merciera/miniconda3/envs/hybridOED/bin/python
BASE="--spacing 48 --rounds 5 --fwi-per-round 20 --probes 12 --design-impl fast --src-chunk 1 --fwi-first"
oed(){ T=$1; shift; O=ldv_pmma_results/inversion/$T
  if [ -f "$O/settings.json" ]; then echo "$(date '+%F %T') skip $T (done)"; return; fi
  mkdir -p "$O"; echo "$(date '+%F %T') start $T ($*)"
  $P src/ldv_pmma/oed_replay.py --tag "$T" $BASE "$@" > /dev/null 2> "$O/stderr.txt"; rc=$?
  [ -f "$O/settings.json" ] && CUDA_VISIBLE_DEVICES="" $P src/ldv_pmma/plot_inversion.py --tag "$T" > /dev/null
  echo "$(date '+%F %T') end $T (exit $rc)"; }
G=ldv_pmma_results/inversion/gc_o48f
if ! grep -q "grad check done" "$G/run.log" 2>/dev/null; then
  mkdir -p "$G"; echo "$(date '+%F %T') start gc_o48f"
  $P src/ldv_pmma/oed_replay.py --tag gc_o48f --spacing 48 --probes 12 --design-impl fast --src-chunk 1 --grad-check o48f > /dev/null 2> "$G/stderr.txt"
  echo "$(date '+%F %T') end gc_o48f (exit $?)"
fi
bash src/ldv_pmma/run_queue.sh rc30 rc48
oed s31t --select targeted --budget 31
oed s31c --select coverage --budget 31
bash src/ldv_pmma/run_queue.sh x48c x48d j30a j30b
oed o48r --move-all --random-moves-from o48f
echo "$(date '+%F %T') 18nov queue finished"
