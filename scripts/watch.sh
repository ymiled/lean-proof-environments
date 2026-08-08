#!/usr/bin/env bash
# One-screen status for a run on the rented host. Run this locally.
#
#     bash scripts/watch.sh            # print once
#     bash scripts/watch.sh -f         # refresh every 60s until the run ends
#
# Reads the four things worth reading, in the order they decide whether the run
# is working, and nothing else. The training log is mostly vLLM progress bars.

set -uo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-ssh2.vast.ai}"
PORT="${PORT:-14986}"
REMOTE="${REMOTE:-/root/pdd}"
FOLLOW=0
[[ "${1:-}" == "-f" ]] && FOLLOW=1

ssh_() { ssh -o BatchMode=yes -o ConnectTimeout=25 -p "$PORT" "root@$HOST" "$@" 2>/dev/null; }

status() {
  local run log
  run=$(ssh_ "cd $REMOTE && ls -td runs/planb-* runs/grpo-* 2>/dev/null | head -1" | tr -d '\r' | sed 's#/$##')
  [[ -n "$run" ]] || { echo "no run found on $HOST"; return 1; }
  log="$REMOTE/$run/train.log"

  printf '\n=== %s   %s ===\n' "$run" "$(date +%H:%M:%S)"

  # 1. Alive, and what the card is doing. A run that has died leaves the log
  #    intact, so "the last line looks fine" is not evidence of anything.
  # Counts the uv wrapper as well as the interpreter, so 0 means dead and
  # anything else means alive; the exact number is not meaningful.
  printf 'alive     %s\n' "$([[ $(ssh_ "pgrep -cf '[p]dd.train_grpo'" | tr -d '\r') -gt 0 ]] && echo yes || echo 'NO -- run has died')"
  printf 'gpu       %s\n' "$(ssh_ 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader' | tr -d '\r')"

  # 2. The result. Held-out tables are the only numbers that speak to
  #    generalisation; everything else is training-split fit.
  echo
  echo '--- held-out evaluations ---'
  ssh_ "tr '\r' '\n' < $log | grep -aA6 'depth  tasks' | tail -28" || echo '  none yet'

  # 3. Whether the optimiser is receiving anything. A degenerate rate near 1.0
  #    means every rollout group scored the same, so the advantage is zero and
  #    the update is exactly nothing, whatever the loss curve shows.
  echo
  echo '--- last reward batches ---'
  ssh_ "tr '\r' '\n' < $log | grep -a 'reward mean reward' | tail -4 | cut -c1-100" || echo '  no GRPO steps yet'

  # 4. Where the run is, and anything that killed it.
  echo
  echo '--- progress ---'
  ssh_ "tr '\r' '\n' < $log | grep -aE '=== stage|-- cycle|supervised pass|distil:|advancing early|STALLED|ledger:|^done\.' | tail -6"
  local err
  err=$(ssh_ "grep -aE '^Traceback|OutOfMemoryError|CUDA out of memory|TorchRuntimeError|FORMAT MISMATCH' $log | tail -3")
  [[ -n "$err" ]] && { echo; echo '--- ERRORS ---'; echo "$err" | cut -c1-140; }
  return 0
}

if [[ $FOLLOW == 0 ]]; then status; exit $?; fi

while true; do
  status || exit 1
  ssh_ "grep -qa '^done\.' $REMOTE/\$(cd $REMOTE && ls -td runs/planb-* runs/grpo-* | head -1)/train.log" && {
    echo; echo 'run finished. pull results with: bash scripts/pull-results.sh'; exit 0; }
  sleep 60
done
