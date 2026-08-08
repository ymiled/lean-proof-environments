#!/usr/bin/env bash
# Try each candidate policy in turn, moving on when one cannot speak the format.
#
#     bash scripts/supervise.sh                    # detached, logged
#     CANDIDATES="a b" bash scripts/supervise.sh   # explicit chain
#
# The failure this exists for is specific and was paid for once already. A model
# whose completions never contain a `-- PROOF` marker scores exactly like a
# model that cannot prove: every reward is the unparseable constant, every
# rollout group is flat, and every metric in the training log reads as weakness
# rather than as a mismatch. `pdd.bootstrap` detects it after the first expert
# iteration round and exits 17; this script treats that status, and only that
# status, as "try the next candidate".
#
# Any other non-zero status stops the chain. A crash is not a reason to spend
# the budget on a different model, and a stall is a training outcome the run
# already knows how to handle on its own.

set -uo pipefail
cd "$(dirname "$0")/.."

FORMAT_MISMATCH_EXIT=17

# Order is deliberate: a Lean-specialised prover first, since the measured
# obstacle is Lean ability rather than scale or formatting, and a general
# instruct model second, since it is known to emit the format correctly and so
# is the safe floor rather than the hopeful ceiling.
CANDIDATES="${CANDIDATES:-deepseek-ai/DeepSeek-Prover-V2-7B unsloth/Qwen3-4B-Instruct-2507}"
SUPLOG="${SUPLOG:-runs/supervise-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$SUPLOG")"

run_chain() {
  for model in $CANDIDATES; do
    echo "=== candidate: $model ==="
    # 4-bit for anything at 7B or above; a 24 GB card also holds a colocated
    # vLLM engine and the optimizer state, and bf16 weights at that size do not
    # leave room for both.
    case "$model" in
      *4B*|*3B*|*1.5B*) fourbit=0 ;;
      *)                fourbit=1 ;;
    esac

    MODEL="$model" FOURBIT="$fourbit" FOREGROUND=1 bash scripts/full-run.sh
    status=$?

    if [[ $status -eq 0 ]]; then
      echo "=== $model completed the run ==="
      return 0
    fi
    if [[ $status -eq $FORMAT_MISMATCH_EXIT ]]; then
      echo "=== $model cannot emit the output format; trying the next candidate ==="
      continue
    fi
    echo "=== $model exited $status, which is not a format mismatch; stopping ==="
    return "$status"
  done
  echo "=== every candidate failed the format check ==="
  return "$FORMAT_MISMATCH_EXIT"
}

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  run_chain
  exit $?
fi

nohup bash -c "$(declare -f run_chain); CANDIDATES='$CANDIDATES' \
  FORMAT_MISMATCH_EXIT=$FORMAT_MISMATCH_EXIT; cd '$PWD'; run_chain" \
  >"$SUPLOG" 2>&1 &
echo "supervisor pid $!"
echo "candidates     $CANDIDATES"
echo "log            $SUPLOG"
echo
echo "follow with:   tail -f $SUPLOG"
