#!/usr/bin/env bash
# The training run: expert iteration, then GRPO under dynamic sampling.
#
#     bash scripts/run.sh                      # defaults, detached, logged
#     MODEL=unsloth/Qwen3-8B FOURBIT=1 bash scripts/run.sh
#     DRY=1 bash scripts/run.sh                # print the command, run nothing
#
# Detached under `nohup` on purpose. A dropped SSH session should not end a
# rental that has already been paid for; reattach with `tail -f`.

set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"

MODEL="${MODEL:-unsloth/Qwen3-4B-Instruct-2507}"
OUT="${OUT:-runs/grpo-$(date +%Y%m%d-%H%M%S)}"
# Lean grading is the step's wall clock and it is CPU-bound. One worker per
# core; the workers are threads blocked in `subprocess.run`, so they do not
# contend for the GIL and oversubscribing buys nothing.
# `nproc` is coreutils and absent on macOS, where this script is only ever
# dry-run to check the flags.
cores() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16; }
WORKERS="${WORKERS:-$(cores)}"
FOURBIT="${FOURBIT:-0}"
SPLIT="${SPLIT:-binops}"

# A quantised 8B sharing one 24 GB card with a colocated vLLM engine is tight
# in two different places, so the profile changes in two places. vLLM gets a
# smaller slice, because 4-bit weights are small but the KV cache for 16
# sequences of 9k tokens is not. And the training microbatch halves, with
# gradient accumulation doubled to keep the optimizer step identical: peak
# activation memory scales with the microbatch, the update does not.
if [[ "$FOURBIT" == "1" ]]; then
  GPUMEM="${GPUMEM:-0.50}"; BATCH="${BATCH:-4}"; ACCUM="${ACCUM:-4}"
else
  GPUMEM="${GPUMEM:-0.55}"; BATCH="${BATCH:-8}"; ACCUM="${ACCUM:-2}"
fi

ARGS=(
  --model "$MODEL"
  --out "$OUT"
  --workers "$WORKERS"
  --split "$SPLIT"

  # -- capacity -----------------------------------------------------------
  # 8k of prompt is what a volume-3 task at depth 3 needs once its supplied
  # lemmas are rendered; tasks over the limit are dropped rather than truncated,
  # because left truncation removes the definitions and keeps the goals.
  --max-seq 10240
  --max-prompt-tokens 8192
  --max-completion-tokens 768
  --lora-r 32
  --gpu-memory-utilization "$GPUMEM"

  # -- the GRPO step ------------------------------------------------------
  # batch x accum = 16 completions per optimizer step = 2 prompts at group 8.
  --group 8
  --batch "$BATCH"
  --accum "$ACCUM"
  --lr 1e-5
  --temperature 1.0

  # DAPO's two corrections that cost nothing. beta 0 drops the KL term, which
  # otherwise pulls the policy toward a base model that cannot do this task and
  # makes trl materialise a reference model this card cannot spare. The
  # asymmetric clip lets a rare tactic's probability rise; the tactics worth
  # learning here all start rare.
  --beta 0.0
  --epsilon 0.2
  --epsilon-high 0.28

  # Per-target advantages over per-target token spans. The single largest
  # effect available: at group 8 and volume 3 it yields 24 contrasts per group
  # instead of 8, and rescues groups whose totals match but whose per-target
  # outcomes differ.
  --factored on

  # -- cold start ---------------------------------------------------------
  # Without this GRPO begins at a pass rate low enough that every group is flat
  # and every update is zero, which is what the first measured run did.
  --ei-rounds "${EI_ROUNDS:-3}"
  --ei-tasks "${EI_TASKS:-192}"
  --ei-samples 8
  --ei-depths 1 2
  --ei-volume 3
  --ei-temperature 1.2
  --ei-sft-steps "${EI_SFT_STEPS:-150}"
  --ei-sft-lr 1e-5
  --ei-stop-score 0.55

  # -- dynamic sampling ---------------------------------------------------
  --refresh-every 40
  --oversample 4
  --explore 0.35
  --on-stall rescue
  --stall-retries 1

  # -- banking ------------------------------------------------------------
  --distil on
  --distil-steps 80
  --distil-lr 1e-5

  # -- held-out measurement ----------------------------------------------
  # This is what the benchmark reports, and its sample size is what decides
  # whether the result says anything. 24 tasks at group 4 puts a 95% interval
  # of [0.000, 0.793] around a zero, which is compatible with almost any claim.
  # Raise both for a run whose numbers are meant to be quoted; the cost is
  # generation, and Lean grading of the eval set is minutes on a many-core host.
  --eval-tasks "${EVAL_TASKS:-24}"
  --eval-group "${EVAL_GROUP:-4}"
)

# A longer curriculum than DEFAULT_CURRICULUM, for a run with the budget for it.
[[ -n "${STAGES:-}" ]] && ARGS+=(--stages "$STAGES")

[[ "$FOURBIT" == "1" ]] && ARGS+=(--load-in-4bit)

mkdir -p "$OUT"
LOG="$OUT/train.log"

if [[ "${DRY:-0}" == "1" ]]; then
  printf 'uv run python -m pdd.train_grpo'
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

echo "model   $MODEL"
echo "out     $OUT"
echo "workers $WORKERS"
echo "log     $LOG"

nohup uv run python -m pdd.train_grpo "${ARGS[@]}" >"$LOG" 2>&1 &
echo "pid     $!"
echo
echo "follow with:  tail -f $LOG"
echo "watch signal: grep -E 'degenerate|rescued|ledger' $LOG | tail -20"
