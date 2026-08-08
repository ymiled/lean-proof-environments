#!/usr/bin/env bash
# The benchmark-grade configuration: longer curriculum, larger held-out sample.
#
#     bash scripts/full-run.sh
#
# `run.sh` defaults are sized to answer "does the cold start work" for about
# two dollars. This is sized to produce numbers worth quoting, and the
# difference is mostly not the training.
#
# The recorded run reported pass@1 of 0.021 at a 95% interval of
# [0.000, 0.793] per depth. An interval that wide is compatible with the policy
# having learned nothing and with it having tripled, so the training curve was
# never the limiting factor -- the evaluation sample was. 48 tasks per depth at
# group 8 is 8x the rollouts and brings the interval down to something that can
# separate those two hypotheses.
#
# Budget at $0.37/h for a quantised 8B: roughly 14 hours, roughly $5.
#
#   expert iteration, 4 rounds        ~1.7h
#   GRPO, up to 1000 steps            ~11h
#   held-out evaluation, 4 stages     ~1.3h
#
# Stages end early when the trailing reward clears `advance_reward`, so the
# GRPO figure is a ceiling rather than a plan.

set -euo pipefail
cd "$(dirname "$0")/.."

# Not a hybrid reasoning model, and that is the reason for the choice rather
# than size. `Qwen3-8B` is Qwen3's hybrid: its chat template takes an
# `enable_thinking` flag which defaults to on, so with `add_generation_prompt`
# and nothing disabling it the model opens a <think> block. At a 768-token
# completion budget it never leaves that block, emits no `-- PROOF` marker, and
# every response grades as unparseable. Measured: 2 proved out of 6144 target
# attempts, a mean score of 0.000 where the base rate is 0.052.
#
# The failure is silent in every number a training log prints, because a policy
# that cannot format looks exactly like a policy that cannot prove. `-2507`
# instruct checkpoints have no thinking mode and need no such handling. Before
# substituting a model here, check `enable_thinking` in its chat template; if it
# is present, either raise --max-completion-tokens well past the reasoning
# budget or disable thinking on both the training and sampling paths.
export MODEL="${MODEL:-unsloth/Qwen3-4B-Instruct-2507}"
export FOURBIT="${FOURBIT:-0}"
export SPLIT="${SPLIT:-binops}"

# Deeper cold start. Every extra round is more kernel-verified data for the
# supervised pass, and the round is skipped automatically once the mean
# per-target score clears --ei-stop-score.
export EI_ROUNDS=4
export EI_TASKS=256
export EI_SFT_STEPS=200

# The measurement, widened. See above.
export EVAL_TASKS=48
export EVAL_GROUP=8

# Longer than DEFAULT_CURRICULUM, and reaching depth 4 with a wider withheld
# set. Volume stays at 3 or more throughout: a single-target task scores in
# {0, 1} however it is graded, which is the degenerate case the shaped reward
# exists to avoid. The last stage drops the depth constraint entirely.
export STAGES='[
  {"depths": [1, 2], "volume": 3, "steps": 200, "prompts": 512},
  {"depths": [2, 3], "volume": 3, "steps": 250, "prompts": 512},
  {"depths": [3, 4], "volume": 4, "steps": 275, "prompts": 512},
  {"volume": 4, "steps": 275, "prompts": 512}
]'

exec bash scripts/run.sh
