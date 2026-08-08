#!/usr/bin/env bash
# Copy a run's results off the rented host. Run this locally.
#
#     bash scripts/pull-results.sh                      # newest run
#     HOST=ssh7.vast.ai PORT=24866 bash scripts/pull-results.sh
#
# Everything worth keeping is small: the reward log, the held-out evaluations,
# the mined proof corpus and the LoRA adapters. The training log is the only
# large file and is fetched truncated, since most of its bulk is vLLM progress
# bars rather than anything anyone will read.
#
# Worth running before the rental ends rather than after. A vast.ai instance
# whose credit runs out is stopped, and a stopped instance is not a durable
# archive.

# No `-e`: every step here is a best-effort fetch of a file that may not
# exist yet, and one missing artefact must not abandon the rest.
set -uo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-ssh7.vast.ai}"
PORT="${PORT:-24866}"
USER_AT="${USER_AT:-root}"
REMOTE="${REMOTE:-/root/pdd}"
DEST="${DEST:-results/remote}"

ssh_() { ssh -o BatchMode=yes -o ConnectTimeout=25 -p "$PORT" "$USER_AT@$HOST" "$@"; }

# Strip the carriage return ssh leaves behind and the trailing slash from
# `ls -d`, but not the separator inside the path.
RUN="${RUN:-$(ssh_ "cd $REMOTE && ls -td runs/grpo-2026*/ | head -1" | tr -d '\r' | sed 's#/$##')}"
[[ -n "$RUN" ]] || { echo "no run directory found on $HOST" >&2; exit 1; }

OUT="$DEST/$(basename "$RUN")"
mkdir -p "$OUT"
echo "run    $RUN"
echo "dest   $OUT"

# The structured results: reward stream, per-stage held-out tables, the
# bootstrap summary, the ledger, and the exact arguments the run was given.
for f in log.json; do
  ssh_ "cat $REMOTE/$RUN/$f" > "$OUT/$f" 2>/dev/null && echo "  got $f" || echo "  no $f yet"
done

# Per-stage artefacts. Adapters are a few tens of MB each; the verified corpus
# is the kernel-accepted proof set, which is the reusable output of the run.
ssh_ "cd $REMOTE/$RUN 2>/dev/null && ls -d stage*/ bootstrap/ 2>/dev/null" | tr -d '\r' | while read -r d; do
  [[ -n "$d" ]] || continue
  mkdir -p "$OUT/$d"
  for f in verified.jsonl log.json; do
    ssh_ "cat $REMOTE/$RUN/$d$f" > "$OUT/$d$f" 2>/dev/null && echo "  got $d$f" || rm -f "$OUT/$d$f"
  done
done

# The readable part of the training log: everything that is not a progress bar.
ssh_ "cd $REMOTE && tr '\r' '\n' < $RUN/train.log | grep -avE 'Processed prompts|Rendering prompts|Adding requests|Capturing CUDA|^INFO ' | tail -4000" \
  > "$OUT/train.trimmed.log" 2>/dev/null || true
[[ -s "$OUT/train.trimmed.log" ]] && echo "  got train.trimmed.log"

# The held-out tables, pulled out on their own because they are the result.
grep -aB2 -A8 'depth  tasks' "$OUT/train.trimmed.log" > "$OUT/held-out-tables.txt" 2>/dev/null || true

# Drop placeholders for artefacts the run has not produced yet, so the
# directory only ever contains things that actually exist.
find "$OUT" -type f -empty -delete 2>/dev/null || true

echo
echo "held-out evaluations so far:"
cat "$OUT/held-out-tables.txt" 2>/dev/null || echo "  none yet"
du -sh "$OUT"
