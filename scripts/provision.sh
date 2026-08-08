#!/usr/bin/env bash
# Provision a rented GPU host for `pdd.train_grpo`. Run this on the instance.
#
#     bash scripts/provision.sh          # after cloning
#     curl -sL <raw-url>/provision.sh | bash -s -- --clone
#
# The environment has two halves and they fail differently. The Python half
# fails loudly: a missing wheel is an ImportError before training starts. The
# Lean half fails silently, and that is what most of this script is about. A
# host with the wrong toolchain, or with `lean` absent from a worker's PATH,
# reports every candidate proof as a compile error -- which is exactly what a
# wrong proof looks like. Training then runs to completion, costs the full
# rental, and produces a flat reward curve with nothing in any log to say why.
#
# So the last step compiles reference proofs and refuses to exit 0 unless the
# kernel accepts them. Do not skip it.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ymiled/lean-proof-environments.git}"
WORKDIR="${WORKDIR:-$HOME/pdd}"
BRANCH="${BRANCH:-main}"
DO_CLONE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone)  DO_CLONE=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s ===\n' "$*"; }

say "system packages"
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null; then
  apt-get update -qq
  # curl for the elan and uv installers; git-lfs because model repos use it;
  # build-essential because some wheels still compile on install.
  apt-get install -y -qq --no-install-recommends \
    ca-certificates curl git git-lfs build-essential tmux jq >/dev/null
fi

say "elan and the pinned Lean toolchain"
if ! command -v elan >/dev/null && [[ ! -x "$HOME/.elan/bin/elan" ]]; then
  curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"

if [[ $DO_CLONE == 1 ]]; then
  say "clone $BRANCH"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# `pdd.grader` passes ELAN_TOOLCHAIN explicitly on every call, because it
# compiles in a temporary directory where elan would never find this file.
# Installing it up front turns a per-call download into a one-off.
TOOLCHAIN="$(cat lean-toolchain)"
say "installing $TOOLCHAIN"
elan toolchain install "$TOOLCHAIN"
elan default "$TOOLCHAIN"
lean --version

say "uv and the training extras"
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra train

say "preflight: environment machinery"
# Neither Lean nor a GPU. Catches a broken dynamic-sampling or curriculum
# install in two seconds rather than two hours in.
uv run python -m pdd.selftest --only-dynamic

say "preflight: the kernel actually accepts reference proofs"
# The one that matters. `--skip-pools` drops the renaming-collision scan, which
# is a property of the corpus rather than of this host and was checked when the
# corpus was built.
uv run python -m pdd.selftest --family vsi --skip-pools

say "preflight: GPU visible to torch"
uv run python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device'
name = torch.cuda.get_device_name(0)
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{name}, {gb:.1f} GB, torch {torch.__version__}')
assert gb > 20, f'{gb:.1f} GB is below what a 24 GB run assumes'
"

say "preflight: grading throughput"
# Lean grading is the wall clock of an RL step and it is CPU-bound, so the
# machine's core count matters more here than its GPU. Printed rather than
# asserted: what counts as too slow depends on the batch size you chose.
nproc
uv run python -m pdd.sanity --help >/dev/null 2>&1 || true

cat <<EOF

Provisioned. Cores: $(nproc). Start the run with:

    cd $WORKDIR && bash scripts/run.sh

EOF
