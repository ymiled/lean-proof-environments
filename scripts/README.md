# Running the training on a rented GPU

The environment runs on CPU; only `pdd.train_grpo` needs CUDA. This is the
procedure for renting one card, provisioning it, and starting a run that
survives a dropped connection.

## Picking the instance

The obvious filter is the wrong one. A 4090 is a 4090, and every offer has the
same 24 GB, so the GPU column does not discriminate between offers at all. What
discriminates is the **CPU**, because the wall clock of a training step is not
generation, it is Lean.

One optimizer step at the settings in `scripts/run.sh` is 16 completions of a
volume-3 task, so up to 48 `lean` invocations, each around 1.2s when the proof
fails and 2.8s when it checks. Deduplication in `rollout.Grader` removes perhaps
40% of them early in training. That leaves roughly 30 sequential seconds of Lean
per step, divided by however many cores the host has. Generation of the same
batch is 10 to 20 seconds. So:

| cores | Lean per step | step time | 700 steps |
| --- | --- | --- | --- |
| 8 | ~34s | ~50s | ~10h |
| 16 | ~17s | ~32s | ~6h |
| 32 | ~9s | ~24s | ~5h |
| 64 | ~4s | ~20s | ~4h |

Past about 32 cores generation dominates and more cores stop paying. Below 16
you are renting a 4090 to watch it idle.

Filters worth setting, in order of how much they matter:

| criterion | value | why |
| --- | --- | --- |
| `cpu_cores_effective` | >= 24 | the actual bottleneck, see above |
| `cpu_ram` | >= 48 GB | dozens of concurrent `lean` processes, each with its own environment |
| `disk_space` | >= 120 GB | torch, vllm and unsloth wheels are ~25 GB before any weights |
| `inet_down` | >= 500 Mbps | those wheels plus the model are an hour of setup on a slow link |
| `reliability` | >= 0.985 | an interrupted run loses the adapter since the last stage boundary |
| `duration` | >= 3 days | some offers expire mid-run |
| `cuda_vers` | >= 12.4 | vllm's wheels |

The `vastai` CLI applies all of them at once, which the web filter cannot:

```sh
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=24 cpu_ram>=48 \
   disk_space>=120 inet_down>=500 reliability>=0.985 cuda_vers>=12.4 \
   rentable=true' -o 'dph+'
```

Rank by `$/h ÷ cores` rather than by `$/h`. A $0.34/h host with 64 cores
finishes the run for less money than a $0.28/h host with 24, because it finishes
sooner.

## Launching

```sh
vastai set api-key <key>
vastai create instance <offer-id> \
  --image nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  --disk 120 --ssh --direct --onstart-cmd 'sleep infinity'
vastai ssh-url <instance-id>
```

A bare CUDA image rather than a PyTorch one is deliberate. `uv sync --extra
train` installs the torch that vllm and unsloth agree on; a preinstalled torch
in the image is a second, different one, and which of the two wins is decided by
import order.

## Provisioning

```sh
curl -sL https://raw.githubusercontent.com/ymiled/lean-proof-environments/main/scripts/provision.sh \
  | bash -s -- --clone
```

This installs elan and the toolchain pinned in `lean-toolchain`, syncs the
Python environment, and then runs three preflights. The second one is the
reason the script exists:

```
uv run python -m pdd.selftest --family vsi --skip-pools
```

It compiles reference proofs and requires the kernel to accept them. A host with
a different Lean scores every candidate proof as a compile error, which is
indistinguishable from the policy being wrong: the run completes, costs the full
rental, and produces a flat reward curve with nothing in the log explaining it.
Do not start training if this step fails.

## Starting the run

```sh
tmux new -s pdd
cd ~/pdd && bash scripts/run.sh
```

`run.sh` detaches under `nohup` and writes to `runs/grpo-<timestamp>/train.log`,
so a dropped SSH session does not end a rental you have already paid for.

Knobs worth changing:

```sh
MODEL=unsloth/Qwen3-8B FOURBIT=1 bash scripts/run.sh   # bigger, quantised
SPLIT=theory bash scripts/run.sh                       # the transfer claim
DRY=1 bash scripts/run.sh                              # print, run nothing
```

`SPLIT` decides what a held-out number is evidence for. `binops` holds out
`ifc-b3*`, which shares every lemma key and ten of eleven reference proofs with
the trained `ifc-b1*`: passing it shows robustness to one more induction case,
and nothing about transfer. `theory` holds out `vsi`, whose definitions and
proofs are shared with nothing in the corpus. Quote `theory` for a
generalization claim, and expect the numbers to be lower.

## What to watch

```sh
tail -f runs/grpo-*/train.log
grep -E 'degenerate|rescued|ledger|dataset' runs/grpo-*/train.log | tail -30
```

Three lines carry the signal, in descending order of importance.

**`degenerate`** is the fraction of rollout groups whose rewards are all equal.
Those groups contribute exactly zero to the update whatever the loss curve says,
so this is the honest measure of whether the run is training. It should fall
through the expert-iteration stage and stay well below 0.9 afterwards. Above
0.97 for a sustained window the stage aborts itself and the rescue path runs an
expert-iteration round rather than burning the rental on zero gradients.

**`factored rescued N/M flat groups`** is the per-target advantage doing its
job: groups whose totals matched but whose per-target outcomes did not, which a
scalar advantage would have discarded. If this is persistently 0 while
`degenerate` is high, the groups are genuinely flat and the problem is
difficulty, not credit assignment.

**`dataset:`** and **`ledger:`** are dynamic sampling. `retired` climbing means
prompts observed to teach nothing are being replaced. If `retired` stays at 0
across many cycles, either the pool is all-unknown (early, expected) or
`--refresh-every` is longer than the run.

Held-out tables print after each stage, by depth. That is the number the
benchmark reports and the only one that separates learning to prove from
memorising the corpus.

## Cost

At $0.30 to $0.40 per hour, the configuration in `run.sh` is roughly:

- expert iteration, 3 rounds: 45 to 90 minutes
- four GRPO stages, up to 900 steps: 5 to 7 hours
- held-out evaluation after each stage: 20 minutes total

Call it 8 hours and under $4. Budget for two runs: the first one tells you
whether the cold start reached a pass rate the second can build on.
