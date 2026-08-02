# proof-depth-decay

A Lean 4 environment for measuring how model proof success decays with
**dependency depth**, and an independent test of a specific published claim.

Theorem's [`lf-lean`](https://theorem.dev/blog/lf-lean/) reports that when
translating proofs without compositional structure, success falls by "roughly a
3x reduction in successful problem count for every additional dependency
solved," and estimates that monolithic translation "would yield less than 20%
success." That decay constant is the justification for treating verified
dependencies as trusted interfaces rather than re-proving them, which is the
central architectural claim of the work.

It was measured once, on one corpus (*Logical Foundations*), with one pipeline.
This repository measures it again, on a different corpus, with a fully open
grader.

## Families

The benchmark covers information-flow security only. Two theories ship,
selected with `--family`:

| family | rungs | depths | maximal rungs | role |
|---|---|---|---|---|
| `vsi` | 14 | 1–5 | 2 | **evidence source.** Lemma chain of a 784-line machine-checked Volpano-Smith-Irvine development with subtyping, `while` loops and functions. Absent from public corpora. |
| `noninterference` | 12 | 1–4 | 3 | **synthetic control.** Toy IFC theory supporting per-seed renaming, which the real corpus cannot. Isolates the contamination variable. |

A Peano-arithmetic family was removed: it measured general proving ability
rather than IFC verification, and it is heavily represented in training data,
which is the confound this corpus exists to avoid. Its results are quarantined
in `results/archive/`.

The `maximal rungs` column is not decoration. Withheld sets must be up-closed,
so antichains can only be drawn from rungs nothing depends on. A family with one
maximal rung is a *funnel*: it supports depth sweeps but the chain/antichain
contrast is structurally impossible in it. `noninterference` originally had one
and was extended specifically to widen its top.

`noninterference` is a loop-free imperative language with a two-point security
lattice; the top rung proves that a well-typed command run from two states
agreeing on public variables yields two states that still agree — no secret
flows to a public observer. Semantics is a total function and typing is
`Bool`-valued, which keeps every proof inside structural induction plus `simp`
with no Mathlib.

## Dependency depth

`depth = 1` for a lemma provable from the definitions alone; otherwise
`1 + max(depth of its dependencies)`. Depth is derived from the graph, never
declared. The `vsi` ladder:

| depth | lemma | needs |
|---|---|---|
| 1 | `lowEq_refl`, `lowEq_symm`, `lowEq_trans` | — |
| 1 | `Lvl.le_refl`, `Lvl.le_trans`, `secure_sub` | — |
| 2 | `sub_base_inv` | `Lvl.le_refl`, `Lvl.le_trans` |
| 2 | `confinement` | `lowEq_refl`, `lowEq_trans`, `secure_sub` |
| 3 | `tyE_var_L`, `tyE_binop_L`, `tyE_call_L` | `sub_base_inv` |
| 4 | `evalE_agree_nocall` | `tyE_var_L`, `tyE_binop_L` |
| 4 | `agree` | `confinement`, the three `tyE_*`, `lowEq_symm`, `lowEq_trans` |
| 5 | `noninterference_full` | `agree` |

These are the dependencies the Lean kernel records, not ones chosen to produce a
gradient. `agree` is deep because expression agreement and command
noninterference must be proved simultaneously by induction on fuel: a call inside
an expression runs a command, and an assignment inside a command evaluates an
expression.

## The experiment

Each rung is posed to a policy under two conditions:

| Condition | Ancestors |
|---|---|
| `compositional` | supplied, already proved, citable by name |
| `monolithic` | absent; must be rediscovered and re-proved inline |

We measure pass@k against depth in both arms. If the 3x figure generalises,
monolithic success should fall roughly as $3^{-d}$ while compositional stays
near flat. Read the confound section below before believing any such curve.

## What is actually hard here

Not the plumbing. Three things, each of which killed an earlier version of the
benchmark:

**1. Goals must stay open in the recursion variable.** The first ladder stated
lemmas at a concrete point (`op_i one y = one`). Every such goal is a *closed
term*, so `simp` does not reason about it, it simply computes it. A depth-3
lemma fell to `simp_all [one, op1, op2, op3]` with no lower lemmas in scope at
all. Every rung here is universally quantified over the variables the
definitions recurse on, which blocks reduction and forces genuine induction.

**2. Depth must be verified, not assumed.** `selftest.py` checks that each
rung's citing proof *fails* when its ancestors are removed. A dependency that
survives removal is decorative and the rung measures nothing.

**3. Both arms must be solvable.** If monolithic tasks were simply impossible,
the decay curve would be a property of the benchmark rather than of the model.
The self-test certifies a known-good proof for every rung in *both* conditions,
inlining ancestors as `have`s for the monolithic arm.

## The grader

A verification reward signal is worth exactly as much as its grader, and Lean
offers several ways to close a goal without proving it.

This is not hypothetical. While building the ladder, one rung's proof failed;
Lean emitted a warning but still produced a usable declaration, and every
downstream lemma citing it compiled cleanly. `#print axioms` on a theorem three
rungs later reported `sorryAx`. **A grader that only asked "did it compile?"
would have scored that entire subtree as success.**

So acceptance requires all of:

1. compiles, exit 0, no `error:` diagnostics
2. `#print axioms` on the target ⊆ `{propext, Classical.choice, Quot.sound}` —
   the load-bearing check, catching `sorry`, `admit`, fresh axioms, and anything
   inherited transitively
3. no banned syntax (`native_decide` trusts the compiler, not the kernel)

Statement drift is prevented structurally rather than by checking: the harness
writes the `theorem ... := by` header itself and the policy supplies **only a
tactic block**. A model never gets to write the statement, so it cannot weaken,
restate, or substitute the goal.

## Contamination

The two families take opposite approaches, which is the point of having both.

`noninterference` is a toy theory, so every identifier is freshly sampled per
seed (`Warp`, `lem_a7f2`, …) and a model cannot pattern-match onto a remembered
proof term. `vsi` cannot be renamed without breaking its own Lean imports, and
does not need to be: the development does not exist in any public corpus.

Renaming removes verbatim recall, not knowledge. A model that knows
Volpano–Smith–Irvine can still transfer the strategy, and arguably should, since
that is legitimate proving. **The renaming defence is itself untested.** Running
matched tasks under renamed and canonical identifiers in `noninterference` would
measure it, and is the designed use of the control family.

## Usage

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # Lean 4 toolchain
uv sync

uv run python -m pdd.selftest --family noninterference   # certify the benchmark
uv run python -m pdd.sweep --policy reference --family vsi
uv run python -m pdd.sweep --policy claude-opus-5 --family noninterference --k 8 --seeds 5
uv run python -m pdd.plot results/noninterference-claude-opus-5@T1.0.json
```

`selftest` must report 0 failures before any sweep is trusted.

## Is this RL?

No, and the distinction matters. There is no training, no policy update, no
gradient. This is the **environment half** of an RL setup: state space, action
space, a task distribution with a difficulty knob, and a reward function that is
a proof kernel rather than a learned proxy. The policy is a frozen model.

`env.py` exposes `reset`/`step` so a learner could be attached, but attaching one
is not part of this repository.

Episodes are single-step: one attempt, one whole proof, one binary reward. A
multi-step version — one tactic per action, intermediate proof states as
observations — needs a persistent Lean server rather than one-shot compilation.
That is the natural next version.

## Layout

```
src/pdd/
  ladder.py    Rung/Family/Instance, depth derivation, renaming
  families/
    vsi.py              machine-checked VSI chain, 14 rungs, depths 1-5
    noninterference.py  synthetic IFC control, 12 rungs, depths 1-4
  task.py      monolithic / compositional rendering, prompts
  grader.py    compile + axiom audit + banned syntax
  selftest.py  benchmark certification (run this first)
  policy.py    reference / empty / Anthropic policies
  env.py       gym-style reset/step
  sweep.py     experiment driver
  plot.py      pass@k vs depth, decay fit
docs/
  design-log.md   what was tried, what broke, and why
```

## A confound you should know about before reading any curve

Depth is not cleanly separable from proof length in either family. Measured
correlations with compositional reference-proof length:

| family | corr(depth, refLoC) | corr(depth, ancestors) |
|---|---|---|
| `noninterference` | +0.56 | +1.00 |

`noninterference` began at +0.97 and was rebuilt down to +0.56 by adding rungs
that deliberately break the pattern: `wt_anti` is an 18-line proof at depth 1,
`ite_high_ni` a 3-line proof at depth 3. A shallow-long rung and a deep-short
rung are what make depth and length separable at all.

Correlation between depth and *monolithic* length is ~0.98 in both, and no
choice of family fixes that: monolithic length is the sum over ancestors, so it
grows with depth by construction.

This matters because a reported difficulty cliff at "17+ marginal LoC" is a
*length* effect, so on any ladder where length tracks depth, a length model with
no depth term predicts the same curve. A raw pass@k-vs-depth plot cannot
distinguish the two.

The identification strategy is therefore a **within-depth contrast on ancestor
count**, not a depth sweep. Verified levers in the surviving families:

| family | depth | contrast |
|---|---|---|
| `vsi` | 4 | `evalE_agree_nocall` (5 ancestors) vs `agree` (11) |
| `vsi` | 2 | `sub_base_inv` (2) vs `confinement` (3) |
| `noninterference` | 2 | `assign_ni` (1) vs `confinement` (4) |

The `vsi` depth-4 pair is the strongest contrast available: identical derived
depth, 2.2x spread in ancestor count, in the uncontaminated corpus. It has not
yet been run at usable sample size.

`docs/design-log.md` records this in full, including the fact that the
noninterference family was built expecting the opposite result.

## Status

Two families, self-test green in both conditions. Model sweeps pending —
no policy has been run yet, so this repository currently contains a calibrated
instrument and no measurement.
