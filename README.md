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

Two theories ship, selected with `--family`:

| family | rungs | depths | top rung |
|---|---|---|---|
| `arithmetic` | 11 | 1–5 | commutativity of multiplication |
| `noninterference` | 7 | 1–3 | soundness of a security type system |

`noninterference` is a loop-free imperative language with a two-point security
lattice; the top rung proves that a well-typed command run from two states
agreeing on public variables yields two states that still agree — no secret
flows to a public observer. Semantics is a total function and typing is
`Bool`-valued, which keeps every proof inside structural induction plus `simp`
with no Mathlib.

## Dependency depth

`depth = 1` for a lemma provable from the definitions alone; otherwise
`1 + max(depth of its dependencies)`. The shipped ladder runs 1 to 5. Writing
`op1` as `+` and `op2` as `*` (both are randomly renamed per instance):

| depth | lemma | statement | needs |
|---|---|---|---|
| 1 | `add_zero_l` | `0 + y = y` | — |
| 1 | `add_succ_l` | `(x+1) + y = (x+y)+1` | — |
| 1 | `add_assoc` | `(x+y)+z = x+(y+z)` | — |
| 2 | `add_comm` | `x + y = y + x` | `add_zero_l`, `add_succ_l` |
| 3 | `add_left_comm` | `x+(y+z) = y+(x+z)` | `add_assoc`, `add_comm` |
| 4 | `mul_succ_l` | `(x+1)*y = x*y + y` | `add_left_comm`, … |
| 5 | `mul_comm` | `x * y = y * x` | `mul_zero_l`, `mul_succ_l` |

The operations recurse on their *second* argument, so the left-handed statements
require induction rather than falling to computation. That asymmetry is what
generates the chain.

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

Both theories are textbook material every frontier model has seen. The defence
is renaming: types, constructors, operators and lemma names are freshly sampled
per seed (`Warp`, `lem_a7f2`, …), so a model cannot pattern-match the goal onto
a remembered `Nat.mul_comm` proof term.

This weakens the confound; it does not eliminate it. A model that knows the
Peano development, or knows Volpano–Smith–Irvine, can still transfer the proof
strategy — and arguably should, since that is legitimate proving. What renaming
removes is verbatim library recall, not knowledge. **The renaming defence is
itself untested**; comparing pass rates on renamed versus native `Nat`
statements would measure it, and has not been run.

## Usage

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh   # Lean 4 toolchain
uv sync

uv run python -m pdd.selftest --family noninterference   # certify the benchmark
uv run python -m pdd.sweep --policy reference --family arithmetic
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
    arith.py            Peano development, 11 rungs, depths 1-5
    noninterference.py  security type system, 7 rungs, depths 1-3
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
| `arithmetic` | −0.13 | +0.97 |
| `noninterference` | +0.97 | +1.00 |

Correlation between depth and *monolithic* length is ~0.98 in both, and no
choice of family fixes that: monolithic length is the sum over ancestors, so it
grows with depth by construction.

This matters because Theorem's own post reports a difficulty cliff at "17+
marginal LoC." In the arithmetic ladder, monolithic length crosses 17 right at
depth 3→4 — **their length cliff predicts our curve without any depth effect at
all.** A raw pass@k-vs-depth plot cannot distinguish the two.

The identification strategy is therefore a **within-depth contrast on ancestor
count**, not a depth sweep. That lever exists in `arithmetic` (at depth 3, one
rung has 4 ancestors and another has 2; at depth 4, 5 versus 4) and not in
`noninterference`, where ancestor count tracks depth at r = 1.000.

`docs/design-log.md` records this in full, including the fact that the
noninterference family was built expecting the opposite result.

## Status

Two families, self-test green in both conditions. Model sweeps pending —
no policy has been run yet, so this repository currently contains a calibrated
instrument and no measurement.
