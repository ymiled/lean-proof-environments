# Design log

What was tried, what broke, and what the failure forced. Kept because the
negative results are the part that is hard to reproduce, and because anyone
building a similar environment will hit the same three walls.

## Attempt 1 — hyperoperation tower, lemmas at a concrete point

Define a unary numeral type and a tower of operations, each in terms of the
last:

```lean
def op1 x .zero = x            -- addition
def op1 x (.succ y) = .succ (op1 x y)
def op2 _ .zero = .zero        -- multiplication
def op2 x (.succ y) = op1 x (op2 x y)
def op3 _ .zero = one          -- exponentiation
def op3 x (.succ y) = op2 x (op3 x y)
```

Ladder: `op1 one y = .succ y`, then `op2 one y = y`, then `op3 one y = one`,
each proved by induction citing the one below.

All four rungs compiled. The chain looked real: removing `L2` from the simp set
left the residual goal `op2 one one = one`, exactly the missing lemma.

**It was not real.** With `one` added to the unfolding set, the whole thing
collapsed:

```lean
theorem L3_alone (y : Tally) : op3 one y = one := by
  induction y with
  | zero => rfl
  | succ k ih => simp_all [one, op1, op2, op3]
-- compiles, with zero lemmas in scope
```

**Diagnosis.** Every lemma was stated at the concrete point `x := one`, so the
residual goals were *closed terms*. `simp` never had to reason about them; it
evaluated them. Dependency depth was an artifact of which lemmas we happened to
hand `simp`, not a property of the mathematics.

**Rule extracted.** Lemmas must be universally quantified over the variables the
definitions recurse on. Then residual goals contain free variables, reduction is
blocked, and induction is genuinely required.

Confirmed by re-testing with `x` open:

```lean
theorem L2_alone (x y : Tally) : op2 (.succ x) y = op1 (op2 x y) y := by
  induction y with
  | zero => rfl
  | succ k ih => simp_all [op1, op2]
-- error: unsolved goals
--   ⊢ op1 x.succ (op1 (op2 x k) k) = (op1 (op1 x (op2 x k)) k).succ
```

Automation cannot close it. The residual needs both left-successor and
associativity of `op1` — a real dependency.

## Attempt 2 — hyperoperations with open variables

Keeps the fix, but does not extend. The level-2 rung `(x+1)*y = x*y + y` is
fine. The level-3 analogue would be a closed form for $(x+1)^y$, which does not
exist. The tower gives at most two rungs of genuine depth.

**Abandoned.** Depth 2 is not enough to fit a decay curve.

## Attempt 3 — the Peano development (shipped)

The standard chain, which is deep precisely because each step needs the previous
ones:

```
add_zero_l ─┐
add_succ_l ─┴→ add_comm ─┐
add_assoc ───────────────┴→ add_left_comm ─→ mul_succ_l ─┐
mul_zero_l ──────────────────────────────────────────────┴→ mul_comm
```

11 rungs, depths 1–5. Every statement open in its recursion variables, so
Attempt 1's collapse cannot recur.

One rung needed care. `mul_succ_l` left the residual

```
⊢ op1 k (op1 (op2 x k) x) = op1 x (op1 (op2 x k) k)
```

which is an add-rearrangement. It closes only once commutativity is supplied as
a *permutative* rewrite (`simp` orders terms rather than looping). This is why
`add_left_comm` exists as its own rung: it is the lemma that turns out to be
load-bearing, and it is not the one you would guess from the statement.

## The `sorryAx` incident

Worth recording, because it is the strongest argument for the grader design.

During Attempt 3, `L07` failed to prove. Lean reported the error — but still
*declared* the theorem. Downstream lemmas citing it compiled with exit status 0
and no diagnostics of their own. Only `#print axioms` revealed the problem:

```
'L07' depends on axioms: [propext, sorryAx]
```

A grader checking compilation alone would have marked that entire subtree as
proved. Hence: the axiom audit, not the exit code, is the acceptance test.

## Rejected: statement comparison

An earlier grader design compared the model's theorem statement against the
requested one up to definitional equality, to catch a model proving something
weaker.

Dropped in favour of never letting the model write the statement at all. The
harness emits the `theorem ... := by` header and splices in only the tactic
block. This makes statement drift structurally impossible rather than
detectable, removes a whole class of grader bugs, and needs no code.

## Open issues

- **Depth caps at 5.** Extending to 7–8 means adding `pow_mul` and friends;
  each new rung needs its reference proof verified by hand first.
- **Contamination is mitigated, not solved.** Renaming blocks verbatim recall of
  library proof terms but not strategy transfer. A genuinely novel theory —
  randomly generated algebraic axioms — would be stronger, at the cost of much
  harder reference-proof generation.
- **Episodes are single-step.** Per-tactic actions with intermediate proof
  states need a persistent Lean server (`lean --server`) rather than one-shot
  `lean file.lean`.
- **`pass@k` pools rungs at equal depth.** With only 2–4 rungs per depth this is
  noisy; more rungs per level would tighten the fit.

## Attempt 4 — noninterference family, and a refuted hypothesis

Built a second family: a security type system for a loop-free imperative
language, with the soundness theorem (noninterference) as the top rung. Seven
rungs, depths 1–3, self-test green.

Three choices made it tractable without Mathlib. Semantics is a **total
function** rather than an inductive relation, so rungs go by structural
induction plus `simp` instead of induction over derivation trees. Typing is
**`Bool`-valued** rather than an inductive judgment, so `wt G pc c = true` can
be taken apart by `simp`. And the language is **loop-free**, since `while` forces
termination reasoning that adds nothing to the noninterference argument.

**The stated motivation was wrong.** The family was built to decorrelate depth
from proof length, on the reasoning that its rungs vary from 1 to 37 lines while
arithmetic's are uniformly ~3. Measured correlation between depth and
compositional reference-proof length:

| family | corr(depth, refLoC) | corr(depth, ndeps) |
|---|---|---|
| arithmetic | **−0.13** | +0.97 |
| noninterference | **+0.97** | **+1.00** |

Exactly backwards. Arithmetic is the family where compositional proof length is
independent of depth. In the noninterference ladder the deep rungs are deep
*because* they do more case analysis, so depth and length grow together, and
ancestor count tracks depth perfectly (r = 1.000).

**What the measurement actually shows.** Correlation between depth and
*monolithic* length is ~0.98 in both families and is not fixable by family
choice: monolithic length is the sum over ancestors, so it grows with depth by
construction. Decorrelating requires contrasting rungs of **equal depth but
differing ancestor count**. That lever exists inside `arithmetic` — at depth 3,
`add_left_comm` has 4 ancestors and `mul_assoc` has 2; at depth 4, 5 versus 4 —
and does not exist here at all.

So the identification strategy is a within-depth contrast on ancestor count, not
a between-family contrast. That is a different experiment from the one
originally planned, and a better-specified one.

**What the family is still worth.** Domain diversity — whether the decay
constant is a property of proving-at-depth or of one corpus needs more than one
theory. And genuine difficulty at the top: the final rung is a real metatheorem
with a 37-line proof, where depth and intrinsic difficulty are entangled in a
way they are not in arithmetic.

## Result 1 — chain versus antichain, Sonnet 5, arithmetic family

First measurement taken. 24 tasks, 3 renamed instances, 2 samples each, one-shot
with no compiler access to the policy.

| k | arm | residual depth | reference LoC | pass |
|---|---|---|---|---|
| 2 | antichain | 1 | 6 | 1.00 |
| 2 | chain | 2 | 6 | 1.00 |
| 3 | antichain | 1 | 9 | 1.00 |
| 3 | chain | 3 | 9 | 1.00 |

**Ceiling effect. The contrast is uninformative at this difficulty.** Both arms
saturate, so the design cannot discriminate chaining from volume here — not
because the hypothesis is wrong, but because the policy is not stressed.

The grader was checked rather than assumed: on the same tasks, a garbage block
(`rfl` everywhere) returns `compile_error` and the reference returns `proved`.
So 1.00 reflects the policy, not a permissive grader.

**Structural cap on the design.** In this family, up-closed *chains* run out at
k = 3, while up-closed antichains reach k = 6. A chain of 4 would need four
totally-ordered rungs whose dependents are all inside the withheld set, and the
DAG does not admit one. So the contrast cannot be pushed to larger k by adding
seeds or samples; it needs either a deeper family or a weaker policy.

Recorded because the negative result constrains the design: any future version
of this experiment must either raise per-rung difficulty or accept that the
chain/antichain contrast only has power against policies that fail somewhere in
k ∈ {2, 3}.
