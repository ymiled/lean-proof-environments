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
