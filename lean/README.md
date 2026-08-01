# `Vsi.lean` — port of a security type system to Lean 4

A formalisation of the type system from
[`ymiled/type-system-for-noninterference`](https://github.com/ymiled/type-system-for-noninterference),
originally implemented in OCaml with explicit proof trees and a hand-written
checker. The point of the port is to replace that checker with Lean's kernel.

This is **separate from the benchmark**. The `noninterference` family in
`src/pdd/families/` is a deliberately simplified system built for measurement;
this file targets the real one.

## What the original system has that the simplified family does not

| feature | `families/noninterference.py` | `Vsi.lean` |
|---|---|---|
| lattice | `Bool`, two points | `Lvl` with `le`, subtyping |
| command types | none, `pc`-indexed judgment | `Cmd(τ1,τ2)`, `Ncmd(τ,n)` |
| subsumption rule | no | yes |
| `while` | no | yes |
| functions | no | yes |
| typing | `Bool`-valued function | inductive relation |

`Ncmd(τ, n)` carries a **nesting-depth counter**: the conditional rule requires
its branches at `Ncmd(τ, n-1)`. Functions are the original author's extension
over the published Volpano–Smith–Irvine rules.

## Status

Compiles clean. `#print axioms` reports `propext` only, no `sorryAx`.

**Done**

- Syntax: `Lvl`, `Ty`, `Exp`, `Com`, `Fn`, and the environments.
- Semantics: `evalE` / `evalC`, mutually recursive and fuel-indexed, covering
  `while` and function calls. Fuel is what makes `while` total; the theorem this
  supports is therefore *termination-insensitive*, relating runs that both
  finish.
- `SubTy`: all seven subtyping rules from `check_sub_rules`, including
  contravariance of `Cmd` in its first component, plus transitivity.
- `TyE` / `TyC`: the expression and command typing rules from `check_type`,
  each with a subsumption constructor.
- `lowEq` and its reflexivity, symmetry, transitivity.
- `Lvl.le_refl`, `Lvl.le_trans`.
- `sub_base_inv`: only a base type subtypes into a base type, and the levels are
  ordered. This is the lemma that makes the rest tractable. Without it every
  inversion has to treat `sub` as an open case, since a typing derivation can
  end in subsumption at any point.

**Not done**

- Inversion lemmas for `TyE` at `base L`.
- Expression agreement: public expressions evaluate alike in agreeing states.
  Harder here than in the simplified family, because `binop` types both operands
  at the *same* level and function calls can appear inside expressions.
- Confinement for `Cmd(H, H)`.
- The soundness theorem. The `while` case needs induction on fuel rather than on
  the command, and the function-call case needs the argument-level restriction
  (`FuncNonVoidCallDeriv` only admits `τ1 = L`) to carry the agreement across a
  call.

Honest estimate for the remainder: a full focused day at least, most of it in
the `while` and function cases.
