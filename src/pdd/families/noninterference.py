"""Noninterference family: security type system for a small imperative language.

A loop-free imperative language with a two-point security lattice. The top rung
is the soundness theorem: a well-typed command run from two states that agree on
public variables produces two states that still agree on public variables. No
secret can flow to a public observer.

Three design choices make this tractable without Mathlib:

*   **Semantics is a total function, not an inductive relation.** `{evalC}` maps
    a state and a command to a state directly, so rungs are discharged by
    structural induction plus `simp` rather than by induction over derivation
    trees.
*   **Typing is `Bool`-valued, not an inductive judgment.** Well-typedness
    becomes computation, so a hypothesis `{wt} G pc c = true` can be taken apart
    by `simp` instead of by case analysis on constructors.
*   **The language is loop-free.** `while` would force termination reasoning
    that adds nothing to the noninterference argument.

Why this family exists alongside `arithmetic`, stated accurately after
measurement rather than from the expectation that motivated building it:

This family was built on the hypothesis that it would *decorrelate* depth from
proof length, since its rungs range from one-line to thirty-line proofs. That
hypothesis is false, and measurably so. Correlations between depth and
compositional reference-proof length:

    arithmetic       -0.13
    noninterference  +0.97

Arithmetic is the family where compositional proof length is independent of
depth; here the two are nearly the same variable. The reason is structural: in
this theory the deep rungs are deep *because* they do more case analysis, so
depth and length grow together, whereas arithmetic's rungs are uniformly ~3
lines regardless of how much sits beneath them.

Correlation between depth and *monolithic* length is ~0.98 in both families and
cannot be removed by choosing a family at all: monolithic length is by
definition the sum over ancestors, so it grows with depth mechanically. Breaking
that requires contrasting rungs of equal depth but differing ancestor count --
a lever that exists within `arithmetic` (at depth 3, one rung has 4 ancestors
and another has 2) and not here (ancestor count and depth correlate at 1.000).

What this family does contribute:

*   **Domain diversity.** Whether the decay constant is a property of proving-at-
    depth or of a particular corpus is only answerable with more than one
    theory.
*   **Genuine difficulty at the top.** The final rung is a real metatheorem with
    a thirty-line proof, not a three-line induction. Depth and intrinsic
    difficulty are entangled here in a way they are not in arithmetic.
"""

from __future__ import annotations

from ..ladder import Family, Rung

DEFINITIONS = """\
inductive {Exp} where
  | {lit} (n : Nat)
  | {var} (x : Nat)
  | {eadd} (a b : {Exp})

inductive {Com} where
  | {cskip}
  | {cassign} (x : Nat) (e : {Exp})
  | {cseq} (c d : {Com})
  | {cite} (e : {Exp}) (c d : {Com})

abbrev {St} := Nat → Nat
abbrev {Ctx} := Nat → Bool   -- true = secret, false = public

def {upd} (s : {St}) (x v : Nat) : {St} := fun y => if y = x then v else s y

def {evalE} (s : {St}) : {Exp} → Nat
  | .{lit} n    => n
  | .{var} x    => s x
  | .{eadd} a b => {evalE} s a + {evalE} s b

def {lvl} (G : {Ctx}) : {Exp} → Bool
  | .{lit} _    => false
  | .{var} x    => G x
  | .{eadd} a b => {lvl} G a || {lvl} G b

def {evalC} (s : {St}) : {Com} → {St}
  | .{cskip}       => s
  | .{cassign} x e => {upd} s x ({evalE} s e)
  | .{cseq} c d    => {evalC} ({evalC} s c) d
  | .{cite} e c d  => if {evalE} s e = 0 then {evalC} s d else {evalC} s c

def {wt} (G : {Ctx}) (pc : Bool) : {Com} → Bool
  | .{cskip}       => true
  | .{cassign} x e => !({lvl} G e || pc) || G x
  | .{cseq} c d    => {wt} G pc c && {wt} G pc d
  | .{cite} e c d  => {wt} G (pc || {lvl} G e) c && {wt} G (pc || {lvl} G e) d

def {lowEq} (G : {Ctx}) (s t : {St}) : Prop := ∀ x, G x = false → s x = t x
"""

RUNGS = (
    Rung(
        key="lowEq_refl",
        role="public agreement is reflexive",
        binders="(G : {Ctx}) (s : {St})",
        statement="{lowEq} G s s",
        proof="  intro x _; rfl",
    ),
    Rung(
        key="lowEq_symm",
        role="public agreement is symmetric",
        binders="(G : {Ctx}) (s t : {St}) (h : {lowEq} G s t)",
        statement="{lowEq} G t s",
        proof="  intro x hx; exact (h x hx).symm",
    ),
    Rung(
        key="lowEq_trans",
        role="public agreement is transitive",
        binders="(G : {Ctx}) (s t u : {St}) (h1 : {lowEq} G s t) "
                "(h2 : {lowEq} G t u)",
        statement="{lowEq} G s u",
        proof="  intro x hx; exact (h1 x hx).trans (h2 x hx)",
    ),
    Rung(
        key="evalE_agree",
        role="public expressions evaluate alike in agreeing states",
        binders="(G : {Ctx}) (s t : {St}) (e : {Exp}) "
                "(hl : {lvl} G e = false) (h : {lowEq} G s t)",
        statement="{evalE} s e = {evalE} t e",
        proof="  induction e with\n"
              "  | {lit} n => rfl\n"
              "  | {var} x => exact h x (by simpa [{lvl}] using hl)\n"
              "  | {eadd} a b iha ihb =>\n"
              "    simp [{lvl}, Bool.or_eq_false_iff] at hl\n"
              "    simp [{evalE}, iha hl.1, ihb hl.2]",
    ),
    Rung(
        key="assign_conf",
        role="a secret-context assignment cannot touch a public variable",
        binders="(G : {Ctx}) (s : {St}) (x : Nat) (e : {Exp}) "
                "(hw : {wt} G true (.{cassign} x e) = true)",
        statement="{lowEq} G s ({upd} s x ({evalE} s e))",
        proof="  simp [{wt}] at hw\n"
              "  intro y hy\n"
              "  have : y ≠ x := by\n"
              "    intro h; rw [h] at hy; rw [hw] at hy; exact Bool.noConfusion hy\n"
              "  simp [{upd}, this]",
    ),
    Rung(
        key="confinement",
        role="secret-context code leaves the public projection fixed",
        binders="(G : {Ctx}) (c : {Com})",
        statement="∀ (s : {St}), {wt} G true c = true → {lowEq} G s ({evalC} s c)",
        proof="  induction c with\n"
              "  | {cskip} => intro s _; exact {lowEq_refl} G s\n"
              "  | {cassign} x e => intro s hw; exact {assign_conf} G s x e hw\n"
              "  | {cseq} c d ihc ihd =>\n"
              "    intro s hw\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    exact {lowEq_trans} G _ _ _ (ihc s hw.1) (ihd ({evalC} s c) hw.2)\n"
              "  | {cite} e c d ihc ihd =>\n"
              "    intro s hw\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    by_cases hg : {evalE} s e = 0\n"
              "    · simp [{evalC}, hg]; exact ihd s hw.2\n"
              "    · simp [{evalC}, hg]; exact ihc s hw.1",
        deps=("lowEq_refl", "assign_conf", "lowEq_trans"),
    ),
    Rung(
        key="noninterference",
        role="well-typed programs leak nothing from secret to public",
        binders="(G : {Ctx}) (c : {Com})",
        statement="∀ (s t : {St}), {wt} G false c = true → {lowEq} G s t → "
                  "{lowEq} G ({evalC} s c) ({evalC} t c)",
        proof="  induction c with\n"
              "  | {cskip} => intro s t _ h; exact h\n"
              "  | {cassign} x e =>\n"
              "    intro s t hw h y hy\n"
              "    by_cases hyx : y = x\n"
              "    · subst hyx\n"
              "      have hlv : {lvl} G e = false := by\n"
              "        simp [{wt}, hy] at hw; exact hw\n"
              "      simp [{evalC}, {upd}, {evalE_agree} G s t e hlv h]\n"
              "    · simp [{evalC}, {upd}, hyx]; exact h y hy\n"
              "  | {cseq} c d ihc ihd =>\n"
              "    intro s t hw h\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    exact ihd _ _ hw.2 (ihc _ _ hw.1 h)\n"
              "  | {cite} e c d ihc ihd =>\n"
              "    intro s t hw h\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    by_cases hlv : {lvl} G e = false\n"
              "    · have hg : {evalE} s e = {evalE} t e := "
              "{evalE_agree} G s t e hlv h\n"
              "      simp [hlv] at hw\n"
              "      by_cases hz : {evalE} s e = 0\n"
              "      · simp [{evalC}, hz, ← hg]; exact ihd _ _ hw.2 h\n"
              "      · simp [{evalC}, hz, ← hg]; exact ihc _ _ hw.1 h\n"
              "    · have hlvT : {lvl} G e = true := by\n"
              "        cases hl : {lvl} G e with\n"
              "        | false => exact absurd hl hlv\n"
              "        | true => rfl\n"
              "      simp [hlvT] at hw\n"
              "      have cs : {lowEq} G s ({evalC} s (.{cite} e c d)) := by\n"
              "        by_cases hz : {evalE} s e = 0\n"
              "        · simp [{evalC}, hz]; exact {confinement} G d s hw.2\n"
              "        · simp [{evalC}, hz]; exact {confinement} G c s hw.1\n"
              "      have ct : {lowEq} G t ({evalC} t (.{cite} e c d)) := by\n"
              "        by_cases hz : {evalE} t e = 0\n"
              "        · simp [{evalC}, hz]; exact {confinement} G d t hw.2\n"
              "        · simp [{evalC}, hz]; exact {confinement} G c t hw.1\n"
              "      exact {lowEq_trans} G _ _ _ "
              "({lowEq_trans} G _ _ _ ({lowEq_symm} G _ _ cs) h) ct",
        deps=("evalE_agree", "confinement", "lowEq_trans", "lowEq_symm"),
    ),
)

NONINTERFERENCE = Family(
    name="noninterference",
    definitions=DEFINITIONS,
    rungs=RUNGS,
    slots={
        "Exp": "type",
        "Com": "type",
        "St": "type",
        "Ctx": "type",
        "lit": "ctor",
        "var": "ctor",
        "eadd": "ctor",
        "cskip": "ctor",
        "cassign": "ctor",
        "cseq": "ctor",
        "cite": "ctor",
        "upd": "fn",
        "evalE": "fn",
        "lvl": "fn",
        "evalC": "fn",
        "wt": "fn",
        "lowEq": "fn",
    },
)
