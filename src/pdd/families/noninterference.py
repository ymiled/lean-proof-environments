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

Role in the benchmark: this is the **synthetic control**, not the evidence
source. It is a toy IFC theory, so it supports per-seed renaming of every
identifier, which an imported or machine-checked corpus cannot without breaking
its own imports. That makes it the only family in which the contamination
defence can be measured: run matched tasks under renamed and canonical
identifiers and compare. `vsi` carries the headline numbers; this family exists
to say how much those numbers owe to recall.

Four rungs were added deliberately after measurement. `conf_ite` and
`assign_ni` extract cases from what was originally a single 37-line soundness
proof, dropping the target to 22 lines; that alone took the top theorem from
unprovable to provable for a fixed policy at a fixed budget. `wt_anti` is an
18-line proof at depth 1 and `ite_high_ni` a 3-line proof at depth 3, a
shallow-long and a deep-short rung, which together cut the correlation between
depth and reference proof length from +0.97 to +0.56. `upd_pair` and `wt_anti`
are also maximal, which is what lets antichains exist here at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ladder import Family, Rung

_GRAPH = Path(__file__).resolve().parents[3] / "results" / "graph-noninterference.json"


def _extracted_deps() -> dict[str, tuple[str, ...]]:
    """Edges established by deletion, not declared here.

    Regenerate with `uv run python -m pdd.extract --family noninterference`.
    Hand-declaring them is what produced four fabricated edges in a sibling
    family, so this module no longer states any.
    """
    data = json.loads(_GRAPH.read_text())
    if data.get("method") != "necessity":
        raise ValueError(f"{_GRAPH} was not produced by necessity extraction")
    return {k: tuple(v) for k, v in data["direct"].items()}

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
    # --- maximal rungs -------------------------------------------------
    # Nothing depends on these, and neither depends on the other. Before they
    # existed this family had exactly one maximal rung -- the top theorem --
    # which made it a funnel: withheld sets must be up-closed, so withholding
    # any lemma dragged in everything above it, and no antichain of size two
    # could be formed at any k. The chain/antichain contrast was therefore
    # structurally impossible here, not merely underpowered.
    Rung(
        key="upd_pair",
        role="public agreement survives a common update",
        binders="(G : {Ctx}) (s t : {St}) (x v : Nat) (h : {lowEq} G s t)",
        statement="{lowEq} G ({upd} s x v) ({upd} t x v)",
        proof="  intro y hy\n"
              "  by_cases hyx : y = x\n"
              "  · simp [{upd}, hyx]\n"
              "  · simp [{upd}, hyx]; exact h y hy",
    ),
    Rung(
        key="wt_anti",
        role="typing is antitone in the program-counter level",
        binders="(G : {Ctx}) (c : {Com})",
        statement="{wt} G true c = true → {wt} G false c = true",
        proof="  induction c with\n"
              "  | {cskip} => intro _; rfl\n"
              "  | {cassign} x e =>\n"
              "    intro hw\n"
              "    simp [{wt}] at hw\n"
              "    simp [{wt}, hw]\n"
              "  | {cseq} c d ihc ihd =>\n"
              "    intro hw\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    simp [{wt}, Bool.and_eq_true]\n"
              "    exact ⟨ihc hw.1, ihd hw.2⟩\n"
              "  | {cite} e c d ihc ihd =>\n"
              "    intro hw\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    simp [{wt}, Bool.and_eq_true]\n"
              "    cases hl : {lvl} G e with\n"
              "    | true => exact hw\n"
              "    | false => exact ⟨ihc hw.1, ihd hw.2⟩",
    ),
    # --- intermediate rung ---------------------------------------------
    # Isolates the conditional step of confinement. It exists to soften the
    # family's difficulty cliff: reference proof lengths ran 1, 1, 1, 5, 6 at
    # depth 1 and then jumped straight to 13 at `confinement` and 37 at the top
    # theorem, so a policy either cleared everything up to 6 lines or nothing at
    # all, and no cell in between could be observed.
    Rung(
        key="conf_ite",
        role="conditional step of confinement",
        binders="(G : {Ctx}) (e : {Exp}) (c d : {Com}) (s : {St}) "
                "(hc : {lowEq} G s ({evalC} s c)) "
                "(hd : {lowEq} G s ({evalC} s d))",
        statement="{lowEq} G s ({evalC} s (.{cite} e c d))",
        proof="  by_cases hz : {evalE} s e = 0\n"
              "  · simp [{evalC}, hz]; exact hd\n"
              "  · simp [{evalC}, hz]; exact hc",
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
              "    exact {conf_ite} G e c d s (ihc s hw.1) (ihd s hw.2)",
    ),
    # --- intermediate rungs, extracted from the top theorem --------------
    # The soundness proof was originally a single 37-line rung, and the family
    # floored: a policy either cleared everything at 6 lines or nothing at all,
    # with no observable cell in between. Splitting it into its case analyses
    # gives graded difficulty and drops the top rung to 17 lines.
    Rung(
        key="assign_ni",
        role="assignment case of noninterference",
        binders="(G : {Ctx}) (s t : {St}) (x : Nat) (e : {Exp}) "
                "(hw : {wt} G false (.{cassign} x e) = true) (h : {lowEq} G s t)",
        statement="{lowEq} G ({evalC} s (.{cassign} x e)) "
                  "({evalC} t (.{cassign} x e))",
        proof="  intro y hy\n"
              "  by_cases hyx : y = x\n"
              "  · subst hyx\n"
              "    have hlv : {lvl} G e = false := by\n"
              "      simp [{wt}, hy] at hw; exact hw\n"
              "    simp [{evalC}, {upd}, {evalE_agree} G s t e hlv h]\n"
              "  · simp [{evalC}, {upd}, hyx]; exact h y hy",
    ),
    Rung(
        key="ite_high_ni",
        role="secret-guard conditional case of noninterference",
        binders="(G : {Ctx}) (e : {Exp}) (c d : {Com}) (s t : {St}) "
                "(hc : {wt} G true c = true) (hd : {wt} G true d = true) "
                "(h : {lowEq} G s t)",
        statement="{lowEq} G ({evalC} s (.{cite} e c d)) "
                  "({evalC} t (.{cite} e c d))",
        proof="  have cs := {conf_ite} G e c d s "
              "({confinement} G c s hc) ({confinement} G d s hd)\n"
              "  have ct := {conf_ite} G e c d t "
              "({confinement} G c t hc) ({confinement} G d t hd)\n"
              "  exact {lowEq_trans} G _ _ _ "
              "({lowEq_trans} G _ _ _ ({lowEq_symm} G _ _ cs) h) ct",
    ),
    Rung(
        key="noninterference",
        role="well-typed programs leak nothing from secret to public",
        binders="(G : {Ctx}) (c : {Com})",
        statement="∀ (s t : {St}), {wt} G false c = true → {lowEq} G s t → "
                  "{lowEq} G ({evalC} s c) ({evalC} t c)",
        proof="  induction c with\n"
              "  | {cskip} => intro s t _ h; exact h\n"
              "  | {cassign} x e => intro s t hw h; "
              "exact {assign_ni} G s t x e hw h\n"
              "  | {cseq} c d ihc ihd =>\n"
              "    intro s t hw h\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    exact ihd _ _ hw.2 (ihc _ _ hw.1 h)\n"
              "  | {cite} e c d ihc ihd =>\n"
              "    intro s t hw h\n"
              "    simp [{wt}, Bool.and_eq_true] at hw\n"
              "    by_cases hlv : {lvl} G e = false\n"
              "    · have hg := {evalE_agree} G s t e hlv h\n"
              "      simp [hlv] at hw\n"
              "      by_cases hz : {evalE} s e = 0\n"
              "      · simp [{evalC}, hz, ← hg]; exact ihd _ _ hw.2 h\n"
              "      · simp [{evalC}, hz, ← hg]; exact ihc _ _ hw.1 h\n"
              "    · have hlvT : {lvl} G e = true := by\n"
              "        cases hl : {lvl} G e with\n"
              "        | false => exact absurd hl hlv\n"
              "        | true => rfl\n"
              "      simp [hlvT] at hw\n"
              "      exact {ite_high_ni} G e c d s t hw.1 hw.2 h",
    ),
)

_deps = _extracted_deps()
RUNGS = tuple(
    Rung(key=r.key, role=r.role, binders=r.binders, statement=r.statement,
         proof=r.proof, deps=_deps.get(r.key, ()), raw=r.raw)
    for r in RUNGS
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
