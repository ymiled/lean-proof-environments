"""Generate information-flow-security theories, each with its own soundness chain.

A security type system is parameterised: how many operators the expression
grammar has, which way the conditional tests its guard, how the state is
represented. Each point in that space is a different Lean theory with the same
metatheorem, and each is a fresh ladder.

Why generate rather than hand-write more families. Two reasons, both learned the
hard way. Hand-authored dependency graphs drift, and four fabricated edges in an
earlier family went undetected because the check meant to catch them was inert.
And a hand-built ladder invites the objection that its shape was chosen to
produce a particular curve. A generated theory has neither problem: its graph is
extracted by necessity, and nobody chose its depths.

What is *not* parameterised is the proof strategy. Every instance is discharged
by the same tactic skeleton, so a variant either compiles for structural reasons
or is rejected by `selftest` before it can enter the corpus. The generator is
allowed to emit theories that do not work; it is not allowed to ship them.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

SPECS = Path(__file__).resolve().parents[2] / "corpus"


@dataclass(frozen=True)
class Variant:
    """One point in the parameter space."""

    binops: int          # 1..3 arithmetic operators in the expression grammar
    guard_zero: bool     # conditional takes the else-branch when the guard is 0
    seq_first: bool      # sequencing evaluates left-to-right (False flips the pair)

    @property
    def tag(self) -> str:
        return f"b{self.binops}{'z' if self.guard_zero else 'n'}{'l' if self.seq_first else 'r'}"


#: The operators available to the expression grammar, in order. Each is a
#: constructor plus its evaluation; `lvl` joins its operands regardless, which is
#: what keeps the agreement proof uniform across variants.
_OPS = [
    ("plus",  "{a} + {b}"),
    ("times", "{a} * {b}"),
    ("maxOf", "max {a} {b}"),
]


def definitions(v: Variant) -> str:
    ctors = "\n".join(
        f"  | {{op{i}}} (a b : {{Exp}})" for i in range(v.binops)
    )
    eval_cases = "\n".join(
        "  | .{op%d} a b => %s" % (i, tmpl.format(a="({evalE} s a)", b="({evalE} s b)"))
        for i, (_, tmpl) in enumerate(_OPS[: v.binops])
    )
    lvl_cases = "\n".join(
        f"  | .{{op{i}}} a b => {{lvl}} G a || {{lvl}} G b" for i in range(v.binops)
    )
    then_br, else_br = (("{evalC} s d", "{evalC} s c") if v.guard_zero
                        else ("{evalC} s c", "{evalC} s d"))
    seq = "{evalC} ({evalC} s c) d" if v.seq_first else "{evalC} ({evalC} s d) c"
    return f"""\
inductive {{Exp}} where
  | {{lit}} (n : Nat)
  | {{var}} (x : Nat)
{ctors}

inductive {{Com}} where
  | {{cskip}}
  | {{cassign}} (x : Nat) (e : {{Exp}})
  | {{cseq}} (c d : {{Com}})
  | {{cite}} (e : {{Exp}}) (c d : {{Com}})

abbrev {{St}} := Nat → Nat
abbrev {{Ctx}} := Nat → Bool   -- true = secret, false = public

def {{upd}} (s : {{St}}) (x v : Nat) : {{St}} := fun y => if y = x then v else s y

def {{evalE}} (s : {{St}}) : {{Exp}} → Nat
  | .{{lit}} n => n
  | .{{var}} x => s x
{eval_cases}

def {{lvl}} (G : {{Ctx}}) : {{Exp}} → Bool
  | .{{lit}} _ => false
  | .{{var}} x => G x
{lvl_cases}

def {{evalC}} (s : {{St}}) : {{Com}} → {{St}}
  | .{{cskip}}       => s
  | .{{cassign}} x e => {{upd}} s x ({{evalE}} s e)
  | .{{cseq}} c d    => {seq}
  | .{{cite}} e c d  => if {{evalE}} s e = 0 then {then_br} else {else_br}

def {{wt}} (G : {{Ctx}}) (pc : Bool) : {{Com}} → Bool
  | .{{cskip}}       => true
  | .{{cassign}} x e => !({{lvl}} G e || pc) || G x
  | .{{cseq}} c d    => {{wt}} G pc c && {{wt}} G pc d
  | .{{cite}} e c d  => {{wt}} G (pc || {{lvl}} G e) c && {{wt}} G (pc || {{lvl}} G e) d

def {{lowEq}} (G : {{Ctx}}) (s t : {{St}}) : Prop := ∀ x, G x = false → s x = t x
"""


def rungs(v: Variant) -> list[dict]:
    """The soundness chain. Cases that vary with the grammar are expanded here."""
    agree_cases = "\n".join(
        f"  | {{op{i}}} a b iha ihb =>\n"
        f"    simp [{{lvl}}, Bool.or_eq_false_iff] at hl\n"
        f"    simp [{{evalE}}, iha hl.1, ihb hl.2]"
        for i in range(v.binops)
    )
    # The conditional branches swap with `guard_zero`, so the two `by_cases`
    # arms are emitted in the matching order.
    hi, lo = ("hd", "hc") if v.guard_zero else ("hc", "hd")
    th_ih, th_hw = ("ihd", "hw.2") if v.guard_zero else ("ihc", "hw.1")
    el_ih, el_hw = ("ihc", "hw.1") if v.guard_zero else ("ihd", "hw.2")
    return [
        dict(key="lowEq_refl", role="public agreement is reflexive",
             binders="(G : {Ctx}) (s : {St})", statement="{lowEq} G s s",
             proof="  intro x _; rfl"),
        dict(key="lowEq_symm", role="public agreement is symmetric",
             binders="(G : {Ctx}) (s t : {St}) (h : {lowEq} G s t)",
             statement="{lowEq} G t s",
             proof="  intro x hx; exact (h x hx).symm"),
        dict(key="lowEq_trans", role="public agreement is transitive",
             binders="(G : {Ctx}) (s t u : {St}) (h1 : {lowEq} G s t) "
                     "(h2 : {lowEq} G t u)",
             statement="{lowEq} G s u",
             proof="  intro x hx; exact (h1 x hx).trans (h2 x hx)"),
        dict(key="upd_pair", role="agreement survives a common update",
             binders="(G : {Ctx}) (s t : {St}) (x v : Nat) (h : {lowEq} G s t)",
             statement="{lowEq} G ({upd} s x v) ({upd} t x v)",
             proof="  intro y hy\n"
                   "  by_cases hyx : y = x\n"
                   "  · simp [{upd}, hyx]\n"
                   "  · simp [{upd}, hyx]; exact h y hy"),
        dict(key="evalE_agree",
             role="public expressions evaluate alike in agreeing states",
             binders="(G : {Ctx}) (s t : {St}) (e : {Exp}) "
                     "(hl : {lvl} G e = false) (h : {lowEq} G s t)",
             statement="{evalE} s e = {evalE} t e",
             proof="  induction e with\n"
                   "  | {lit} n => rfl\n"
                   "  | {var} x => exact h x (by simpa [{lvl}] using hl)\n"
                   + agree_cases),
        dict(key="assign_conf",
             role="a secret-context assignment cannot touch a public variable",
             binders="(G : {Ctx}) (s : {St}) (x : Nat) (e : {Exp}) "
                     "(hw : {wt} G true (.{cassign} x e) = true)",
             statement="{lowEq} G s ({upd} s x ({evalE} s e))",
             proof="  simp [{wt}] at hw\n"
                   "  intro y hy\n"
                   "  have : y ≠ x := by\n"
                   "    intro h; rw [h] at hy; rw [hw] at hy; exact Bool.noConfusion hy\n"
                   "  simp [{upd}, this]"),
        dict(key="conf_ite", role="conditional step of confinement",
             binders="(G : {Ctx}) (e : {Exp}) (c d : {Com}) (s : {St}) "
                     "(hc : {lowEq} G s ({evalC} s c)) "
                     "(hd : {lowEq} G s ({evalC} s d))",
             statement="{lowEq} G s ({evalC} s (.{cite} e c d))",
             proof="  by_cases hz : {evalE} s e = 0\n"
                   f"  · simp [{{evalC}}, hz]; exact {hi}\n"
                   f"  · simp [{{evalC}}, hz]; exact {lo}"),
        dict(key="confinement",
             role="secret-context code leaves the public projection fixed",
             binders="(G : {Ctx}) (c : {Com})",
             statement="∀ (s : {St}), {wt} G true c = true → "
                       "{lowEq} G s ({evalC} s c)",
             proof="  induction c with\n"
                   "  | {cskip} => intro s _; exact {lowEq_refl} G s\n"
                   "  | {cassign} x e => intro s hw; exact {assign_conf} G s x e hw\n"
                   "  | {cseq} c d ihc ihd =>\n"
                   "    intro s hw\n"
                   "    simp [{wt}, Bool.and_eq_true] at hw\n"
                   + ("    exact {lowEq_trans} G _ _ _ (ihc s hw.1) (ihd ({evalC} s c) hw.2)\n"
                      if v.seq_first else
                      "    exact {lowEq_trans} G _ _ _ (ihd s hw.2) (ihc ({evalC} s d) hw.1)\n")
                   + "  | {cite} e c d ihc ihd =>\n"
                     "    intro s hw\n"
                     "    simp [{wt}, Bool.and_eq_true] at hw\n"
                     "    exact {conf_ite} G e c d s (ihc s hw.1) (ihd s hw.2)"),
        dict(key="assign_ni", role="assignment case of noninterference",
             binders="(G : {Ctx}) (s t : {St}) (x : Nat) (e : {Exp}) "
                     "(hw : {wt} G false (.{cassign} x e) = true) "
                     "(h : {lowEq} G s t)",
             statement="{lowEq} G ({evalC} s (.{cassign} x e)) "
                       "({evalC} t (.{cassign} x e))",
             proof="  intro y hy\n"
                   "  by_cases hyx : y = x\n"
                   "  · subst hyx\n"
                   "    have hlv : {lvl} G e = false := by\n"
                   "      simp [{wt}, hy] at hw; exact hw\n"
                   "    simp [{evalC}, {upd}, {evalE_agree} G s t e hlv h]\n"
                   "  · simp [{evalC}, {upd}, hyx]; exact h y hy"),
        dict(key="ite_high_ni",
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
                   "({lowEq_trans} G _ _ _ ({lowEq_symm} G _ _ cs) h) ct"),
        dict(key="noninterference",
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
                   + ("    exact ihd _ _ hw.2 (ihc _ _ hw.1 h)\n" if v.seq_first
                      else "    exact ihc _ _ hw.1 (ihd _ _ hw.2 h)\n")
                   + "  | {cite} e c d ihc ihd =>\n"
                     "    intro s t hw h\n"
                     "    simp [{wt}, Bool.and_eq_true] at hw\n"
                     "    by_cases hlv : {lvl} G e = false\n"
                     "    · have hg := {evalE_agree} G s t e hlv h\n"
                     "      simp [hlv] at hw\n"
                     "      by_cases hz : {evalE} s e = 0\n"
                   + f"      · simp [{{evalC}}, hz, ← hg]; exact {th_ih} _ _ {th_hw} h\n"
                   + f"      · simp [{{evalC}}, hz, ← hg]; exact {el_ih} _ _ {el_hw} h\n"
                   + "    · have hlvT : {lvl} G e = true := by\n"
                     "        cases hl : {lvl} G e with\n"
                     "        | false => exact absurd hl hlv\n"
                     "        | true => rfl\n"
                     "      simp [hlvT] at hw\n"
                     "      exact {ite_high_ni} G e c d s t hw.1 hw.2 h"),
    ]


SLOTS = {
    "Exp": "type", "Com": "type", "St": "type", "Ctx": "type",
    "lit": "ctor", "var": "ctor", "cskip": "ctor", "cassign": "ctor",
    "cseq": "ctor", "cite": "ctor",
    "upd": "fn", "evalE": "fn", "lvl": "fn", "evalC": "fn", "wt": "fn",
    "lowEq": "fn",
}


def spec_for(v: Variant) -> dict:
    slots = dict(SLOTS)
    for i in range(v.binops):
        slots[f"op{i}"] = "ctor"
    return {
        "name": f"ifc-{v.tag}",
        "variant": {"binops": v.binops, "guard_zero": v.guard_zero,
                    "seq_first": v.seq_first},
        "definitions": definitions(v),
        "slots": slots,
        "rungs": rungs(v),
    }


def all_variants() -> list[Variant]:
    return [
        Variant(binops=b, guard_zero=g, seq_first=s)
        for b in (1, 2, 3)
        for g in (True, False)
        for s in (True, False)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SPECS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    started = time.time()
    written = []
    for v in all_variants():
        spec = spec_for(v)
        p = out / f"{spec['name']}.json"
        p.write_text(json.dumps(spec, indent=1))
        written.append(spec["name"])
    print(f"wrote {len(written)} theory specs to {out} in {time.time()-started:.1f}s")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
