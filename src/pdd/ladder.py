"""The lemma ladder: a dependency DAG over a freshly-named algebraic theory.

Design constraints, each of which was forced by an experiment rather than chosen
a priori (see `docs/design-log.md`):

1.  **Goals must stay open in the recursion variable.** An earlier ladder stated
    lemmas at a concrete point (`op_i one y = one`). Every such goal is a closed
    term, so `simp` simply *computes* it and the whole dependency structure
    collapses -- `simp_all [one, op1, op2, op3]` proved a depth-3 lemma with no
    lower lemmas in scope. Every rung here is universally quantified over the
    variables the definitions recurse on, which blocks reduction and forces
    genuine induction.

2.  **The theory is renamed per instance.** Type, constructors, operators and
    lemmas all get fresh identifiers drawn from a seed. The mathematical content
    is Peano arithmetic, which a model certainly knows, but the goal is to
    prevent *verbatim recall of a library proof term* (`Nat.mul_comm` and
    friends are in every training set). The model must reconstruct the argument
    against unfamiliar names. This weakens but does not eliminate the
    contamination confound, and the README says so.

3.  **Depth is measured, not asserted.** `depth` is derived from the DAG by
    longest path, so adding a rung cannot silently misreport difficulty.

The reference proofs are carried alongside each rung. They exist so the
generator can certify -- by actually compiling them -- that every task it emits
is solvable with the tactic vocabulary we permit. A benchmark that ships
unsolvable tasks measures nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property

#: Tactics the reference proofs stay inside. Deliberately small: no Mathlib, no
#: `decide`, no `native_decide`. Keeps the environment self-contained and keeps
#: the difficulty in the proof structure rather than in library search.
TACTIC_VOCABULARY = ("induction", "simp", "rw", "rfl", "exact", "apply", "intro")


@dataclass(frozen=True)
class Rung:
    """One lemma in the ladder.

    `statement` and `proof` are format strings over the renaming map, so a rung
    is a *schema* rather than a fixed piece of Lean text.
    """

    key: str
    #: Human-readable role, used in reports and plots.
    role: str
    #: Binders, e.g. "(x y : {T})".
    binders: str
    #: Proposition body, e.g. "{op1} {zero} y = y".
    statement: str
    #: Reference tactic block, newline separated, already indented two spaces.
    proof: str
    #: Keys of rungs this proof genuinely needs.
    deps: tuple[str, ...] = ()


# The theory: op1 = addition, op2 = multiplication, op3 = exponentiation, each
# defined by recursion on its second argument over a unary numeral type.
DEFINITIONS = """\
inductive {T} where
  | {zero} : {T}
  | {succ} : {T} → {T}

def {op1} : {T} → {T} → {T}
  | x, .{zero}   => x
  | x, .{succ} y => .{succ} ({op1} x y)

def {op2} : {T} → {T} → {T}
  | _, .{zero}   => .{zero}
  | x, .{succ} y => {op1} ({op2} x y) x

def {op3} : {T} → {T} → {T}
  | _, .{zero}   => .{succ} .{zero}
  | x, .{succ} y => {op2} ({op3} x y) x
"""


RUNGS: tuple[Rung, ...] = (
    Rung(
        key="add_zero_l",
        role="left unit for op1",
        binders="(y : {T})",
        statement="{op1} .{zero} y = y",
        proof="  induction y with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op1}, ih]",
    ),
    Rung(
        key="add_succ_l",
        role="left successor for op1",
        binders="(x y : {T})",
        statement="{op1} (.{succ} x) y = .{succ} ({op1} x y)",
        proof="  induction y with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op1}, ih]",
    ),
    Rung(
        key="add_assoc",
        role="associativity of op1",
        binders="(x y z : {T})",
        statement="{op1} ({op1} x y) z = {op1} x ({op1} y z)",
        proof="  induction z with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op1}, ih]",
    ),
    Rung(
        key="mul_zero_l",
        role="left absorbing element for op2",
        binders="(y : {T})",
        statement="{op2} .{zero} y = .{zero}",
        proof="  induction y with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op2}, ih, {op1}]",
    ),
    Rung(
        key="add_comm",
        role="commutativity of op1",
        binders="(x y : {T})",
        statement="{op1} x y = {op1} y x",
        proof="  induction y with\n"
              "  | {zero} => simp [{op1}, {add_zero_l}]\n"
              "  | {succ} k ih => simp [{op1}, {add_succ_l}, ih]",
        deps=("add_zero_l", "add_succ_l"),
    ),
    Rung(
        key="mul_add_distrib",
        role="op2 distributes over op1",
        binders="(x y z : {T})",
        statement="{op2} x ({op1} y z) = {op1} ({op2} x y) ({op2} x z)",
        proof="  induction z with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op1}, {op2}, ih, {add_assoc}]",
        deps=("add_assoc",),
    ),
    Rung(
        key="add_left_comm",
        role="left commutativity of op1",
        binders="(x y z : {T})",
        statement="{op1} x ({op1} y z) = {op1} y ({op1} x z)",
        proof="  rw [← {add_assoc}, {add_comm} x y, {add_assoc}]",
        deps=("add_assoc", "add_comm"),
    ),
    Rung(
        key="mul_assoc",
        role="associativity of op2",
        binders="(x y z : {T})",
        statement="{op2} ({op2} x y) z = {op2} x ({op2} y z)",
        proof="  induction z with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op2}, ih, {mul_add_distrib}]",
        deps=("mul_add_distrib",),
    ),
    Rung(
        key="mul_succ_l",
        role="left successor for op2",
        binders="(x y : {T})",
        statement="{op2} (.{succ} x) y = {op1} ({op2} x y) y",
        proof="  induction y with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op2}, ih, {op1}, {add_assoc}, "
              "{add_left_comm}, {add_comm}]",
        deps=("add_assoc", "add_left_comm", "add_comm"),
    ),
    Rung(
        key="pow_add",
        role="op3 turns op1 into op2",
        binders="(x y z : {T})",
        statement="{op3} x ({op1} y z) = {op2} ({op3} x y) ({op3} x z)",
        proof="  induction z with\n"
              "  | {zero} => simp [{op1}, {op3}, {op2}, {add_zero_l}]\n"
              "  | {succ} k ih => simp [{op1}, {op3}, ih, {mul_assoc}]",
        deps=("add_zero_l", "mul_assoc"),
    ),
    Rung(
        key="mul_comm",
        role="commutativity of op2",
        binders="(x y : {T})",
        statement="{op2} x y = {op2} y x",
        proof="  induction y with\n"
              "  | {zero} => simp [{op2}, {mul_zero_l}]\n"
              "  | {succ} k ih => simp [{op2}, {mul_succ_l}, ih]",
        deps=("mul_zero_l", "mul_succ_l"),
    ),
)

BY_KEY = {r.key: r for r in RUNGS}


def depth_of(key: str) -> int:
    """Longest path to a dependency-free rung. Depth 1 means "needs no lemma"."""
    rung = BY_KEY[key]
    if not rung.deps:
        return 1
    return 1 + max(depth_of(d) for d in rung.deps)


def transitive_deps(key: str) -> list[str]:
    """All ancestors of `key`, ordered so each appears after its own deps."""
    seen: list[str] = []

    def visit(k: str) -> None:
        for d in BY_KEY[k].deps:
            visit(d)
            if d not in seen:
                seen.append(d)

    visit(key)
    return seen


# Identifier pools for renaming. Chosen to be pronounceable and to carry no
# arithmetic connotation, so the surface syntax offers no hint that this is
# Peano arithmetic.
_TYPE_NAMES = ("Tally", "Quiver", "Strand", "Rung", "Gleam", "Notch", "Warp", "Spool")
_ZERO_NAMES = ("nil", "base", "root", "seed", "origin", "stem")
_SUCC_NAMES = ("step", "tick", "next", "bump", "grow", "shift")
_OP_NAMES = ("melt", "braid", "fuse", "weave", "bind", "knit", "forge", "spin",
             "twine", "clasp", "hew", "meld")


@dataclass
class Instance:
    """A renamed copy of the theory plus its ladder, ready to render as Lean."""

    seed: int
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def sample(cls, seed: int) -> "Instance":
        rng = random.Random(seed)
        ops = rng.sample(_OP_NAMES, 3)
        names = {
            "T": rng.choice(_TYPE_NAMES),
            "zero": rng.choice(_ZERO_NAMES),
            "succ": rng.choice(_SUCC_NAMES),
            "op1": ops[0],
            "op2": ops[1],
            "op3": ops[2],
        }
        # Lemma names are opaque so the role leaks nothing. A model that sees
        # `add_comm` can pattern-match the goal to a remembered proof; `lem_a7f2`
        # forces it to read the statement.
        for rung in RUNGS:
            names[rung.key] = f"lem_{rng.randrange(16**4):04x}"
        return cls(seed=seed, names=names)

    @cached_property
    def definitions(self) -> str:
        return DEFINITIONS.format(**self.names)

    def statement_of(self, key: str) -> str:
        r = BY_KEY[key]
        return (
            f"theorem {self.names[key]} {r.binders.format(**self.names)} : "
            f"{r.statement.format(**self.names)}"
        )

    def reference_proof_of(self, key: str) -> str:
        return BY_KEY[key].proof.format(**self.names)

    def render_rung(self, key: str) -> str:
        return f"{self.statement_of(key)} := by\n{self.reference_proof_of(key)}"

    def render_reference(self, key: str) -> str:
        """Full Lean file proving `key`, ancestors included, all with reference proofs.

        Compiling this is how the generator certifies a task is solvable.
        """
        parts = [self.definitions]
        parts.extend(self.render_rung(k) for k in transitive_deps(key))
        parts.append(self.render_rung(key))
        parts.append(f"#print axioms {self.names[key]}")
        return "\n\n".join(parts) + "\n"
