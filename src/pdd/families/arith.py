"""RETAINED FOR REGRADING ARCHIVED RUNS ONLY -- not in `FAMILIES`.

The arithmetic family was removed from the benchmark (see `__init__`): it
measures general proving ability rather than IFC verification and is heavily
represented in training data. The module is kept so the archived sweeps in
`results/archive/` can be re-graded against the exact ladder that produced them,
which a description in prose could not guarantee. Nothing new is collected here.

Arithmetic family: the Peano development over a fresh numeral type.

Operations recurse on their *second* argument, which is what makes the
left-handed statements (`0 + y`, `(x+1) * y`) require induction rather than
falling to computation. That asymmetry is the entire source of the chain.

Known limitation, measured rather than assumed: in this family depth,
dependency count and monolithic proof length are collinear at r > 0.96, so a
decay curve here cannot attribute the effect to depth specifically. See the
`noninterference` family, which decorrelates them.
"""

from __future__ import annotations

from ..ladder import Family, Rung

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

RUNGS = (
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
    # --- maximal rungs -------------------------------------------------
    # Nothing depends on these, and none depends on another, so together with
    # `mul_comm` and `pow_add` they form a six-element antichain.
    #
    # They exist because of a structural fact discovered while building the
    # chain/antichain contrast: a withheld set must be up-closed (a supplied
    # lemma cites its own dependencies, so withholding a lemma forces
    # withholding everything above it). Antichains can therefore only be drawn
    # from *maximal* elements. Before these rungs the family had exactly two,
    # which capped matched contrasts at k = 2 and left the design with a single
    # comparison and no dose-response curve.
    Rung(
        key="one_mul_l",
        role="left unit for op2",
        binders="(y : {T})",
        statement="{op2} (.{succ} .{zero}) y = y",
        proof="  induction y with\n"
              "  | {zero} => rfl\n"
              "  | {succ} k ih => simp [{op2}, ih, {op1}]",
    ),
    Rung(
        key="mul_one_r",
        role="right unit for op2",
        binders="(x : {T})",
        statement="{op2} x (.{succ} .{zero}) = x",
        proof="  simp [{op2}, {add_zero_l}]",
        deps=("add_zero_l",),
    ),
    Rung(
        key="mul_two_r",
        role="op2 by two is self-op1",
        binders="(x : {T})",
        statement="{op2} x (.{succ} (.{succ} .{zero})) = {op1} x x",
        proof="  simp [{op2}, {add_zero_l}]",
        deps=("add_zero_l",),
    ),
    Rung(
        key="add_swap4",
        role="op1 swaps its outer operands",
        binders="(x y z : {T})",
        statement="{op1} ({op1} x y) z = {op1} ({op1} x z) y",
        proof="  rw [{add_assoc}, {add_comm} y z, ← {add_assoc}]",
        deps=("add_assoc", "add_comm"),
    ),
)

ARITHMETIC = Family(
    name="arithmetic",
    definitions=DEFINITIONS,
    rungs=RUNGS,
    slots={
        "T": "type",
        "zero": "ctor",
        "succ": "ctor",
        "op1": "fn",
        "op2": "fn",
        "op3": "fn",
    },
)
