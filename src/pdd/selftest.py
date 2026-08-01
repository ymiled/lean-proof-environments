"""Certify a family: every task must be solvable, and every dependency real.

Run before any sweep. Three checks per rung:

1.  The compositional rendering is proved by its reference proof.
2.  The monolithic rendering is proved by its inlining reference proof. Without
    this a decay curve could be reporting that the tasks were impossible rather
    than anything about the policy.
3.  The *citing* proof fails once ancestors are removed. A dependency that
    survives removal is decorative, and the rung measures nothing. This check
    exists because an early ladder passed checks 1 and 2 while having no real
    depth at all: its goals were closed terms, so `simp` evaluated them.
"""

from __future__ import annotations

import argparse
import sys

from .families import DEFAULT, FAMILIES
from .grader import grade, grade_source
from .ladder import Instance
from .task import Condition, Task


def run(family_name: str = DEFAULT, seed: int = 0, verbose: bool = True) -> int:
    family = FAMILIES[family_name]
    inst = Instance.sample(family, seed)
    failures = 0

    ordered = sorted(family.by_key, key=family.depth_of)
    for key in ordered:
        depth = family.depth_of(key)
        comp = Task.single(inst, key, Condition.COMPOSITIONAL)
        res = grade(comp, comp.reference_solution())
        if not res.ok:
            failures += 1
        if verbose:
            print(
                f"[{'ok ' if res.ok else 'FAIL'}] d={depth} {key:<18}"
                f"{inst.names[key]:<10} {res.verdict.value}"
                + (f" :: {res.detail}" if res.detail else "")
            )

        mono = Task.single(inst, key, Condition.MONOLITHIC)
        mres = grade(mono, mono.reference_solution())
        if not mres.ok:
            failures += 1
            if verbose:
                print(f"       ^ MONOLITHIC UNSOLVABLE: {mres.verdict.value} "
                      f":: {mres.detail}")

        # Depth check. Compile the rung with its ancestors absent from the file
        # entirely. Grading a monolithic *task* cannot do this: there the
        # ancestors are themselves targets, so any missing block is filled with
        # `sorry` and the axiom audit rejects it for the wrong reason, which
        # makes the check silently vacuous for every multi-target rung.
        if family.by_key[key].deps:
            solo = (
                f"{inst.definitions}\n\n{inst.render_rung(key)}\n\n"
                f"#print axioms {inst.names[key]}\n"
            )
            cheat = grade_source(solo, [inst.names[key]])
            if cheat.ok:
                failures += 1
                if verbose:
                    print(f"       ^ DEPTH NOT REAL: {key} still proves with "
                          f"ancestors removed")

    if verbose:
        print()
        print(f"family={family_name}  {len(ordered)} rungs  "
              f"depths {min(map(family.depth_of, ordered))}"
              f"..{max(map(family.depth_of, ordered))}  {failures} failure(s)")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=DEFAULT, choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(1 if run(args.family, args.seed) else 0)


if __name__ == "__main__":
    main()
