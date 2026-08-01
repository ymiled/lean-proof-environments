"""Certify the benchmark: every task must be solvable by its reference proof.

Run before any sweep. If a rung fails here, the sweep would be scoring models
against an impossible task and the resulting decay curve would be an artifact of
the benchmark rather than a property of the models.

Also checks the converse for the compositional condition: a task whose reference
proof cites ancestors must *fail* when those ancestors are removed. That is what
makes depth real rather than decorative -- if the monolithic rendering of a
depth-5 rung compiles with the same short proof, the rung is measuring nothing.
"""

from __future__ import annotations

import sys

from .grader import grade
from .ladder import BY_KEY, Instance, depth_of
from .task import Condition, Task


def run(seed: int = 0, verbose: bool = True) -> int:
    inst = Instance.sample(seed)
    failures = 0

    ordered = sorted(BY_KEY, key=depth_of)
    for key in ordered:
        depth = depth_of(key)
        comp = Task(inst, key, Condition.COMPOSITIONAL)
        res = grade(comp, comp.reference_solution())

        status = "ok " if res.ok else "FAIL"
        if not res.ok:
            failures += 1
        if verbose:
            print(
                f"[{status}] d={depth} {key:<18} {inst.names[key]:<10} "
                f"{res.verdict.value}"
                + (f" :: {res.detail}" if res.detail else "")
            )

        # The monolithic arm must also be solvable, or its decay curve would be
        # an artifact of impossible tasks rather than a property of the policy.
        mono = Task(inst, key, Condition.MONOLITHIC)
        mres = grade(mono, mono.reference_solution())
        if not mres.ok:
            failures += 1
            if verbose:
                print(
                    f"       ^ MONOLITHIC UNSOLVABLE: {mres.verdict.value} "
                    f":: {mres.detail}"
                )

        # Depth sanity: the *citing* proof must break once ancestors are gone.
        # If it still compiles, the dependency is decorative and the rung
        # measures nothing.
        if BY_KEY[key].deps:
            cheat = grade(mono, comp.reference_solution())
            if cheat.ok:
                failures += 1
                if verbose:
                    print(
                        f"       ^ DEPTH NOT REAL: {key} still proves with "
                        f"ancestors removed"
                    )

    if verbose:
        print()
        print(f"{len(ordered)} rungs, depths {min(map(depth_of, ordered))}"
              f"..{max(map(depth_of, ordered))}, {failures} failure(s)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
