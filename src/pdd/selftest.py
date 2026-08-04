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

Plus one check that is a property of the renaming machinery rather than of any
one family: no identifier in the renaming pools may already mean something to
Lean. See `check_pools`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from .families import DEFAULT, FAMILIES
from .grader import _lean_binary, _lean_env, grade, grade_source, pinned_toolchain
from .ladder import POOLS, Instance
from .task import Condition, Task


def check_toolchain(verbose: bool = True) -> int:
    """The running Lean must be the one the reference proofs were written for.

    Worth its own check because the failure mode is silent rather than loud. A
    reference proof that stops compiling under a different toolchain is reported
    as a compile error, which is exactly what a wrong proof looks like -- so on
    a host with the wrong Lean, every reward is deflated and nothing says so.
    That is survivable in a sweep, where the numbers look obviously broken, and
    corrosive in a training run, where the policy simply learns from noise.
    """
    pin = pinned_toolchain()
    if pin is None:
        if verbose:
            print("[warn] no lean-toolchain pin; grading uses whatever lean is "
                  "on PATH")
        return 0
    proc = subprocess.run(
        [_lean_binary(), "--version"],
        capture_output=True, text=True, env=_lean_env(),
    )
    version = proc.stdout.strip() or proc.stderr.strip()
    # `leanprover/lean4:v4.32.2` -> `4.32.2`, which is how `lean --version`
    # spells it.
    wanted = pin.rsplit(":v", 1)[-1]
    ok = f"version {wanted}" in version
    if verbose:
        print(f"[{'ok ' if ok else 'FAIL'}] toolchain {pin} :: {version}")
    return 0 if ok else 1


def check_pools(verbose: bool = True) -> int:
    """Every renaming identifier must be unknown to Lean's prelude.

    A pool name that Lean already knows shadows nothing in the file but wins
    name resolution inside tactics, so `simp [f]` picks up the prelude's `f`
    instead of the definition. `bind` was in the pool and did exactly that:
    `simp [bind]` resolved to `Bind.bind`, which is not a proposition, and the
    *reference* proof stopped compiling. Only the seeds that drew that name were
    affected, so it stayed invisible while sweeps used a handful of fixed seeds.

    Checked by asking Lean directly rather than against a hardcoded list, so the
    check keeps working across toolchain versions.
    """
    names = sorted({n for pool in POOLS.values() for n in pool})
    source = "\n".join(f"#check @{n}" for n in names) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Pools.lean"
        path.write_text(source)
        proc = subprocess.run(
            [_lean_binary(), str(path)], capture_output=True, text=True
        )
    output = proc.stdout + proc.stderr
    unknown = {n for n in names if f"`{n}`" in output or f"'{n}'" in output}
    collisions = sorted(set(names) - unknown)
    if verbose:
        for name in collisions:
            print(f"[FAIL] pool identifier {name!r} already exists in Lean")
        print(f"pools: {len(names)} identifiers, {len(collisions)} collision(s)")
    return len(collisions)


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
    ap.add_argument("--seeds", type=int, default=1,
                    help="check seeds `seed` .. `seed + seeds - 1`. Renaming "
                         "faults are seed-dependent, so one seed proves little")
    ap.add_argument("--skip-pools", action="store_true")
    args = ap.parse_args()

    failures = check_toolchain()
    if not args.skip_pools:
        failures += check_pools()
    for s in range(args.seed, args.seed + args.seeds):
        if args.seeds > 1:
            print(f"\n--- seed {s} ---")
        failures += run(args.family, s)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
