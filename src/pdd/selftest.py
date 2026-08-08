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


def check_dynamic(verbose: bool = True) -> int:
    """Dynamic sampling must actually retire prompts and keep informative ones.

    Needs no Lean and no GPU: the rollouts are fabricated, since what is under
    test is the bookkeeping, not the kernel. It is here rather than left to a
    training run because its failure mode is invisible from a training log. A
    ledger that never influences the dataset produces exactly the curves a
    working one produces, only worse, and the first version of `pdd.dynamic`
    failed this way -- it redrew prompts at random each cycle, so by the time a
    prompt had a verdict it had already been discarded, and the filter applied
    to nothing. The pool exists to fix that; this asserts that it does.
    """
    import random

    from .dynamic import PromptPool, curate, prompt_id
    from .env import TaskSampler, all_families, split_families
    from .reward import Shaped
    from .rollout import Rollout

    def tokenize(text):  # every prompt fits; the limit is not what is under test
        return {"input_ids": text.split()}

    def fake(task, per):
        proved = sum(1 for v in per.values() if v == "proved")
        n = len(task.targets)
        return Rollout(
            task=task, text="", blocks={},
            shaped=Shaped(reward=proved / n, score=proved / n,
                          binary=proved == n, per_target=per,
                          n_targets=n, n_proved=proved),
        )

    train, _ = split_families(all_families(include_corpus=True))
    sampler = TaskSampler(families=train, depths=(1, 2), volume=3,
                          rng=random.Random(0))
    pool = PromptPool()
    pool.reconfigure(((1, 2), 3))
    def cycle(count: int, seed: int):
        return curate(
            sampler, pool, count=count, tokenizer=tokenize,
            max_prompt_tokens=10**9, oversample=6, rng=random.Random(seed),
        )

    first = cycle(32, 0)
    failures = 0

    def want(condition: bool, message: str) -> int:
        if condition:
            return 0
        if verbose:
            print(f"[FAIL] dynamic sampling: {message}")
        return 1

    failures += want(len(first.tasks) == 32, "cold pass returned a short dataset")
    failures += want(first.pool == 192, f"pool is {first.pool}, expected 192")

    # Four prompts every rollout fails flat; four whose per-target outcomes
    # differ between rollouts even though their totals match.
    group = 4
    rollouts = []
    for i, task in enumerate(first.tasks[:8]):
        for j in range(group):
            if i < 4:
                per = {k: "compile_error" for k in task.targets}
            else:
                per = {k: ("proved" if (j + n) % 2 else "compile_error")
                       for n, k in enumerate(task.targets)}
            rollouts.append(fake(task, per))
    pool.ledger.observe(rollouts, group)

    verdicts = [pool.ledger.verdict(prompt_id(t)) for t in first.tasks[:8]]
    failures += want(verdicts[:4] == ["hopeless"] * 4,
                     f"flat-zero prompts judged {verdicts[:4]}")
    failures += want(verdicts[4:] == ["informative"] * 4,
                     f"prompts with per-target spread judged {verdicts[4:]}")

    hopeless = {prompt_id(t) for t in first.tasks[:4]}
    informative = {prompt_id(t) for t in first.tasks[4:8]}
    second = cycle(32, 1)
    chosen = {prompt_id(t) for t in second.tasks}
    failures += want(second.retired == 4, f"retired {second.retired}, expected 4")
    failures += want(not (chosen & hopeless), "a retired prompt came back")
    failures += want(chosen >= informative,
                     "an informative prompt was not selected")

    # A stage change invalidates the prompts but not what was learned of them.
    pool.reconfigure(((3, 4), 4))
    failures += want(pool.tasks == {}, "pool survived a difficulty change")
    failures += want(
        pool.ledger.verdict(next(iter(informative))) == "informative",
        "the ledger was cleared along with the pool",
    )

    if verbose:
        print(f"dynamic sampling: {8 - failures}/8 properties hold"
              if failures else "dynamic sampling: retires, reserves, resets")
    return failures


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
    ap.add_argument("--only-dynamic", action="store_true",
                    help="run just the dynamic-sampling check, which needs "
                         "neither Lean nor a GPU")
    args = ap.parse_args()

    if args.only_dynamic:
        sys.exit(1 if check_dynamic() else 0)

    failures = check_toolchain()
    failures += check_dynamic()
    if not args.skip_pools:
        failures += check_pools()
    for s in range(args.seed, args.seed + args.seeds):
        if args.seeds > 1:
            print(f"\n--- seed {s} ---")
        failures += run(args.family, s)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
