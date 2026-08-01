"""Run the chain/antichain contrast against a policy that is an external agent.

Splits the sweep into two halves so a policy with no HTTP API can sit in the
middle:

    export   -- write one prompt file per task into a run directory
    ingest   -- read the corresponding answer files, grade, summarise

The point of the experiment is that chain and antichain tasks are matched on
volume (same number of lemmas) and on reference proof length, and differ only in
whether the targets form a dependency chain. A gap in pass rate implicates
chaining; no gap implicates volume.

    uv run python -m pdd.agentrun export --seeds 3 --samples 2
    # ... an agent writes runs/<id>.out for each runs/<id>.txt ...
    uv run python -m pdd.agentrun ingest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .design import find_contrasts
from .families import FAMILIES
from .grader import grade
from .ladder import Instance
from .task import Task, parse_blocks

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Spec:
    task_id: str
    family: str
    seed: int
    sample: int
    arm: str
    k: int
    targets: tuple[str, ...]
    residual_depth: int
    reference_loc: int


def build_depth(family_name: str, seeds: int, samples: int) -> list[tuple[Spec, Task]]:
    """The depth experiment: every rung, compositional versus monolithic.

    Complements the contrast experiment. Where chain/antichain isolates chaining
    from volume at matched size, this measures the raw quantity Theorem report --
    success against dependency depth with ancestors supplied or withheld.
    """
    from .task import Condition

    family = FAMILIES[family_name]
    out: list[tuple[Spec, Task]] = []
    for seed in range(seeds):
        inst = Instance.sample(family, seed)
        for key in family.by_key:
            for cond in Condition:
                task = Task.single(inst, key, cond)
                for s in range(samples):
                    spec = Spec(
                        task_id=f"{family_name}-d{family.depth_of(key)}-"
                                f"{cond.value}-{key}-s{seed}-n{s}",
                        family=family_name,
                        seed=seed,
                        sample=s,
                        arm=cond.value,
                        k=family.depth_of(key),
                        targets=task.targets,
                        residual_depth=task.residual_depth,
                        reference_loc=task.reference_loc,
                    )
                    out.append((spec, task))
    return out


def build(family_name: str, seeds: int, samples: int) -> list[tuple[Spec, Task]]:
    family = FAMILIES[family_name]
    out: list[tuple[Spec, Task]] = []
    for seed in range(seeds):
        inst = Instance.sample(family, seed)
        for c in find_contrasts(family, inst):
            for arm, targets in (("chain", c.chain), ("antichain", c.antichain)):
                task = Task(inst, targets, arm)
                for s in range(samples):
                    spec = Spec(
                        task_id=f"{family_name}-k{c.k}-{arm}-s{seed}-n{s}",
                        family=family_name,
                        seed=seed,
                        sample=s,
                        arm=arm,
                        k=c.k,
                        targets=targets,
                        residual_depth=task.residual_depth,
                        reference_loc=task.reference_loc,
                    )
                    out.append((spec, task))
    return out


def group_of(spec: Spec) -> str:
    """Which policy batch a task may safely share.

    A compositional prompt supplies the target's ancestors *with their reference
    proofs*. So if one agent holds two compositional tasks and one target is an
    ancestor of the other, the second prompt hands over the first answer. That
    is a real leak: during the first noninterference run, agents batched across
    conditions copied ancestor proofs out of compositional prompts into their
    monolithic answers, and said so in their reports.

    Grouping rule:

    *   Monolithic prompts contain no proofs at all, so they may share a batch
        freely.
    *   Compositional tasks are grouped by depth. Two rungs at equal depth
        cannot be ancestors of one another -- an ancestor has strictly smaller
        depth -- so a depth level is an antichain, and no prompt in the batch
        can contain another target's proof.

    `verify_groups` checks the property directly rather than trusting this
    argument.
    """
    if spec.arm == "monolithic":
        return "monolithic"
    return f"compositional-d{spec.k}"


def verify_groups(pairs: list[tuple[Spec, Task]]) -> list[str]:
    """Pre-flight: no prompt may contain a proof of another task's target.

    Returns a list of violations; empty means the batching is safe.
    """
    from collections import defaultdict

    by_group: dict[str, list[tuple[Spec, Task]]] = defaultdict(list)
    for spec, task in pairs:
        by_group[group_of(spec)].append((spec, task))

    problems: list[str] = []
    for group, members in by_group.items():
        targets = {t for spec, _ in members for t in spec.targets}
        for spec, task in members:
            leaked = targets & set(task.supplied)
            if leaked:
                problems.append(
                    f"{group}: {spec.task_id} supplies {sorted(leaked)}, "
                    f"which another task in the same batch must prove"
                )
    return problems


def cmd_export(args) -> None:
    RUNS.mkdir(exist_ok=True)
    for f in RUNS.glob("*"):
        f.unlink()
    builder = build_depth if args.mode == "depth" else build
    pairs = builder(args.family, args.seeds, args.samples)

    problems = verify_groups(pairs)
    if problems:
        print("UNSAFE BATCHING -- refusing to export:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)

    specs = []
    groups: dict[str, list[str]] = {}
    for spec, task in pairs:
        (RUNS / f"{spec.task_id}.txt").write_text(task.prompt())
        specs.append(asdict(spec))
        groups.setdefault(group_of(spec), []).append(spec.task_id)
    (RUNS / "_specs.json").write_text(json.dumps(specs, indent=2))
    (RUNS / "_groups.json").write_text(json.dumps(groups, indent=2))
    print(f"wrote {len(specs)} prompts to {RUNS}")
    print(f"batching verified safe across {len(groups)} groups:")
    for g, ids in sorted(groups.items()):
        print(f"  {g:<22} {len(ids)} tasks")


def cmd_ingest(args) -> None:
    specs = json.loads((RUNS / "_specs.json").read_text())
    family = FAMILIES[specs[0]["family"]]
    rows = []
    for s in specs:
        inst = Instance.sample(family, s["seed"])
        task = Task(inst, tuple(s["targets"]), s["arm"])
        out = RUNS / f"{s['task_id']}.out"
        if not out.exists():
            rows.append({**s, "verdict": "missing", "ok": False, "detail": ""})
            continue
        blocks = parse_blocks(out.read_text(), task)
        res = grade(task, blocks)
        rows.append({
            **s,
            "verdict": res.verdict.value,
            "ok": res.ok,
            "detail": res.detail[:160],
        })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"contrast-{args.label}.json").write_text(json.dumps(rows, indent=2))

    print(f"{'k':>3} {'arm':>10} {'resid_d':>8} {'LoC':>5} {'pass':>6}  n")
    for k in sorted({r["k"] for r in rows}):
        for arm in ("antichain", "chain"):
            sel = [r for r in rows if r["k"] == k and r["arm"] == arm]
            if not sel:
                continue
            rate = sum(r["ok"] for r in sel) / len(sel)
            print(f"{k:>3} {arm:>10} {sel[0]['residual_depth']:>8} "
                  f"{sel[0]['reference_loc']:>5} {rate:>6.2f}  {len(sel)}")
    bad = [r for r in rows if r["verdict"] not in ("proved", "missing")]
    if bad:
        print(f"\nfailure modes: "
              f"{ {v: sum(1 for r in bad if r['verdict'] == v) for v in {r['verdict'] for r in bad} } }")


def cmd_plan(args) -> None:
    """Print the exact agent assignments implied by the verified grouping.

    This exists because the grouping rule is easy to state, easy to check, and
    easy to violate by hand. During the first arithmetic run the grouping was
    verified correct and then overridden twice while spawning agents -- batches
    were merged to save agent count, which silently reintroduced the leak
    (`mul_assoc` is an ancestor of `pow_add`; `mul_succ_l` is an ancestor of
    `mul_comm`). Both merges put a lemma's proof in one prompt and the same
    lemma as a goal in another, inside a single policy context.

    So the runner emits the assignment rather than leaving it to judgment.
    """
    groups = json.loads((RUNS / "_groups.json").read_text())
    for name, ids in sorted(groups.items()):
        print(f"### agent: {name}  ({len(ids)} tasks)")
        for i in sorted(ids):
            print(f"  {i}.txt")
        print()
    print("One agent per block above. Do NOT merge blocks: a compositional "
          "prompt supplies its target's ancestors WITH proofs, so merging two "
          "compositional depths leaks the shallower answers.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--family", default="arithmetic")
    e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--samples", type=int, default=2)
    e.add_argument("--mode", default="contrast", choices=("contrast", "depth"))
    e.set_defaults(func=cmd_export)
    pl = sub.add_parser("plan")
    pl.set_defaults(func=cmd_plan)
    i = sub.add_parser("ingest")
    i.add_argument("--label", default="agent")
    i.set_defaults(func=cmd_ingest)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
