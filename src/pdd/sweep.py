"""Experiment driver: pass@k against dependency depth, in both conditions.

Usage:

    uv run python -m pdd.sweep --policy reference --k 1
    uv run python -m pdd.sweep --policy claude-opus-5 --k 8 --seeds 5

Results land in `results/<policy>.json` for `plot.py` to consume.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .grader import grade
from .ladder import BY_KEY, Instance, depth_of
from .policy import AnthropicPolicy, EmptyPolicy, Policy, ReferencePolicy
from .task import Condition, Task

RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Attempt:
    seed: int
    key: str
    depth: int
    condition: str
    sample: int
    verdict: str
    detail: str
    ok: bool


def _one(policy: Policy, task: Task, seed: int, sample: int) -> Attempt:
    try:
        block = policy.act(task)
        result = grade(task, block)
        verdict, detail, ok = result.verdict.value, result.detail, result.ok
    except Exception as exc:  # a policy failure is data, not a crash
        verdict, detail, ok = "policy_error", str(exc)[:200], False
    return Attempt(
        seed=seed,
        key=task.key,
        depth=task.depth,
        condition=task.condition.value,
        sample=sample,
        verdict=verdict,
        detail=detail,
        ok=ok,
    )


def sweep(
    policy: Policy,
    k: int = 1,
    seeds: int = 1,
    workers: int = 4,
) -> list[Attempt]:
    jobs: list[tuple[Task, int, int]] = []
    for s in range(seeds):
        inst = Instance.sample(s)
        for condition in Condition:
            for key in BY_KEY:
                task = Task(inst, key, condition)
                jobs.extend((task, s, i) for i in range(k))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: _one(policy, *j), jobs))


def summarize(attempts: list[Attempt]) -> dict:
    """pass@k per (condition, depth), pooling seeds and rungs at that depth."""
    buckets: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for a in attempts:
        buckets[(a.condition, a.depth)].append(a.ok)

    # pass@k here is "any sample succeeded", grouped per (seed, key).
    per_task: dict[tuple[str, int, int, str], list[bool]] = defaultdict(list)
    for a in attempts:
        per_task[(a.condition, a.depth, a.seed, a.key)].append(a.ok)

    passk: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for (cond, depth, _, _), oks in per_task.items():
        passk[(cond, depth)].append(any(oks))

    return {
        f"{cond}:{depth}": {
            "pass_at_k": sum(v) / len(v),
            "per_sample": sum(buckets[(cond, depth)]) / len(buckets[(cond, depth)]),
            "n_tasks": len(v),
            "n_samples": len(buckets[(cond, depth)]),
        }
        for (cond, depth), v in sorted(passk.items())
    }


def _build(name: str) -> Policy:
    if name == "reference":
        return ReferencePolicy()
    if name == "empty":
        return EmptyPolicy()
    return AnthropicPolicy(model=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="reference")
    ap.add_argument("--k", type=int, default=1, help="samples per task")
    ap.add_argument("--seeds", type=int, default=1, help="renamed instances")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    policy = _build(args.policy)
    started = time.time()
    attempts = sweep(policy, k=args.k, seeds=args.seeds, workers=args.workers)
    elapsed = time.time() - started

    summary = summarize(attempts)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{policy.name.replace('/', '_')}.json"
    out.write_text(
        json.dumps(
            {
                "policy": policy.name,
                "k": args.k,
                "seeds": args.seeds,
                "elapsed_s": round(elapsed, 1),
                "summary": summary,
                "attempts": [asdict(a) for a in attempts],
            },
            indent=2,
        )
    )

    depths = sorted({depth_of(k) for k in BY_KEY})
    print(f"policy={policy.name}  k={args.k}  seeds={args.seeds}  {elapsed:.0f}s")
    print(f"{'depth':>6}  {'monolithic':>12}  {'compositional':>14}")
    for d in depths:
        m = summary.get(f"monolithic:{d}", {}).get("pass_at_k")
        c = summary.get(f"compositional:{d}", {}).get("pass_at_k")
        fm = "n/a" if m is None else f"{m:.2f}"
        fc = "n/a" if c is None else f"{c:.2f}"
        print(f"{d:>6}  {fm:>12}  {fc:>14}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
