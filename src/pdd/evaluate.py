"""Held-out evaluation, broken down by depth.

Training reward is not evidence. It is measured on the theories being trained
on, at whatever depth the curriculum currently allows, and it rises when the
policy memorises the corpus exactly as readily as when it learns to prove. The
number that means something is pass rate on theories the run never saw, reported
per depth -- which is the same curve the benchmark reports, so a trained policy
drops straight onto the existing plots.

Two quantities per depth, and they answer different questions:

*   **pass@1** -- the probability a single sample discharges the whole task.
    Comparable to `sweep.py` and to the paper.
*   **mean score** -- the fraction of the task's targets proved. This is the
    quantity the reward-granularity result is about, and it moves long before
    pass@1 does, which makes it the useful early signal.

`pass@k` is reported with the unbiased estimator rather than "did any of the k
samples succeed", because the naive version is biased upward and the bias grows
with k, so it would flatter a run that simply sampled more.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .env import TaskSampler, all_families, split_families
from .reward import RewardConfig
from .rollout import Grader, Rollout
from .task import Task

RESULTS = Path(__file__).resolve().parents[2] / "results"


class Completer(Protocol):
    """Anything that turns prompts into completions, in order."""

    def __call__(self, prompts: "list[str]") -> "list[str]": ...


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k), for c successes out of n samples.

    Computed in the product form rather than with factorials, which overflow and
    lose precision for no benefit here.
    """
    if n - c < k:
        return 1.0
    out = 1.0
    for i in range(n - c + 1, n + 1):
        out *= 1.0 - k / i
    return 1.0 - out


@dataclass
class DepthResult:
    depth: int
    tasks: int = 0
    samples: int = 0
    #: Successes per task, in task order. Needed for the pass@k estimator.
    successes: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    unparseable: int = 0

    @property
    def pass_at_1(self) -> float:
        return sum(self.successes) / self.samples if self.samples else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def pass_at(self, k: int, per_task: int) -> float:
        if not self.successes:
            return 0.0
        return sum(
            pass_at_k(per_task, c, k) for c in self.successes
        ) / len(self.successes)


@dataclass
class Report:
    per_depth: dict[int, DepthResult]
    group: int
    label: str = ""

    @property
    def overall_score(self) -> float:
        scores = [s for d in self.per_depth.values() for s in d.scores]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def overall_pass_at_1(self) -> float:
        hits = sum(sum(d.successes) for d in self.per_depth.values())
        n = sum(d.samples for d in self.per_depth.values())
        return hits / n if n else 0.0

    def table(self) -> str:
        lines = [f"{'depth':>6}{'tasks':>7}{'pass@1':>9}"
                 f"{'pass@' + str(self.group):>9}{'score':>9}"]
        for depth in sorted(self.per_depth):
            d = self.per_depth[depth]
            lines.append(
                f"{depth:>6}{d.tasks:>7}{d.pass_at_1:>9.3f}"
                f"{d.pass_at(self.group, self.group):>9.3f}"
                f"{d.mean_score:>9.3f}"
            )
        lines.append(
            f"{'all':>6}{sum(d.tasks for d in self.per_depth.values()):>7}"
            f"{self.overall_pass_at_1:>9.3f}{'':>9}{self.overall_score:>9.3f}"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "group": self.group,
            "overall_pass_at_1": self.overall_pass_at_1,
            "overall_score": self.overall_score,
            "per_depth": {
                str(depth): {
                    "tasks": d.tasks,
                    "samples": d.samples,
                    "pass_at_1": d.pass_at_1,
                    f"pass_at_{self.group}": d.pass_at(self.group, self.group),
                    "mean_score": d.mean_score,
                    "unparseable": d.unparseable,
                }
                for depth, d in sorted(self.per_depth.items())
            },
        }


def evaluate(
    complete: Completer,
    families: dict,
    grader: Grader,
    *,
    depths: "tuple[int, ...] | None" = None,
    volume: int | None = 3,
    tasks_per_depth: int = 16,
    group: int = 4,
    seed: int = 0,
    format_example: bool = True,
    label: str = "",
) -> Report:
    """Sample tasks at each depth, complete them, grade them, tabulate.

    Held constant across depths: the number of tasks and the number of samples
    per task. Depth then varies alone, which is the whole point of the
    breakdown -- an evaluation that drew fewer deep tasks would confound the
    depth effect with sample size exactly where the curve matters most.
    """
    available = sorted({d for f in families.values() for d in f.depths})
    wanted = [d for d in available if depths is None or d in depths]

    per_depth: dict[int, DepthResult] = {}
    for depth in wanted:
        sampler = TaskSampler(
            families=families,
            depths=(depth,),
            volume=volume,
            rng=random.Random(seed + depth),
        )
        try:
            tasks: list[Task] = sampler.batch(tasks_per_depth)
        except ValueError:
            continue  # no family reaches this depth

        flat = [t for t in tasks for _ in range(group)]
        prompts = [t.prompt(format_example=format_example) for t in flat]
        rollouts: list[Rollout] = grader.grade_batch(flat, complete(prompts))

        result = DepthResult(depth=depth, tasks=len(tasks),
                             samples=len(rollouts))
        by_task: dict[int, int] = defaultdict(int)
        for i, r in enumerate(rollouts):
            by_task[i // group] += int(r.shaped.binary)
            result.scores.append(r.shaped.score)
            result.unparseable += int(not r.shaped.parsed)
        result.successes = [by_task[i] for i in range(len(tasks))]
        per_depth[depth] = result

    return Report(per_depth=per_depth, group=group, label=label)


def evaluate_held_out(
    complete: Completer,
    grader: Grader,
    **kwargs,
) -> Report:
    """Convenience wrapper: evaluate on the theories `split_families` reserves."""
    _, held_out = split_families(all_families(include_corpus=True))
    return evaluate(complete, held_out, grader, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="held-out",
                    choices=("held-out", "train", "all"))
    ap.add_argument("--depths", type=int, nargs="*", default=None)
    ap.add_argument("--volume", type=int, default=3)
    ap.add_argument("--tasks-per-depth", type=int, default=16)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-format-example", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from .policy import LocalPolicy

    families = all_families(include_corpus=True)
    train, held_out = split_families(families)
    chosen = {"held-out": held_out, "train": train, "all": families}[args.split]
    print(f"split={args.split}: {sorted(chosen)}")

    policy = LocalPolicy(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    grader = Grader(workers=args.workers, config=RewardConfig())

    report = evaluate(
        policy.complete_many,
        chosen,
        grader,
        depths=tuple(args.depths) if args.depths else None,
        volume=args.volume or None,
        tasks_per_depth=args.tasks_per_depth,
        group=args.group,
        seed=args.seed,
        format_example=not args.no_format_example,
        label=f"{policy.name} [{args.split}]",
    )
    print()
    print(report.table())
    print()
    print(grader.stats)

    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = RESULTS / path
        path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
