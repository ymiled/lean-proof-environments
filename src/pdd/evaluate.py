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
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .env import SPLITS, TaskSampler, all_families, split_families
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


def wilson(hits: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because the proportions here are
    near zero, where the normal interval famously runs negative and has poor
    coverage. At 4 successes in 192 the difference is the whole answer.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(
    groups: "list[list[float]]", reps: int = 2000, seed: int = 0
) -> "tuple[float, float]":
    """Percentile CI for a mean, resampling *tasks* rather than samples.

    Samples drawn from one task are not independent: they share a theory, a
    renaming seed and a withheld set, so a task that happens to be easy makes
    every one of its samples succeed together. Resampling individual samples
    would treat those as independent evidence and report an interval far too
    narrow. Resampling whole tasks keeps the correlation intact.
    """
    if not groups:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(groups)
    means: list[float] = []
    for _ in range(reps):
        picked = [groups[rng.randrange(n)] for _ in range(n)]
        flat = [v for g in picked for v in g]
        if flat:
            means.append(sum(flat) / len(flat))
    if not means:
        return (0.0, 0.0)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return (lo, hi)


@dataclass
class DepthResult:
    depth: int
    tasks: int = 0
    samples: int = 0
    #: Successes per task, in task order. Needed for the pass@k estimator.
    successes: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    #: The same scores, kept grouped by task so intervals can cluster on task.
    task_scores: list[list[float]] = field(default_factory=list)
    unparseable: int = 0

    @property
    def pass_at_1(self) -> float:
        return sum(self.successes) / self.samples if self.samples else 0.0

    @property
    def pass_at_1_ci(self) -> "tuple[float, float]":
        return wilson(sum(self.successes), self.samples)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def score_ci(self) -> "tuple[float, float]":
        return cluster_bootstrap(self.task_scores, seed=self.depth)

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

    @property
    def overall_score_ci(self) -> "tuple[float, float]":
        groups = [g for d in self.per_depth.values() for g in d.task_scores]
        return cluster_bootstrap(groups, seed=99)

    @property
    def overall_pass_at_1_ci(self) -> "tuple[float, float]":
        hits = sum(sum(d.successes) for d in self.per_depth.values())
        n = sum(d.samples for d in self.per_depth.values())
        return wilson(hits, n)

    def table(self) -> str:
        """Every rate carries an interval, because these rates are tiny.

        A pass@1 of 0.021 is four successes. Printed bare it invites a
        comparison it cannot support, so the interval is not decoration here.
        """
        lines = [f"{'depth':>6}{'tasks':>7}{'pass@1':>9}{'95% CI':>17}"
                 f"{'pass@' + str(self.group):>9}{'score':>9}{'95% CI':>17}"]
        for depth in sorted(self.per_depth):
            d = self.per_depth[depth]
            plo, phi = d.pass_at_1_ci
            slo, shi = d.score_ci
            lines.append(
                f"{depth:>6}{d.tasks:>7}{d.pass_at_1:>9.3f}"
                f"{f'[{plo:.3f},{phi:.3f}]':>17}"
                f"{d.pass_at(self.group, self.group):>9.3f}"
                f"{d.mean_score:>9.3f}{f'[{slo:.3f},{shi:.3f}]':>17}"
            )
        plo, phi = self.overall_pass_at_1_ci
        slo, shi = self.overall_score_ci
        lines.append(
            f"{'all':>6}{sum(d.tasks for d in self.per_depth.values()):>7}"
            f"{self.overall_pass_at_1:>9.3f}{f'[{plo:.3f},{phi:.3f}]':>17}"
            f"{'':>9}{self.overall_score:>9.3f}"
            f"{f'[{slo:.3f},{shi:.3f}]':>17}"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "group": self.group,
            "overall_pass_at_1": self.overall_pass_at_1,
            "overall_pass_at_1_ci": list(self.overall_pass_at_1_ci),
            "overall_score": self.overall_score,
            "overall_score_ci": list(self.overall_score_ci),
            "per_depth": {
                str(depth): {
                    "tasks": d.tasks,
                    "samples": d.samples,
                    "pass_at_1": d.pass_at_1,
                    "pass_at_1_ci": list(d.pass_at_1_ci),
                    f"pass_at_{self.group}": d.pass_at(self.group, self.group),
                    "mean_score": d.mean_score,
                    "mean_score_ci": list(d.score_ci),
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
        grouped: dict[int, list[float]] = defaultdict(list)
        for i, r in enumerate(rollouts):
            by_task[i // group] += int(r.shaped.binary)
            grouped[i // group].append(r.shaped.score)
            result.scores.append(r.shaped.score)
            result.unparseable += int(not r.shaped.parsed)
        result.successes = [by_task[i] for i in range(len(tasks))]
        result.task_scores = [grouped[i] for i in range(len(tasks))]
        per_depth[depth] = result

    return Report(per_depth=per_depth, group=group, label=label)


def merge(reports: "list[Report]", label: str = "") -> Report:
    """Pool several seeds into one report.

    Tasks are concatenated rather than averaged, so the intervals widen or
    narrow according to the total evidence instead of hiding between-seed
    spread. Two seeds that disagree produce a wide interval, which is the
    honest outcome.
    """
    if not reports:
        raise ValueError("nothing to merge")
    group = reports[0].group
    pooled: dict[int, DepthResult] = {}
    for rep in reports:
        for depth, d in rep.per_depth.items():
            acc = pooled.setdefault(depth, DepthResult(depth=depth))
            acc.tasks += d.tasks
            acc.samples += d.samples
            acc.successes.extend(d.successes)
            acc.scores.extend(d.scores)
            acc.task_scores.extend(d.task_scores)
            acc.unparseable += d.unparseable
    return Report(per_depth=pooled, group=group, label=label)


def evaluate_seeds(
    complete: Completer,
    families: dict,
    grader: Grader,
    *,
    seeds: "tuple[int, ...]" = (0,),
    label: str = "",
    **kwargs,
) -> Report:
    """Run `evaluate` once per seed and pool.

    One seed fixes one draw of tasks and one draw of renamings, and both are
    high-variance at the task counts that are affordable. A difference that
    survives only on seed 0 is not a difference.
    """
    reps = [
        evaluate(complete, families, grader, seed=s, label=f"{label}/seed{s}",
                 **kwargs)
        for s in seeds
    ]
    return merge(reps, label=label or f"{len(seeds)} seeds")


def evaluate_held_out(
    complete: Completer,
    grader: Grader,
    *,
    split: "tuple[str, ...] | str" = "binops",
    seeds: "tuple[int, ...] | None" = None,
    **kwargs,
) -> Report:
    """Evaluate on the theories `split_families` reserves.

    `split` selects which held-out set, and the choice decides what the number
    is evidence for. See `env.SPLITS`: `binops` is a robustness check, `theory`
    is the transfer claim.
    """
    _, held_out = split_families(all_families(include_corpus=True), split)
    if seeds is None:
        return evaluate(complete, held_out, grader, **kwargs)
    return evaluate_seeds(complete, held_out, grader, seeds=seeds, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="held-out",
                    choices=("held-out", "train", "all"))
    ap.add_argument("--held-out", default="binops", choices=sorted(SPLITS),
                    help="which theories to reserve. 'binops' varies one "
                         "parameter of a trained theory and is a robustness "
                         "check; 'theory' reserves vsi outright and is the "
                         "transfer claim")
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
    ap.add_argument("--seeds", type=int, default=1,
                    help="pool this many seeds starting at --seed. One seed "
                         "fixes one draw of tasks and renamings, and both are "
                         "high-variance at affordable task counts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from .policy import LocalPolicy

    families = all_families(include_corpus=True)
    train, held_out = split_families(families, args.held_out)
    chosen = {"held-out": held_out, "train": train, "all": families}[args.split]
    print(f"split={args.split} ({args.held_out}): {sorted(chosen)}")
    if args.split == "held-out" and args.held_out == "binops":
        print("note: ifc-b3* shares all 11 statements and 10 of 11 reference "
              "proofs with\n      the trained ifc-b1*. Use --held-out theory "
              "for a transfer claim.")

    policy = LocalPolicy(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    grader = Grader(workers=args.workers, config=RewardConfig())

    report = evaluate_seeds(
        policy.complete_many,
        chosen,
        grader,
        seeds=tuple(range(args.seed, args.seed + max(1, args.seeds))),
        depths=tuple(args.depths) if args.depths else None,
        volume=args.volume or None,
        tasks_per_depth=args.tasks_per_depth,
        group=args.group,
        format_example=not args.no_format_example,
        label=f"{policy.name} [{args.split}/{args.held_out}]",
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
