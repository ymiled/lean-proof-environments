"""Batched, parallel, deduplicated grading -- the thing that makes RL affordable.

Measured on the development machine, one `grade` call on a depth-5 VSI task
takes 2.75s when the proof checks and 1.24s when it does not. `grade_partial`
costs one such call per target. A GRPO step of 16 prompts x 8 rollouts over
volume-4 tasks is therefore 16 * 8 * 4 = 512 Lean invocations, or roughly two
hours if run one at a time. That is not a training loop, it is a batch job.

Three things bring it back to something usable:

*   **Parallelism at the target level, not the rollout level.** Each target of
    each rollout is an independent `lean` subprocess, so the natural unit of
    work is the (rollout, target) pair. Pooling at that granularity keeps every
    worker busy even when a batch contains few distinct tasks.

*   **Threads rather than processes.** `grade` spends all of its time inside
    `subprocess.run`, which releases the GIL, so threads give full parallelism
    without pickling a `Task` -- and a `Task` carries its family's entire Lean
    source, which is not something to copy 512 times per step.

*   **Deduplication.** Rollouts sampled from one prompt collide constantly,
    especially early in training when the policy is nearly deterministic and
    emits `simp` for everything. Two rollouts proposing the same block for the
    same target are the same Lean invocation, and running it twice buys nothing.

The cache key is content-addressed on (family, renaming seed, target set,
target, block text), which is exactly what determines the file that gets
compiled. It is deliberately not keyed on the whole response: two responses that
differ in one target still share the other three.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .grader import BANNED, Verdict, grade
from .reward import DEFAULT, RewardConfig, Shaped, combine, unparseable
from .task import Task, parse_blocks

#: (family, seed, targets, target, block) -> Verdict
CacheKey = tuple[str, int, tuple[str, ...], str, str]


def cache_key(task: Task, target: str, block: str) -> CacheKey:
    return (
        task.family.name,
        task.instance.seed,
        task.targets,
        target,
        block,
    )


@dataclass
class Stats:
    """Where the wall clock went. Print this; it is how the loop gets tuned."""

    tasks: int = 0
    jobs: int = 0
    cache_hits: int = 0
    lean_calls: int = 0
    seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.jobs if self.jobs else 0.0

    @property
    def calls_per_second(self) -> float:
        return self.lean_calls / self.seconds if self.seconds else 0.0

    def __str__(self) -> str:
        return (
            f"{self.tasks} tasks, {self.jobs} target-jobs, "
            f"{self.lean_calls} lean calls "
            f"({self.hit_rate:.0%} cached), {self.seconds:.1f}s "
            f"({self.calls_per_second:.1f} calls/s)"
        )


@dataclass
class Rollout:
    """One sampled response against one task, with its reward."""

    task: Task
    text: str
    blocks: dict[str, str]
    shaped: Shaped

    @property
    def reward(self) -> float:
        return self.shaped.reward

    @property
    def info(self) -> dict:
        return {
            **self.shaped.info,
            "depth": self.task.depth,
            "key": self.task.key,
            "volume": self.task.volume,
            "residual_depth": self.task.residual_depth,
            "family": self.task.family.name,
            "seed": self.task.instance.seed,
            "condition": self.task.label,
        }


class Grader:
    """A reusable, thread-pooled, caching grader over (task, response) pairs.

    Hold one of these for the lifetime of a training run. The cache persists
    across steps on purpose: a policy that has learned a proof re-emits it, and
    re-verifying a proof already verified this run tells nobody anything.
    """

    def __init__(
        self,
        workers: int = 16,
        config: RewardConfig = DEFAULT,
        timeout: float = 120.0,
        cache_size: int = 200_000,
    ) -> None:
        self.workers = workers
        self.config = config
        self.timeout = timeout
        self.cache_size = cache_size
        self._cache: "OrderedDict[CacheKey, Verdict]" = OrderedDict()
        self.stats = Stats()
        # `stats.lean_calls` is the one counter written from worker threads,
        # and `+=` is three bytecodes, not one.
        self._lock = threading.Lock()

    # -- cache -------------------------------------------------------------

    def _get(self, key: CacheKey) -> "Verdict | None":
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
        return hit

    def _put(self, key: CacheKey, verdict: Verdict) -> None:
        self._cache[key] = verdict
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # -- grading -----------------------------------------------------------

    def _verdict(self, task: Task, target: str, block: str) -> Verdict:
        """One target, judged in a file where every other target is reference.

        Isolating like this is what stops one target's compile error from
        aborting elaboration and denying every other target a verdict. It is the
        same construction `grader.grade_partial` uses, restated here so the work
        can be scheduled as an independent job.
        """
        lowered = block.lower()
        if any(tok in lowered for tok in BANNED):
            return Verdict.BANNED_SYNTAX
        mixed = {**task.reference_solution(), target: block}
        with self._lock:
            self.stats.lean_calls += 1
        return grade(task, mixed, timeout=self.timeout).verdict

    def grade_batch(
        self, tasks: "list[Task]", responses: "list[str]"
    ) -> list[Rollout]:
        """Grade a whole batch of responses in one pooled pass.

        `tasks` and `responses` are parallel lists; the same task may appear
        many times, which is the normal case for a group-relative method.
        """
        if len(tasks) != len(responses):
            raise ValueError("tasks and responses must be the same length")

        start = time.perf_counter()
        parsed = [parse_blocks(text, task)
                  for task, text in zip(tasks, responses)]

        # Collect every distinct (task, target, block) the batch needs.
        jobs: dict[CacheKey, tuple[Task, str, str]] = {}
        attempted: list[set[str]] = []
        for task, blocks in zip(tasks, parsed):
            live = {k for k, v in blocks.items() if v and v.strip()}
            attempted.append(live)
            for target in live:
                key = cache_key(task, target, blocks[target])
                self.stats.jobs += 1
                if self._get(key) is not None:
                    self.stats.cache_hits += 1
                elif key not in jobs:
                    jobs[key] = (task, target, blocks[target])

        if jobs:
            keys = list(jobs)
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                verdicts = pool.map(lambda k: self._verdict(*jobs[k]), keys)
                for key, verdict in zip(keys, verdicts):
                    self._put(key, verdict)

        out: list[Rollout] = []
        for task, blocks, live, text in zip(tasks, parsed, attempted, responses):
            if not live:
                shaped = unparseable(task, self.config)
            else:
                per = {
                    t: self._cache[cache_key(task, t, blocks[t])]
                    for t in live
                }
                # Targets with no block never reach the kernel; `combine`
                # scores them as missing and never reads their verdict.
                per.update({t: Verdict.COMPILE_ERROR
                            for t in task.targets if t not in live})
                shaped = combine(task, per, live, self.config)
            out.append(Rollout(task=task, text=text, blocks=blocks, shaped=shaped))

        self.stats.tasks += len(tasks)
        self.stats.seconds += time.perf_counter() - start
        return out

    def reward_fn(self, tasks: "list[Task]", responses: "list[str]") -> list[float]:
        """The shape a trainer wants: rewards only, same order as the inputs."""
        return [r.reward for r in self.grade_batch(tasks, responses)]


# -- diagnostics -----------------------------------------------------------


@dataclass
class GroupReport:
    """Whether a rollout group carries any learning signal at all.

    This is the number to watch. A group whose rewards are all equal has zero
    advantage under any group-relative estimator, so it contributes nothing to
    the update regardless of how the rewards were computed. If `degenerate` sits
    near 1.0 the run is not training, whatever the loss curve says.
    """

    groups: int = 0
    degenerate: int = 0
    rewards: list[float] = field(default_factory=list)

    @property
    def degenerate_rate(self) -> float:
        return self.degenerate / self.groups if self.groups else 0.0

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def __str__(self) -> str:
        return (
            f"mean reward {self.mean_reward:+.3f}, "
            f"{self.degenerate_rate:.0%} of {self.groups} groups degenerate"
        )


def group_report(rollouts: "list[Rollout]", group_size: int,
                 tol: float = 1e-9) -> GroupReport:
    """Split `rollouts` into consecutive groups and check for flat rewards."""
    rep = GroupReport()
    for i in range(0, len(rollouts), group_size):
        group = rollouts[i:i + group_size]
        if len(group) < 2:
            continue
        rewards = [r.reward for r in group]
        rep.groups += 1
        rep.rewards.extend(rewards)
        if max(rewards) - min(rewards) <= tol:
            rep.degenerate += 1
    return rep
