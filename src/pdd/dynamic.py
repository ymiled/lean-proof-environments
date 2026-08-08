"""Dynamic sampling: spend the batch on prompts that can still teach something.

The failure mode this addresses is the one `rollout.GroupReport` measures. A
group-relative estimator learns from the spread of rewards *within* one prompt's
rollouts, so a prompt every rollout fails and a prompt every rollout solves both
contribute exactly zero. In the first measured run most prompts were the former,
which is why 400 steps moved pass@1 from 0.005 to 0.021 and no further: most of
the compute bought no gradient.

DAPO's remedy is to oversample prompts, discard those whose accuracy came back 0
or 1, and refill the batch until it is made of prompts with spread. The
oversample-and-refill form assumes the trainer can ask for more prompts
mid-step. TRL cannot: its dataset is fixed when the trainer is constructed. So
the same policy is applied one level up, at the dataset:

*   Every graded batch updates a **ledger** of what each prompt did -- how often
    it was flat, and at what score.
*   The dataset is **rebuilt periodically** from that ledger, keeping prompts
    with observed spread, dropping ones observed to be hopeless or trivial, and
    reserving a share of every dataset for prompts never tried.

The reserved share is not a detail. A ledger-only policy converges on whatever
was informative early and never discovers that a formerly hopeless family became
learnable, which is precisely what a curriculum expects to happen.

Two departures from DAPO, both forced by the environment:

*   **Difficulty is read off the score, not the accuracy.** `shaped.score` is
    the fraction of a task's lemmas that the kernel accepted, so a prompt can be
    informative at 3/4 while its binary accuracy is a flat 0. Judging by
    accuracy would discard exactly the prompts the shaped reward exists to
    rescue.
*   **A flat *reward* is not conclusive.** `pdd.factored` extracts a gradient
    from groups whose totals match but whose per-target outcomes differ, so a
    prompt is only counted flat when its per-target outcomes are flat too.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .task import Task

#: What identifies a prompt: the theory, its renaming, and the withheld set.
#: Not the prompt string, which is long and would be duplicated per rollout.
PromptId = tuple[str, int, tuple[str, ...]]


def prompt_id(task: Task) -> PromptId:
    return (task.family.name, task.instance.seed, task.targets)


#: A prompt is too hard below this mean score, too easy above the other, and
#: worth spending rollouts on in between. The lower bound is not zero: a prompt
#: that has only ever scored 2% is one lucky `rfl` away from noise.
TOO_HARD = 0.02
TOO_EASY = 0.98


@dataclass
class PromptStat:
    """What one prompt has done, across every batch it has appeared in."""

    batches: int = 0
    #: Batches whose rollouts did not all produce the same per-target outcome.
    spread: int = 0
    rollouts: int = 0
    score_sum: float = 0.0
    solved: int = 0

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.rollouts if self.rollouts else 0.0

    @property
    def spread_rate(self) -> float:
        return self.spread / self.batches if self.batches else 0.0


@dataclass
class Ledger:
    """Per-prompt difficulty, accumulated over a run.

    Kept outside the trainer so it survives the trainer being rebuilt between
    cycles, which is the whole mechanism by which the dataset can change.
    """

    stats: dict[PromptId, PromptStat] = field(default_factory=dict)
    too_hard: float = TOO_HARD
    too_easy: float = TOO_EASY
    #: Batches a prompt must be seen in before its verdict is trusted. One
    #: group of eight is a small sample, but it is the sample DAPO uses, and
    #: raising this trades responsiveness for prompts that stay in the dataset
    #: producing nothing.
    min_batches: int = 1

    def observe(self, rollouts: "list", group_size: int) -> None:
        """Fold one graded batch into the ledger.

        `rollouts` is in TRL's batch order: `group_size` consecutive entries
        share a prompt. A chunk whose entries disagree about their prompt means
        that assumption broke, and it is skipped rather than recorded against
        the wrong prompt.
        """
        for i in range(0, len(rollouts), group_size):
            chunk = rollouts[i:i + group_size]
            if len(chunk) < 2:
                continue
            ids = {prompt_id(r.task) for r in chunk}
            if len(ids) != 1:
                continue
            stat = self.stats.setdefault(ids.pop(), PromptStat())
            stat.batches += 1
            stat.rollouts += len(chunk)
            stat.score_sum += sum(r.shaped.score for r in chunk)
            stat.solved += sum(1 for r in chunk if r.shaped.binary)
            # Per-target outcomes, not rewards: two rollouts that each prove one
            # lemma of two score the same and are still a usable contrast.
            outcomes = {
                tuple(sorted(r.shaped.per_target.items())) for r in chunk
            }
            if len(outcomes) > 1:
                stat.spread += 1

    def verdict(self, pid: PromptId) -> str:
        """One of `unknown`, `informative`, `hopeless`, `trivial`."""
        stat = self.stats.get(pid)
        if stat is None or stat.batches < self.min_batches:
            return "unknown"
        if stat.spread > 0:
            return "informative"
        if stat.mean_score >= self.too_easy:
            return "trivial"
        if stat.mean_score <= self.too_hard:
            return "hopeless"
        # Flat every time, but at a middling score: the policy has memorised a
        # partial answer. Still nothing to learn from until something changes.
        return "trivial"

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for pid in self.stats:
            v = self.verdict(pid)
            counts[v] = counts.get(v, 0) + 1
        return {"prompts": len(self.stats), **counts}


@dataclass
class Curated:
    """The result of one selection pass, and why it came out that way."""

    tasks: list[Task]
    kept_informative: int = 0
    kept_unknown: int = 0
    kept_fallback: int = 0
    pool: int = 0
    retired: int = 0
    added: int = 0
    oversized: int = 0

    def __str__(self) -> str:
        return (
            f"{len(self.tasks)} prompts ({self.kept_informative} "
            f"known-informative, {self.kept_unknown} unexplored, "
            f"{self.kept_fallback} fallback) from a pool of {self.pool}; "
            f"retired {self.retired}, added {self.added}, "
            f"{self.oversized} oversized"
        )


@dataclass
class PromptPool:
    """A persistent set of candidate prompts, pruned by what the ledger learns.

    A pool is what makes the ledger actionable, and it is not optional. The
    corpus admits thousands of distinct prompts per depth, so a policy that
    redraws at random every cycle revisits a given prompt roughly never: by the
    time the ledger has an opinion about a prompt, that prompt is gone, and
    filtering has no effect on anything. This was measured, not assumed --
    against the generated corpus, two consecutive random draws of 32 prompts
    from 192 candidates shared none at all.

    So prompts are drawn once into a pool and reused across cycles. Each cycle:

    1.  Prompts the ledger has judged hopeless or trivial are **retired**.
        Their slots are the compute that dynamic sampling reclaims.
    2.  The pool is **topped back up** from the sampler, which is where new
        difficulty enters -- and the only way a curriculum can advance.
    3.  A dataset is **selected** from the pool, informative prompts first,
        with a reserved share for prompts nobody has tried yet.

    The pool is tied to one stage's `(depths, volume)`. When those change the
    pool is stale by construction -- the ledger's verdicts still apply to the
    prompts, but the prompts themselves are the wrong difficulty -- so
    `reconfigure` clears it. The ledger is not cleared: a prompt reappearing
    under a later stage keeps its history.
    """

    ledger: Ledger = field(default_factory=Ledger)
    #: Live candidates, in insertion order.
    tasks: dict[PromptId, Task] = field(default_factory=dict)
    #: The `(depths, volume)` these candidates were drawn for.
    signature: object = None
    #: Never retire below this many prompts, whatever the ledger says. A pool
    #: emptied by an unlucky early stretch cannot refill from itself.
    floor: int = 32
    retired_total: int = 0

    def reconfigure(self, signature) -> bool:
        """Point the pool at a new difficulty. Returns whether it was cleared."""
        if signature == self.signature:
            return False
        self.tasks.clear()
        self.signature = signature
        return True

    def retire(self) -> int:
        """Drop prompts the ledger has judged not worth sampling from."""
        doomed = [
            pid for pid in self.tasks
            if self.ledger.verdict(pid) in ("hopeless", "trivial")
        ]
        # Retire the worst-established first when the floor binds, so a prompt
        # seen once and failed outlives nothing it should outlive.
        allowed = max(0, len(self.tasks) - self.floor)
        doomed = doomed[:allowed]
        for pid in doomed:
            del self.tasks[pid]
        self.retired_total += len(doomed)
        return len(doomed)

    def top_up(
        self,
        sampler,
        *,
        size: int,
        tokenizer,
        max_prompt_tokens: int,
        format_example: bool = True,
        max_draws: "int | None" = None,
    ) -> tuple[int, int]:
        """Refill to `size` candidates. Returns (added, oversized).

        Drawing a task is cheap -- it is constructed from an already-loaded
        family -- so the real cost here is tokenizing each new prompt to check
        it against the context limit. That is paid once per prompt for as long
        as the prompt stays in the pool, rather than once per cycle.
        """
        added = 0
        oversized = 0
        draws = 0
        budget = max_draws if max_draws is not None else max(size * 20, 200)
        while len(self.tasks) < size and draws < budget:
            draws += 1
            task = sampler.sample()
            pid = prompt_id(task)
            if pid in self.tasks:
                continue
            prompt = task.prompt(format_example=format_example)
            if len(tokenizer(prompt)["input_ids"]) > max_prompt_tokens:
                oversized += 1
                continue
            self.tasks[pid] = task
            added += 1
        return added, oversized

    def select(
        self,
        count: int,
        *,
        explore: float = 0.35,
        rng: "random.Random | None" = None,
    ) -> Curated:
        """Choose `count` prompts from the pool for one cycle's dataset.

        `explore` reserves a share for prompts the ledger has never seen.
        Without it a curriculum cannot advance: every prompt at a new depth
        starts unknown, so a purely exploitative rule would never draw one and
        the ledger would never learn that the depth had become reachable.

        Falls back to prompts already judged flat rather than returning a short
        dataset. A stage whose pool has been mostly retired is better served by
        a full batch of hard prompts than by forty easy ones repeated.
        """
        rng = rng or random.Random(0)
        informative: list[Task] = []
        unknown: list[Task] = []
        fallback: list[Task] = []
        for pid, task in self.tasks.items():
            verdict = self.ledger.verdict(pid)
            if verdict == "informative":
                informative.append(task)
            elif verdict == "unknown":
                unknown.append(task)
            else:
                fallback.append(task)

        rng.shuffle(informative)
        rng.shuffle(unknown)
        rng.shuffle(fallback)

        quota = int(count * explore)
        chosen = unknown[:quota]
        n_unknown = len(chosen)
        chosen += informative[:count - len(chosen)]
        n_informative = len(chosen) - n_unknown
        if len(chosen) < count:
            extra = unknown[n_unknown:n_unknown + count - len(chosen)]
            chosen += extra
            n_unknown += len(extra)
        n_fallback = 0
        if len(chosen) < count:
            extra = fallback[:count - len(chosen)]
            chosen += extra
            n_fallback = len(extra)

        rng.shuffle(chosen)
        return Curated(
            tasks=chosen,
            kept_informative=n_informative,
            kept_unknown=n_unknown,
            kept_fallback=n_fallback,
            pool=len(self.tasks),
        )


def curate(
    sampler,
    pool: PromptPool,
    *,
    count: int,
    tokenizer,
    max_prompt_tokens: int,
    format_example: bool = True,
    oversample: int = 6,
    explore: float = 0.35,
    rng: "random.Random | None" = None,
) -> Curated:
    """One cycle of retire, top up, select. See `PromptPool`.

    `oversample` sets the pool size as a multiple of the dataset size. It is the
    slack dynamic sampling has to work with: at 1 the pool is exactly the
    dataset and retiring a prompt can only be answered with a fresh unknown
    one, which is random sampling with extra steps.
    """
    retired = pool.retire()
    added, oversized = pool.top_up(
        sampler,
        size=max(count * oversample, count),
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        format_example=format_example,
    )
    result = pool.select(count, explore=explore, rng=rng)
    result.retired = retired
    result.added = added
    result.oversized = oversized
    return result
