"""Gym-style interface over the ladder, plus the task sampler a learner needs.

The sweep in `sweep.py` does not use any of this -- it enumerates tasks and
grades one sample each. This module exists so the benchmark is actually
trainable rather than nominally trainable, which turned out to need three things
beyond `reset`/`step`:

1.  **A reward with structure.** See `reward.py`. Binary task reward produces
    all-zero rollout groups for a small model and therefore no gradient.

2.  **Control over task volume.** A single-target task scores in `{0, 1}`
    however it is graded, so per-target credit buys nothing there. Withholding
    several lemmas at once makes the reward genuinely graded. `TaskSampler`
    draws up-closed withheld sets of a requested size, which is the same
    construction `design.py` uses for its chain/antichain contrasts and the same
    one `corpus.task_shapes` counts.

3.  **A curriculum.** Depth is the difficulty axis this repository measures, so
    it is the obvious axis to schedule along. `TaskSampler.depths` is mutable at
    run time for exactly that.

Episodes remain single-step. One tactic per action with intermediate proof
states as observations needs a persistent Lean server rather than one-shot
compilation; that is the next version, not this one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .families import DEFAULT, FAMILIES
from .ladder import Family, Instance
from .reward import RewardConfig, Shaped, binary_reward, shaped_reward
from .task import Condition, Task

#: Generated theories reserved for evaluation, by name prefix. Holding out a
#: whole bucket keeps the split structural rather than cosmetic.
HELD_OUT_PREFIXES = ("ifc-b3",)


def all_families(include_corpus: bool = True) -> dict[str, Family]:
    """Hand-written families plus every generated theory with a real graph.

    Generated theories are skipped silently when their extraction has not been
    run; `corpus.load_generated` already enforces that a loaded graph came from
    necessity extraction rather than declaration.
    """
    out: dict[str, Family] = dict(FAMILIES)
    if include_corpus:
        from .corpus import load_generated

        for fam in load_generated():
            out[fam.name] = fam
    return out


def split_families(
    families: dict[str, Family], held_out: tuple[str, ...] = HELD_OUT_PREFIXES
) -> tuple[dict[str, Family], dict[str, Family]]:
    """Partition by name prefix into (train, eval).

    Splitting by *theory* rather than by task keeps the two sides structurally
    independent. Splitting by task would leak: two withheld sets over one theory
    share their definitions, their supplied lemmas and most of their proofs.
    """
    train: dict[str, Family] = {}
    evalset: dict[str, Family] = {}
    for name, fam in families.items():
        (evalset if name.startswith(held_out) else train)[name] = fam
    return train, evalset


# -- task construction -----------------------------------------------------


def renders_cleanly(family: Family, withheld: "set[str]") -> bool:
    """Whether a withheld set produces a file that compiles at all.

    Only the ancestors of the withheld lemmas are rendered; anything above them
    is absent from the file entirely and so constrains nothing. Of what *is*
    rendered, every supplied lemma arrives carrying its own reference proof,
    which cites its own dependencies by name -- so no supplied lemma may depend
    on a withheld one, or its proof references a declaration that is not there.

    This is the same admissibility test `corpus.task_shapes` counts over and
    `design.is_up_closed` applies to whole families. Restricting it to the
    rendered set is what makes antichains admissible: three independent depth-1
    lemmas have no rendered ancestors at all, so nothing can be broken by
    withholding them, even though lemmas elsewhere in the DAG depend on them.
    """
    supplied: set[str] = set()
    for target in withheld:
        supplied |= set(family.transitive_deps(target))
    supplied -= withheld
    return all(
        not (withheld & set(family.by_key[s].deps)) for s in supplied
    )


def sample_targets(
    family: Family, key: str, volume: int, rng: random.Random
) -> tuple[str, ...]:
    """An admissible withheld set of up to `volume` lemmas, topped by `key`.

    Candidates are capped at `key`'s depth so the task's nominal depth stays
    what the caller asked for. Beyond that the growth is greedy over a shuffled
    candidate list, keeping any addition that leaves the set admissible: at
    depth 1 that yields antichains, deeper it yields a mix of chains and
    antichains, which is the right default when the goal is a dense reward
    rather than a controlled contrast. `design.find_contrasts` is where to go
    for the controlled version.

    The set can come back smaller than `volume` when the family is too thin to
    supply that many admissible lemmas. That is reported honestly through
    `Task.volume` rather than papered over.
    """
    ceiling = family.depth_of(key)
    chosen = {key}
    candidates = [
        k for k in family.by_key
        if k != key and family.depth_of(k) <= ceiling
    ]
    rng.shuffle(candidates)

    for cand in candidates:
        if len(chosen) >= volume:
            break
        if renders_cleanly(family, chosen | {cand}):
            chosen.add(cand)

    return tuple(sorted(chosen, key=family.depth_of))


@dataclass
class TaskSampler:
    """Draws tasks from a set of families under depth and volume constraints.

    `depths` and `volume` are plain mutable attributes so a training loop can
    advance a curriculum without rebuilding the sampler and resetting its RNG.
    """

    families: dict[str, Family]
    #: Allowed depths for the deepest target. `None` means every depth.
    depths: tuple[int, ...] | None = None
    #: How many lemmas to withhold. 1 reproduces the compositional condition;
    #: `None` reproduces the monolithic one (the whole ancestor cone).
    volume: int | None = 1
    #: Renaming seeds to draw from. `vsi` ignores renaming -- it cannot be
    #: renamed without breaking its own imports -- so it contributes one
    #: instance regardless of this.
    seeds: int = 64
    rng: random.Random = field(default_factory=random.Random)

    def eligible(self, fam: Family) -> list[str]:
        return [
            k
            for k in fam.by_key
            if self.depths is None or fam.depth_of(k) in self.depths
        ]

    def _label(self) -> str:
        if self.volume is None:
            return Condition.MONOLITHIC.value
        if self.volume == 1:
            return Condition.COMPOSITIONAL.value
        return f"volume-{self.volume}"

    def sample(self) -> Task:
        """One task. Tries families in random order until a depth constraint fits."""
        names = list(self.families)
        self.rng.shuffle(names)
        for name in names:
            fam = self.families[name]
            keys = self.eligible(fam)
            if not keys:
                continue
            inst = Instance.sample(fam, self.rng.randrange(self.seeds))
            key = self.rng.choice(keys)
            if self.volume is None:
                targets: tuple[str, ...] = (*fam.transitive_deps(key), key)
            else:
                targets = sample_targets(fam, key, self.volume, self.rng)
            return Task(inst, targets, self._label())
        available = sorted({d for f in self.families.values() for d in f.depths})
        raise ValueError(
            f"no family has a rung at depth {self.depths}; available: {available}"
        )

    def batch(self, n: int) -> list[Task]:
        return [self.sample() for _ in range(n)]


# -- environment -----------------------------------------------------------


@dataclass
class Step:
    observation: str
    reward: float
    terminated: bool
    info: dict


class LadderEnv:
    """One episode = one attempt at one withheld set.

    `reward_mode="shaped"` is the default and is what a learner should use.
    `"binary"` is kept as the control arm: a run that only ever trains under
    shaping has not shown that the shaping was what made it work.
    """

    def __init__(
        self,
        condition: Condition = Condition.COMPOSITIONAL,
        family: str | None = DEFAULT,
        depths: tuple[int, ...] | None = None,
        seed: int = 0,
        *,
        volume: int | None = -1,
        reward_mode: str = "shaped",
        reward_config: RewardConfig | None = None,
        include_corpus: bool = False,
        timeout: float = 120.0,
    ) -> None:
        if reward_mode not in ("shaped", "binary"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")

        pool = all_families(include_corpus)
        self.families = pool if family is None else {family: pool[family]}
        self.condition = condition
        self.reward_mode = reward_mode
        self.reward_config = reward_config or RewardConfig()
        self.timeout = timeout

        # `-1` is the sentinel for "derive from the condition"; `None` is a
        # meaningful value meaning "withhold the whole cone".
        if volume == -1:
            volume = None if condition is Condition.MONOLITHIC else 1
        self.sampler = TaskSampler(
            families=self.families,
            depths=depths,
            volume=volume,
            rng=random.Random(seed),
        )
        self.task: Task | None = None

    @property
    def family(self) -> Family:
        """The single family, for callers predating multi-family sampling."""
        return next(iter(self.families.values()))

    @property
    def depths(self) -> tuple[int, ...] | None:
        return self.sampler.depths

    def set_depths(self, depths: tuple[int, ...] | None) -> None:
        """Advance the curriculum without disturbing the sampler's RNG."""
        self.sampler.depths = depths

    def set_volume(self, volume: int | None) -> None:
        self.sampler.volume = volume

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            self.sampler.rng.seed(seed)
        self.task = self.sampler.sample()
        return self.task.prompt()

    def evaluate(self, blocks: "dict[str, str] | str") -> Shaped:
        """Reward for a response against the current task, without stepping."""
        if self.task is None:
            raise RuntimeError("call reset() before evaluate()")
        if self.reward_mode == "binary":
            return binary_reward(self.task, blocks, timeout=self.timeout)
        return shaped_reward(
            self.task, blocks, self.reward_config, timeout=self.timeout
        )

    def step(self, tactic_block: "dict[str, str] | str") -> Step:
        if self.task is None:
            raise RuntimeError("call reset() before step()")
        shaped = self.evaluate(tactic_block)
        return Step(
            observation="",
            reward=shaped.reward,
            terminated=True,
            info={
                **shaped.info,
                "depth": self.task.depth,
                "key": self.task.key,
                "volume": self.task.volume,
                "residual_depth": self.task.residual_depth,
                "condition": self.task.label,
                "family": self.task.family.name,
            },
        )
