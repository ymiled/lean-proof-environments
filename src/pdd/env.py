"""Gym-style interface over the ladder.

The experiment in `sweep.py` does not need this -- it samples a whole proof and
grades it once. The interface exists so the benchmark is drop-in trainable: a
learner needs `reset`/`step` and a scalar reward, and providing them costs
almost nothing once the grader exists.

Episodes are single-step by construction. Making them multi-step (one tactic per
action, with intermediate proof states as observations) would require a
persistent Lean server rather than one-shot compilation. That is the natural
next version and is noted in the README as future work rather than pretended at
here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .grader import Result, grade
from .families import DEFAULT, FAMILIES
from .ladder import Instance
from .task import Condition, Task


@dataclass
class Step:
    observation: str
    reward: float
    terminated: bool
    info: dict


class LadderEnv:
    """One episode = one attempt at one rung.

    Reward is binary and comes from the kernel: 1.0 if the grader accepts,
    0.0 otherwise. There is deliberately no partial credit for "compiles but
    leaves goals" -- that is exactly the proxy-reward failure Theorem argues
    against, and rewarding it would teach a policy to produce plausible-looking
    proofs that do not close.
    """

    def __init__(
        self,
        condition: Condition = Condition.COMPOSITIONAL,
        family: str = DEFAULT,
        depths: tuple[int, ...] | None = None,
        seed: int = 0,
    ) -> None:
        self.condition = condition
        self.family = FAMILIES[family]
        self.rng = random.Random(seed)
        self.keys = [
            k for k in self.family.by_key
            if depths is None or self.family.depth_of(k) in depths
        ]
        self.task: Task | None = None

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            self.rng.seed(seed)
        inst = Instance.sample(self.family, self.rng.randrange(2**31))
        self.task = Task(inst, self.rng.choice(self.keys), self.condition)
        return self.task.prompt()

    def step(self, tactic_block: str) -> Step:
        if self.task is None:
            raise RuntimeError("call reset() before step()")
        result: Result = grade(self.task, tactic_block)
        return Step(
            observation="",
            reward=1.0 if result.ok else 0.0,
            terminated=True,
            info={
                "verdict": result.verdict.value,
                "detail": result.detail,
                "depth": self.task.depth,
                "key": self.task.key,
                "condition": self.condition.value,
                "family": self.family.name,
            },
        )
