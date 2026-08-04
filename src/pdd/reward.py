"""Turning kernel verdicts into a scalar a policy-gradient method can learn from.

`grade` answers one question -- did the whole task check -- and that is the right
question for *measurement*. It is the wrong question for *training*. Group-
relative methods (GRPO and relatives) centre the reward inside a group of
rollouts sampled from one prompt: the advantage is `(r_i - mean(r)) / std(r)`.
If every rollout in the group scores zero, the mean is zero, the spread is zero,
and every advantage is zero. The update is exactly nothing. A small model
attempting a depth-4 Lean task fails every rollout for a long time, so binary
task reward gives an all-zero group on essentially every prompt and training
does not begin.

`grade_partial` already computes the finer signal this repository's main result
rests on: per-target verdicts, each held to the same axiom bar. This module
turns that into a reward.

Two design points worth stating, since one of them contradicts a comment that
used to live in `env.py`:

*   **Partial credit is per *target*, not per *goal state*.** The objection to
    partial credit is that rewarding "compiles but leaves goals open" teaches a
    policy to emit plausible non-proofs. That objection is about crediting
    unfinished proofs. Nothing here does that. Every credited target passed
    compilation *and* the `#print axioms` audit -- the identical bar `grade`
    applies. What varies is how many of the task's obligations were discharged,
    which is a real difference in what the policy accomplished.

*   **Failures are ranked below zero when they are evasions.** A compile error
    is an honest miss and scores 0. `sorry`, a fresh `axiom`, or `native_decide`
    are attempts to obtain the reward without the proof, and score negative, so
    the gradient points away from them from the first update rather than after
    the grader has rejected them enough times for the policy to notice.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grader import PartialResult, Verdict, grade_partial
from .task import Task


@dataclass(frozen=True)
class RewardConfig:
    """Per-verdict weights. Per-target terms are averaged over the targets.

    Defaults put a proved task at `1.0 + all_bonus` and a wholly failed one at
    or below `0.0`, so the sign of the reward already tells you whether anything
    was proved.
    """

    #: Credit for one target that compiled and cleared the axiom audit.
    proved: float = 1.0
    #: Added once if *every* target passed. Keeps the binary objective visible
    #: to the policy: without it, proving 3 of 4 twice beats proving 4 of 4 once.
    all_bonus: float = 0.2
    #: An honest failure to prove. The neutral point.
    compile_error: float = 0.0
    #: Compiles, but leans on a disallowed axiom -- i.e. `sorry` reached the
    #: kernel. An evasion, not a miss.
    bad_axioms: float = -0.10
    #: Rejected by the textual scan before Lean ran. Same intent, caught earlier.
    banned: float = -0.10
    #: Ran out of time. Mildly discouraged: usually a runaway `simp` loop.
    timeout: float = -0.05
    #: The response contained no block for this target at all.
    missing: float = -0.05
    #: The response contained no parseable block for *any* target. A formatting
    #: failure rather than a proving failure, but it must not be free.
    unparseable: float = -0.10


DEFAULT = RewardConfig()

_VERDICT_FIELD = {
    Verdict.PROVED: "proved",
    Verdict.COMPILE_ERROR: "compile_error",
    Verdict.BAD_AXIOMS: "bad_axioms",
    Verdict.BANNED_SYNTAX: "banned",
    Verdict.TIMEOUT: "timeout",
}


@dataclass
class Shaped:
    """A reward plus everything needed to explain it in a training log."""

    reward: float
    #: Fraction of targets proved. The quantity the paper reports.
    score: float
    #: Whether the task would have been accepted all-or-nothing.
    binary: bool
    per_target: dict[str, str]
    n_targets: int
    n_proved: int
    #: False when the response yielded no usable blocks at all.
    parsed: bool = True

    @property
    def info(self) -> dict:
        return {
            "reward": self.reward,
            "score": self.score,
            "binary": self.binary,
            "n_proved": self.n_proved,
            "n_targets": self.n_targets,
            "parsed": self.parsed,
            "per_target": self.per_target,
        }


def unparseable(task: Task, config: RewardConfig = DEFAULT) -> Shaped:
    """The reward for a response that yielded no block for any target."""
    return Shaped(
        reward=config.unparseable,
        score=0.0,
        binary=False,
        per_target={t: "unparseable" for t in task.targets},
        n_targets=len(task.targets),
        n_proved=0,
        parsed=False,
    )


def combine(
    task: Task,
    verdicts: dict[str, Verdict],
    attempted: "set[str] | frozenset[str]",
    config: RewardConfig = DEFAULT,
) -> Shaped:
    """Fold per-target verdicts into one reward.

    Kept separate from grading so a caller that already has verdicts -- from a
    batched or cached grader, say -- does not have to recompute them. `attempted`
    is the set of targets the response actually supplied a block for; the rest
    are scored as missing rather than as compile errors, which is a real
    distinction the policy should feel.
    """
    n = len(task.targets) or 1
    total = 0.0
    for target in task.targets:
        if target not in attempted:
            total += config.missing
            continue
        total += getattr(config, _VERDICT_FIELD[verdicts[target]])

    proved = [t for t in task.targets if verdicts.get(t) is Verdict.PROVED]
    binary = len(proved) == len(task.targets)
    reward = total / n + (config.all_bonus if binary else 0.0)

    return Shaped(
        reward=reward,
        score=len(proved) / n,
        binary=binary,
        per_target={
            t: ("missing" if t not in attempted else verdicts[t].value)
            for t in task.targets
        },
        n_targets=len(task.targets),
        n_proved=len(proved),
    )


def shaped_reward(
    task: Task,
    blocks: "dict[str, str] | str",
    config: RewardConfig = DEFAULT,
    timeout: float = 120.0,
) -> Shaped:
    """Grade `blocks` per target and collapse the verdicts into one number.

    Costs one Lean invocation per target (see `grade_partial`). That is the
    price of a non-degenerate advantage estimate, and it is the dominant cost of
    an RL step on this environment. Parallelise it at the rollout level rather
    than here -- `rollout.Grader` does, and it also deduplicates identical
    blocks across a rollout group, which this single-task path cannot.
    """
    if isinstance(blocks, str):
        blocks = {task.targets[0]: blocks}

    attempted = {k for k, v in blocks.items() if v and v.strip()}
    if not attempted:
        return unparseable(task, config)

    result: PartialResult = grade_partial(task, blocks, timeout=timeout)
    return combine(task, result.per_target, attempted, config)


def binary_reward(
    task: Task, blocks: "dict[str, str] | str", timeout: float = 120.0
) -> Shaped:
    """The all-or-nothing reward, kept as the ablation arm.

    Training against this is the control that shows the shaped reward is doing
    the work rather than the rest of the setup.
    """
    from .grader import grade

    result = grade(task, blocks, timeout=timeout)
    targets = task.targets
    return Shaped(
        reward=1.0 if result.ok else 0.0,
        score=1.0 if result.ok else 0.0,
        binary=result.ok,
        per_target={t: result.verdict.value for t in targets},
        n_targets=len(targets),
        n_proved=len(targets) if result.ok else 0,
    )
