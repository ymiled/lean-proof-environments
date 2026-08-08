"""Per-target advantages, so a correct proof is not punished for its neighbours.

GRPO compares rollouts drawn from one prompt and gives each rollout a single
scalar advantage, which is then applied to every token it generated:

    A_i = (r_i - mean_j r_j) / std_j r_j

That is the right shape when a rollout is one indivisible attempt. Here it is
not. A rollout answers `n` independent lemmas, and its reward is their mean. So
a rollout that proves lemma A and fails lemma B receives one number that
averages a success and a failure, and every token of the *correct* proof of A
gets pushed down because B dragged the mean below the group's.

The environment already computes what is needed to fix this: a kernel verdict
per target, and the character range each target occupies in the response. This
module turns those into one advantage per (rollout, target), comparing each
target only against the *same* target elsewhere in the group.

Two things improve.

**More comparisons.** The group yields `G * n` independent contrasts instead of
`G`. At the settings used in the measured run, `G = 8` and `n = 3`, so 24
instead of 8.

**Degenerate groups are partly rescued.** A group is discarded when every
rollout scores the same. But equal *totals* do not mean equal *content*: if
rollout 1 proves only A and rollout 2 proves only B, both score `1/n` and the
aggregate advantage is zero for both, even though the per-target comparison is
clean in both directions. Those groups carry signal that the scalar form
discards entirely.

The trainer-side integration lives in `pdd.train_grpo`; everything here is pure
and testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .grader import Verdict
from .rollout import Rollout
from .task import section_spans

#: Advantage below which a span is not worth masking, to keep the token work down.
EPS = 1e-8


@dataclass
class TargetCredit:
    """One target's advantage, and where in the response it applies."""

    target: str
    advantage: float
    #: Character range within the rollout's raw text. Empty when the response
    #: had no identifiable section for this target.
    span: "tuple[int, int] | None" = None
    proved: bool = False


@dataclass
class FactoredGroup:
    """Per-rollout credits for one group of rollouts sampled from one prompt."""

    credits: list[list[TargetCredit]] = field(default_factory=list)
    #: Targets whose outcome was identical across the whole group, and which
    #: therefore contribute nothing however the advantage is computed.
    flat_targets: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return any(
            abs(c.advantage) > EPS for row in self.credits for c in row
        )


def _standardise(
    values: "list[float]", scale: bool = True
) -> "list[float]":
    """Centre, and optionally divide by the standard deviation.

    `scale=False` is the Dr. GRPO correction. Dividing by the group standard
    deviation makes low-variance groups contribute larger updates per unit of
    reward difference, and variance here is a direct function of difficulty, so
    the division systematically reweights the curriculum toward whichever
    targets are currently near-uniform. Centring alone is unbiased.
    """
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    centred = [v - mean for v in values]
    if not scale:
        return centred
    var = sum(c * c for c in centred) / n
    sd = var ** 0.5
    if sd <= EPS:
        return [0.0] * n
    return [c / sd for c in centred]


def factor_group(
    rollouts: "list[Rollout]", scale: bool = True
) -> FactoredGroup:
    """Advantages per (rollout, target) for one group sampled from one prompt.

    Every rollout in the group must carry the same task, which is what makes the
    per-target comparison meaningful: target `t` of rollout `i` is compared with
    target `t` of every other rollout, holding the lemma and its renaming fixed.
    """
    if not rollouts:
        return FactoredGroup()
    task = rollouts[0].task
    if any(r.task.targets != task.targets for r in rollouts):
        raise ValueError("factor_group needs one task per group")

    # `Shaped.per_target` holds verdict *values*, not `Verdict` members, and
    # `Verdict` is a `str` enum -- so `is` silently fails where `==` succeeds.
    proved = Verdict.PROVED.value
    outcomes: dict[str, list[float]] = {}
    for target in task.targets:
        outcomes[target] = [
            1.0 if r.shaped.per_target.get(target) == proved else 0.0
            for r in rollouts
        ]

    advantages = {t: _standardise(v, scale) for t, v in outcomes.items()}
    flat = tuple(
        t for t, v in advantages.items() if all(abs(a) <= EPS for a in v)
    )

    spans = [section_spans(r.text, r.task) for r in rollouts]
    credits: list[list[TargetCredit]] = []
    for i in range(len(rollouts)):
        row = [
            TargetCredit(
                target=t,
                advantage=advantages[t][i],
                span=spans[i].get(t),
                proved=outcomes[t][i] > 0.5,
            )
            for t in task.targets
        ]
        credits.append(row)
    return FactoredGroup(credits=credits, flat_targets=flat)


def factor_batch(
    rollouts: "list[Rollout]", group_size: int, scale: bool = True
) -> "list[FactoredGroup]":
    """Split a contiguous batch into groups and factor each one."""
    return [
        factor_group(rollouts[i:i + group_size], scale)
        for i in range(0, len(rollouts), group_size)
    ]


# -- token-level materialisation -------------------------------------------


def span_to_token_mask(
    offsets: "list[tuple[int, int]]", span: "tuple[int, int]"
) -> "list[bool]":
    """Which tokens fall inside a character range, given offset mappings.

    A token counts as inside when it overlaps the range at all. Tokens straddle
    section boundaries only at the boundary itself, so the ambiguity is one
    token per section and assigning it to both sides is harmless.
    """
    lo, hi = span
    return [not (end <= lo or start >= hi) for start, end in offsets]


def per_token_advantages(
    credits: "list[TargetCredit]",
    offsets: "list[tuple[int, int]]",
    fallback: float = 0.0,
) -> "list[float]":
    """Spread one rollout's per-target advantages over its tokens.

    Tokens outside every section, such as a preamble or a sign-off, get
    `fallback`. Zero is the honest default: nothing was learned about them, so
    they should neither be encouraged nor suppressed.
    """
    out = [fallback] * len(offsets)
    for credit in credits:
        if credit.span is None or abs(credit.advantage) <= EPS:
            continue
        mask = span_to_token_mask(offsets, credit.span)
        for i, inside in enumerate(mask):
            if inside:
                out[i] = credit.advantage
    return out


def coverage(credits: "list[TargetCredit]") -> float:
    """Fraction of a rollout's targets that were actually located in its text.

    Worth logging. A response whose sections cannot be found still receives a
    scalar reward, but it cannot receive factored credit, so if this falls the
    factoring is quietly degrading to something close to the old behaviour.
    """
    if not credits:
        return 0.0
    return sum(1 for c in credits if c.span is not None) / len(credits)
