"""Policies: things that emit a tactic block given a task.

The name is deliberate. In MDP terms these are $\\pi_\\theta$, and the rest of
the package is the environment. Nothing here is trained -- this repository
measures a frozen policy against a graded environment. `env.py` exposes the
reset/step interface a learner would need if one were attached later.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from .task import Task

_FENCE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL)


class Policy(Protocol):
    name: str

    def act(self, task: Task) -> str:
        """Return a candidate tactic block."""
        ...


def _strip_fences(text: str) -> str:
    """Models add code fences despite being asked not to. Tolerate it.

    Being lenient here is correct: fence-stripping is a formatting concern, and
    charging a model a failure for it would confound presentation with proving
    ability, which is the thing we are trying to measure.
    """
    match = _FENCE.search(text)
    body = match.group(1) if match else text
    lines = [ln for ln in body.splitlines() if ln.strip()]
    # Drop a restated `theorem ... := by` header if one slipped through.
    while lines and (
        lines[0].lstrip().startswith(("theorem", "lemma", "example"))
        or lines[0].strip() == ":= by"
    ):
        lines.pop(0)
    return "\n".join(lines)


class ReferencePolicy:
    """Oracle. Always emits the known-good proof.

    Used to certify the benchmark and to provide the ceiling line on plots.
    """

    name = "reference"

    def act(self, task: Task) -> str:
        return task.reference_solution()


class EmptyPolicy:
    """Floor. Emits a proof that never works.

    Gives the plot a zero line and catches graders that accept garbage.
    """

    name = "empty"

    def act(self, task: Task) -> str:
        return "  rfl"


class AnthropicPolicy:
    """A frozen Claude model, sampled at fixed temperature."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"{model}@T{temperature}"

    def act(self, task: Task) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": task.prompt()}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _strip_fences(text)
