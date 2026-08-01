"""Rendering a set of lemmas into a checkable Lean file.

A task is *prove these lemmas*; everything else in the family is supplied,
already proved, as a trusted interface. That single formulation covers all three
experiments:

*   **compositional** -- targets = one lemma, all ancestors supplied.
*   **monolithic** -- targets = one lemma plus every ancestor.
*   **chain / antichain contrast** -- targets = k lemmas of matched size and
    length, differing only in whether they form a dependency chain. See
    `pdd.design`.

The policy is asked for *tactic blocks only*, never whole files. The harness
writes each `theorem ... := by` header itself. This is not a convenience: it
makes statement drift structurally impossible. A model cannot weaken a goal,
restate it, or prove a different lemma under a matching name, because it never
gets to write the statement. That removes the largest class of reward hacking
without any statement-comparison logic at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ladder import Instance

TARGET_MARKER = "-- PROOF "


def _binder_vars(binders: str) -> str:
    """"(G : Ctx) (s t : St) (h : P)" -> "G s t h".

    Scans balanced groups rather than splitting on delimiters, because binder
    types may themselves contain parentheses.
    """
    names: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(binders):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                names.extend(binders[start:i].split(":")[0].split())
    return " ".join(names)


class Condition(str, Enum):
    """The two classic arms, kept as named constructors over the general form."""

    MONOLITHIC = "monolithic"
    COMPOSITIONAL = "compositional"


@dataclass(frozen=True)
class Task:
    instance: Instance
    targets: tuple[str, ...]
    #: Free-text label for reporting: a Condition value, or "chain"/"antichain".
    label: str = "custom"

    # -- construction ------------------------------------------------------

    @classmethod
    def single(cls, instance: Instance, key: str, condition: Condition) -> "Task":
        if condition is Condition.COMPOSITIONAL:
            targets = (key,)
        else:
            targets = (*instance.family.transitive_deps(key), key)
        return cls(instance, tuple(targets), condition.value)

    # -- structure ---------------------------------------------------------

    @property
    def family(self):
        return self.instance.family

    @property
    def key(self) -> str:
        """Deepest target, used for labelling and for the legacy `depth` field."""
        return max(self.targets, key=self.family.depth_of)

    @property
    def depth(self) -> int:
        return self.family.depth_of(self.key)

    @property
    def volume(self) -> int:
        """How many lemmas the policy must produce."""
        return len(self.targets)

    @property
    def residual_depth(self) -> int:
        """Longest dependency chain *within* the target set.

        This is what the chain/antichain contrast manipulates, and it is
        distinct from `depth`, which is a property of a lemma in the full DAG
        regardless of what has been supplied. An antichain of size k has
        residual depth 1; a chain of size k has residual depth k.
        """
        inside = set(self.targets)
        memo: dict[str, int] = {}

        def rec(k: str) -> int:
            if k not in memo:
                deps = [d for d in self.family.by_key[k].deps if d in inside]
                memo[k] = 1 + max((rec(d) for d in deps), default=0)
            return memo[k]

        return max(rec(t) for t in self.targets)

    @property
    def ordered_targets(self) -> list[str]:
        return sorted(self.targets, key=self.family.depth_of)

    @property
    def supplied(self) -> list[str]:
        """Lemmas provided already proved, in dependency order."""
        needed: list[str] = []
        for t in self.targets:
            for a in self.family.transitive_deps(t):
                if a not in self.targets and a not in needed:
                    needed.append(a)
        return sorted(needed, key=self.family.depth_of)

    @property
    def theorem_names(self) -> list[str]:
        return [self.instance.names[k] for k in self.ordered_targets]

    @property
    def reference_loc(self) -> int:
        return sum(
            len(self.instance.reference_proof_of(t).splitlines())
            for t in self.targets
        )

    # -- rendering ---------------------------------------------------------

    def preamble(self) -> str:
        parts = [self.instance.definitions]
        parts.extend(self.instance.render_rung(k) for k in self.supplied)
        return "\n\n".join(parts)

    @staticmethod
    def _indent(block: str) -> str:
        return "\n".join(
            ln if ln.startswith("  ") or not ln.strip() else "  " + ln
            for ln in block.rstrip().splitlines()
        )

    def assemble(self, blocks: dict[str, str] | str) -> str:
        """Splice candidate tactic blocks into a complete, checkable Lean file."""
        if isinstance(blocks, str):
            if len(self.targets) != 1:
                raise ValueError("multi-target task needs a dict of blocks")
            blocks = {self.targets[0]: blocks}

        parts = [self.preamble()]
        for key in self.ordered_targets:
            body = self._indent(blocks.get(key) or "  sorry")
            parts.append(f"{self.instance.statement_of(key)} := by\n{body}")
        checks = "\n".join(f"#print axioms {n}" for n in self.theorem_names)
        return "\n\n".join(parts) + "\n\n" + checks + "\n"

    def reference_solution(self) -> dict[str, str]:
        """Known-good blocks for every target. Certifies the task is solvable."""
        return {k: self.instance.reference_proof_of(k) for k in self.targets}

    def prompt(self) -> str:
        """What the policy sees. No hint about any lemma's mathematical role."""
        avail = ""
        if self.supplied:
            names = ", ".join(self.instance.names[k] for k in self.supplied)
            avail = (
                "\nAlready-proved lemmas in scope, which you may cite by name: "
                f"{names}\n"
            )
        goals = "\n\n".join(
            f"{TARGET_MARKER}{self.instance.names[k]}\n"
            f"{self.instance.statement_of(k)} := by"
            for k in self.ordered_targets
        )
        n = len(self.targets)
        plural = "s" if n > 1 else ""
        return (
            f"Prove the following {n} Lean 4 theorem{plural}.\n\n"
            "Context already in scope:\n\n"
            "```lean\n"
            f"{self.preamble()}\n"
            "```\n"
            f"{avail}\n"
            f"Goal{plural}:\n\n"
            "```lean\n"
            f"{goals}\n"
            "```\n\n"
            f"For each goal, output a line `{TARGET_MARKER}<name>` followed by "
            "ONLY the tactic block that completes it, indented two spaces. "
            "Earlier goals are in scope for later ones. No commentary, no code "
            "fences, no restatement of the theorem. Do not use `sorry`, "
            "`native_decide`, or declare axioms. Mathlib is not available.\n"
        )


def parse_blocks(text: str, task: Task) -> dict[str, str]:
    """Split a policy response into per-target tactic blocks.

    Falls back to treating the whole response as one block for single-target
    tasks, since a model given one goal often omits the marker.
    """
    name_to_key = {task.instance.names[k]: k for k in task.targets}
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(TARGET_MARKER.strip()):
            if current is not None:
                out[current] = "\n".join(buf)
            marker = stripped[len(TARGET_MARKER.strip()):].strip()
            current = name_to_key.get(marker)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf)

    if not out and len(task.targets) == 1:
        out[task.targets[0]] = text
    return out
