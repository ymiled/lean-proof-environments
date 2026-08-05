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

import re
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

    def prompt(self, format_example: bool = False) -> str:
        """What the policy sees. No hint about any lemma's mathematical role.

        `format_example` appends a worked example of the output *shape* -- not
        of any proof -- and defaults to off so measurement prompts stay exactly
        what they were. It exists for training. A multi-target response with no
        `-- PROOF` markers cannot be split, so it scores as unparseable however
        good its Lean is, and a small model that has not worked out the
        convention will sit at that floor emitting nothing learnable. Charging
        it for the convention is fair when measuring a frontier model and
        useless when bootstrapping a 4B one.
        """
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
        shape = ""
        if format_example:
            names = [self.instance.names[k] for k in self.ordered_targets]
            body = "\n".join(
                f"{TARGET_MARKER}{name}\n  <tactics for {name}>" for name in names
            )
            shape = (
                "\nYour reply must have exactly this shape, one marker line per "
                f"goal:\n\n{body}\n"
            )
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
            f"{shape}"
        )


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)(?:```|\Z)", re.DOTALL)
#: What a line may start with to count as announcing a section rather than
#: mentioning a lemma in passing. A markdown bullet (`- `) is deliberately not
#: here: prose bullets name lemmas constantly.
_HEADER_LEAD = ("#", "**", "--")


def _clean_block(block: str) -> str:
    """Reduce one section of a response to the tactics it actually contains.

    Two reductions, in order. First, if the section has fenced code, keep only
    the fenced parts and drop the prose around them: a model that narrates its
    proof and then states it is not making a formatting error worth punishing.
    All fences are kept, not just the first, because a section may open a fence
    per `have`. Second, drop a restated `theorem ... := by` header. Restating is
    harmless -- the harness writes the real statement itself, so a restatement
    can never widen or weaken the goal, it can only fail to compile -- but it
    has to be removed rather than tolerated, since it would otherwise be
    spliced *under* the header the harness already emitted.

    Both reductions are no-ops on a response that is already a bare tactic
    block, so this does not change how a well-behaved policy is scored.
    """
    fences = [m.group(1) for m in _FENCE.finditer(block)]
    body = "\n".join(fences) if fences else block

    lines = [ln for ln in body.splitlines() if ln.strip()]
    if lines and lines[0].lstrip().startswith(("theorem", "lemma", "example")):
        # A statement may wrap across several lines, so consume through the
        # line that closes it rather than just the first one.
        for i, ln in enumerate(lines):
            if ln.rstrip().endswith(":= by") or ln.strip() == ":= by":
                lines = lines[i + 1:]
                break
        else:
            lines = lines[1:]
    while lines and lines[0].strip() == ":= by":
        lines.pop(0)
    return "\n".join(lines)


def _section_starts(text: str, task: Task) -> "dict[str, int]":
    """Line index at which each target's section begins, however it is announced.

    `-- PROOF <name>` is the convention the prompt asks for, and a model that
    follows it is read exactly as before. But a model with its own strong
    output format will not adopt it: DeepSeek-Prover-V2 announces each lemma as
    `### Proof for \\`name\\`` and restates the theorem, and under a
    marker-only parser 98% of its responses were scored unparseable without a
    single Lean invocation. That measures markdown compliance, not proving.

    So a section also starts at a heading line naming the target, or at the
    target's own `theorem <name>` line. Only the earliest such line counts, and
    a name mentioned in running prose does not start anything.
    """
    starts: dict[str, int] = {}
    for key in task.targets:
        name = task.instance.names[key]
        for i, line in enumerate(text.splitlines()):
            stripped = line.strip()
            if name not in stripped:
                continue
            marker = (
                stripped.startswith(TARGET_MARKER.strip())
                and stripped[len(TARGET_MARKER.strip()):].strip() == name
            )
            heading = stripped.startswith(_HEADER_LEAD)
            decl = stripped.startswith((f"theorem {name}", f"lemma {name}"))
            if marker or heading or decl:
                starts[key] = i
                break
    return starts


def parse_blocks(text: str, task: Task) -> dict[str, str]:
    """Split a policy response into per-target tactic blocks.

    Falls back to treating the whole response as one block for single-target
    tasks, since a model given one goal often omits the marker.
    """
    lines = text.splitlines()
    starts = _section_starts(text, task)
    out: dict[str, str] = {}

    if starts:
        ordered = sorted(starts.items(), key=lambda kv: kv[1])
        bounds = [i for _, i in ordered] + [len(lines)]
        for (key, begin), end in zip(ordered, bounds[1:]):
            out[key] = _clean_block("\n".join(lines[begin + 1:end]))

    if not out and len(task.targets) == 1:
        out[task.targets[0]] = _clean_block(text)
    return {k: v for k, v in out.items() if v.strip()}
