"""Rendering a rung into the two experimental conditions.

The policy is asked for a *tactic block only*, never a whole file. The harness
supplies the `theorem ... : ... := by` header itself and splices the model's
text underneath. This is not a convenience: it makes statement drift
structurally impossible. A model cannot weaken the goal, restate it, or prove a
different lemma under a matching name, because it never gets to write the
statement. That removes the largest class of reward hacking without any
statement-comparison logic at all.

The remaining hacks -- `sorry`, fresh axioms, `native_decide` -- are the
grader's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ladder import Instance


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
                group = binders[start:i]
                names.extend(group.split(":")[0].split())
    return " ".join(names)


class Condition(str, Enum):
    """The two arms of the experiment.

    MONOLITHIC mirrors lf-lean's "prove the whole dependency tree yourself";
    COMPOSITIONAL mirrors their trusted-interface approach, the setting in which
    they report the O(|P|) -> O(max_i |c_i|) context reduction.
    """

    MONOLITHIC = "monolithic"
    COMPOSITIONAL = "compositional"


@dataclass(frozen=True)
class Task:
    instance: Instance
    key: str
    condition: Condition

    @property
    def family(self):
        return self.instance.family

    @property
    def depth(self) -> int:
        return self.family.depth_of(self.key)

    @property
    def theorem_name(self) -> str:
        return self.instance.names[self.key]

    @property
    def ancestors(self) -> list[str]:
        return self.family.transitive_deps(self.key)

    def preamble(self) -> str:
        """Everything above the target theorem."""
        parts = [self.instance.definitions]
        if self.condition is Condition.COMPOSITIONAL:
            # Ancestors arrive already proved, as trusted interfaces.
            parts.extend(self.instance.render_rung(k) for k in self.ancestors)
        return "\n\n".join(parts)

    def header(self) -> str:
        return f"{self.instance.statement_of(self.key)} := by"

    def assemble(self, tactic_block: str) -> str:
        """Splice a candidate tactic block into a complete, checkable Lean file."""
        body = "\n".join(
            line if line.startswith("  ") or not line.strip() else "  " + line
            for line in tactic_block.rstrip().splitlines()
        )
        return (
            f"{self.preamble()}\n\n"
            f"{self.header()}\n{body}\n\n"
            f"#print axioms {self.theorem_name}\n"
        )

    def reference_solution(self) -> str:
        """A known-good tactic block for this task, in this condition.

        Compositional: cite the ancestors, which are already in scope.

        Monolithic: the ancestors do not exist, so they are re-proved inline as
        `have`s in dependency order. This is what certifies that the monolithic
        arm is solvable at every depth. Without it a decay curve could just as
        easily be reporting that the tasks were impossible, and the experiment
        would be measuring the benchmark rather than the model.
        """
        target = self.instance.reference_proof_of(self.key)
        if self.condition is Condition.COMPOSITIONAL:
            return target

        blocks = []
        for anc in self.ancestors:
            rung = self.family.by_key[anc]
            binders = rung.binders.format(**self.instance.names)
            stmt = rung.statement.format(**self.instance.names)
            body = self.instance.reference_proof_of(anc)
            indented = "\n".join("  " + ln for ln in body.splitlines())
            quantified = f"∀ {binders}, {stmt}" if binders else stmt
            intro = _binder_vars(binders)
            intro_line = f"    intro {intro}\n" if intro else ""
            blocks.append(
                f"  have {self.instance.names[anc]} : {quantified} := by\n"
                f"{intro_line}{indented}"
            )
        blocks.append(target)
        return "\n".join(blocks)

    def prompt(self) -> str:
        """What the policy sees. No hint about the lemma's mathematical role."""
        avail = ""
        if self.condition is Condition.COMPOSITIONAL and self.ancestors:
            names = ", ".join(self.instance.names[k] for k in self.ancestors)
            avail = (
                "\nAlready-proved lemmas available in scope, which you may cite "
                f"by name: {names}\n"
            )
        return (
            "Prove the final theorem in this Lean 4 file.\n\n"
            "```lean\n"
            f"{self.preamble()}\n\n"
            f"{self.header()}\n"
            "```\n"
            f"{avail}\n"
            "Reply with ONLY the tactic block that completes the proof, indented "
            "two spaces. No code fences, no commentary, no restatement of the "
            "theorem. Do not use `sorry`, `native_decide`, or declare axioms. "
            "Mathlib is not available.\n"
        )


def all_tasks(instance: Instance, condition: Condition) -> list[Task]:
    return [Task(instance, key, condition) for key in instance.family.by_key]
