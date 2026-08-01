"""The grader: decide whether a candidate proof actually proves the theorem.

This is the part of the environment that has to be right. A verification reward
signal is only as trustworthy as its grader, and Lean offers several ways to
close a goal without proving it. Concretely, during development of this
benchmark a rung whose proof failed still produced a *usable declaration*, and
every downstream lemma that cited it compiled cleanly -- `#print axioms` on a
theorem three rungs later reported `sorryAx`. A grader that only asked "did the
file compile?" would have scored that entire subtree as success.

So acceptance requires all of:

1.  The file compiles with exit status 0 (no `error:` diagnostics).
2.  `#print axioms` on the target reports a subset of the standard three.
    This is the real check: it catches `sorry`, `admit`, any freshly declared
    axiom, and anything that sneaks in transitively through a cited lemma.
3.  No banned syntax, checked on the model's own text only. `native_decide`
    trusts the compiler rather than the kernel, so it is excluded even though it
    leaves no axiom trace of the kind we scan for.

Point 2 subsumes most of point 3, but both are kept: the textual scan gives a
precise reason for rejection, which matters when reading transcripts.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .task import Task

#: Axioms Lean's own standard library depends on. `simp` pulls in `propext`
#: routinely, so an empty axiom set is too strict a bar.
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

#: Rejected on sight in model output.
BANNED = (
    "sorry",
    "admit",
    "native_decide",
    "axiom ",
    "@[implemented_by",
    "unsafe ",
    "partial ",
    "extern",
)

_AXIOM_LINE = re.compile(r"depends on axioms: \[(.*?)\]")
_NO_AXIOMS = re.compile(r"does not depend on any axioms")


class Verdict(str, Enum):
    PROVED = "proved"
    COMPILE_ERROR = "compile_error"
    BANNED_SYNTAX = "banned_syntax"
    BAD_AXIOMS = "bad_axioms"
    TIMEOUT = "timeout"


@dataclass
class Result:
    verdict: Verdict
    #: Axioms the target ended up depending on, when we got far enough to tell.
    axioms: frozenset[str] = frozenset()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PROVED


def _lean_binary() -> str:
    candidate = Path.home() / ".elan" / "bin" / "lean"
    return str(candidate) if candidate.exists() else "lean"


def grade(
    task: Task, blocks: "dict[str, str] | str", timeout: float = 120.0
) -> Result:
    """Check candidate proofs against a task.

    A multi-target task is graded all-or-nothing: every target must compile and
    every target must pass the axiom audit. That matches the quantity under
    test, which is a count of *problems solved*, and it prevents a policy from
    scoring by proving only the easy members of a set.
    """
    texts = [blocks] if isinstance(blocks, str) else list(blocks.values())
    for text in texts:
        lowered = text.lower()
        for token in BANNED:
            if token in lowered:
                return Result(Verdict.BANNED_SYNTAX, detail=f"contains {token!r}")

    source = task.assemble(blocks)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Candidate.lean"
        path.write_text(source)
        try:
            proc = subprocess.run(
                [_lean_binary(), str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(Verdict.TIMEOUT, detail=f"exceeded {timeout}s")

    output = proc.stdout + proc.stderr

    # Lean reports `sorry` as a warning, not an error, so exit status alone is
    # not sufficient -- hence the axiom check below.
    if proc.returncode != 0 or "error:" in output:
        first = next(
            (ln for ln in output.splitlines() if "error:" in ln), output[:200]
        )
        return Result(Verdict.COMPILE_ERROR, detail=first.strip())

    # One `#print axioms` line per target. Every one must clear the bar: a
    # `sorryAx` anywhere in the set means the set was not proved.
    reports = _AXIOM_LINE.findall(output)
    clean = len(_NO_AXIOMS.findall(output))
    expected = len(task.theorem_names)
    if len(reports) + clean < expected:
        return Result(
            Verdict.BAD_AXIOMS,
            detail=f"expected {expected} axiom reports, saw {len(reports) + clean}",
        )

    axioms = frozenset(
        a.strip() for r in reports for a in r.split(",") if a.strip()
    )
    extra = axioms - ALLOWED_AXIOMS
    if extra:
        return Result(
            Verdict.BAD_AXIOMS,
            axioms=axioms,
            detail=f"disallowed: {sorted(extra)}",
        )

    return Result(Verdict.PROVED, axioms=axioms)


def grade_source(source: str, theorem_names: list[str],
                 timeout: float = 120.0) -> Result:
    """Grade a complete Lean file directly, outside the `Task` machinery.

    Needed by the depth check in `selftest`: to show a declared dependency is
    real, a rung has to be compiled with its ancestors *absent from the file
    entirely*, which no `Task` rendering produces.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Solo.lean"
        path.write_text(source)
        try:
            proc = subprocess.run(
                [_lean_binary(), str(path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(Verdict.TIMEOUT, detail=f"exceeded {timeout}s")

    output = proc.stdout + proc.stderr
    if proc.returncode != 0 or "error:" in output:
        first = next((ln for ln in output.splitlines() if "error:" in ln),
                     output[:200])
        return Result(Verdict.COMPILE_ERROR, detail=first.strip())

    reports = _AXIOM_LINE.findall(output)
    clean = len(_NO_AXIOMS.findall(output))
    if len(reports) + clean < len(theorem_names):
        return Result(Verdict.BAD_AXIOMS, detail="missing axiom reports")
    axioms = frozenset(a.strip() for r in reports for a in r.split(",") if a.strip())
    extra = axioms - ALLOWED_AXIOMS
    if extra:
        return Result(Verdict.BAD_AXIOMS, axioms=axioms,
                      detail=f"disallowed: {sorted(extra)}")
    return Result(Verdict.PROVED, axioms=axioms)
