"""Proof states, without taking a REPL dependency.

The environment grades one shot: the whole tactic block goes to `lean`, and one
verdict comes back. That makes an episode a contextual bandit, and it hides
everything about *where* a proof went wrong. A rollout that gets four tactics
right and fails on the fifth is indistinguishable from one that fails on the
first.

The usual fix is a persistent Lean server holding elaboration state between
tactics. That is the right long-term answer and it costs a new dependency which
has to be built against the pinned toolchain, plus a session pool, plus a
rewrite of the batching in `rollout.py`.

There is a cheaper route that needs nothing new. Lean will happily elaborate a
*prefix* of a tactic block if the remainder is `sorry`, and `trace_state` prints
the goals at the point it appears. So compiling

    <tactic 1>
    ...
    <tactic k>
    trace_state
    sorry

recovers the proof state after `k` tactics, and whether the prefix elaborates at
all tells us whether tactic `k` was valid. Sweeping `k` walks the whole
trajectory. This is what a REPL would give, one process per step instead of one
process per session, so it is far slower per query and exactly as informative.

Slow is acceptable for what this is for: diagnosing where proofs fail, and
deciding whether the interactive reformulation is worth building properly.
Bisection keeps the cost near `log k` when only the failure point is wanted.

The `sorry` is why this cannot be a reward. A prefix capped with `sorry` always
depends on `sorryAx`, so the axiom audit rejects it by construction. Success
here means "elaborates", which is strictly weaker than "proves", and the kernel
remains the only authority on the latter.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .grader import _lean_binary, _lean_env
from .task import Task

_ERROR_LINE = re.compile(r"error: (.*)")
#: `#print axioms` reports in the preamble land on stdout alongside the goal
#: that `trace_state` prints, and have to be told apart from it.
_AXIOM_NOISE = re.compile(
    r"depends on axioms|does not depend on any axioms"
)
#: Lean's complaint when a prefix already closed every goal. Not a failure:
#: it means the proof finished early, which is exactly what we want to detect.
_NO_GOALS = "no goals"
#: Diagnostic lines carry `file:line:col:`; the traced goal never does.
_DIAG = re.compile(r":\d+:\d+:")


def split_tactics(block: str) -> "list[str]":
    """Split a tactic block into independently elaborable steps.

    Two wrinkles, both learned from real reference proofs.

    The first line arrives unindented, because it is spliced directly after
    `:= by`, while its siblings carry the block's indentation. Taking the
    minimum indent over all lines would therefore make the first line the only
    top-level step and swallow the entire rest of the proof as its
    continuation. The base is taken from the remaining lines instead.

    A line opening with `|` continues the step above it rather than starting a
    new one. These are the arms of `induction ... with` or `cases ... with`,
    and they are not separately elaborable: a prefix ending at `induction e
    with` is a syntax error, so cutting there would report a failure that is an
    artefact of where we cut.
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return []
    rest = lines[1:]
    base = (
        min(len(ln) - len(ln.lstrip()) for ln in rest) if rest
        else len(lines[0]) - len(lines[0].lstrip())
    )

    steps: list[list[str]] = []
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        opens_arm = line.lstrip().startswith("|")
        if not steps or (i and indent <= base and not opens_arm):
            steps.append([line])
        else:
            steps[-1].append(line)
    return ["\n".join(s) for s in steps]


@dataclass
class Step:
    """One point along a proof, and what Lean thought of it."""

    index: int
    tactic: str
    ok: bool
    #: Goals after this tactic, as `trace_state` printed them. Empty when the
    #: prefix failed, or when no goals remained.
    goals: str = ""
    error: str = ""

    @property
    def closed(self) -> bool:
        """Whether the proof was complete at this point."""
        return self.ok and not self.goals.strip()


@dataclass
class Trace:
    """A whole tactic block, replayed one step at a time."""

    target: str
    steps: "list[Step]"

    @property
    def first_failure(self) -> "int | None":
        for s in self.steps:
            if not s.ok:
                return s.index
        return None

    @property
    def survived(self) -> int:
        """How many tactics elaborated before the first failure."""
        bad = self.first_failure
        return len(self.steps) if bad is None else bad

    @property
    def fraction_survived(self) -> float:
        return self.survived / len(self.steps) if self.steps else 0.0

    def __str__(self) -> str:
        bad = self.first_failure
        where = "all elaborated" if bad is None else f"failed at step {bad + 1}"
        return (
            f"{self.target}: {self.survived}/{len(self.steps)} steps, {where}"
        )


def _render(task: Task, target: str, prefix: str, trace: bool) -> str:
    """A file proving every other target by reference, `target` by `prefix`."""
    body = prefix.rstrip()
    tail = "\n  trace_state\n  sorry" if trace else "\n  sorry"
    blocks = dict(task.reference_solution())
    blocks[target] = body + tail
    parts = [task.preamble()]
    for key in task.ordered_targets:
        chunk = Task._indent(blocks[key])
        parts.append(f"{task.instance.statement_of(key)} := by\n{chunk}")
    return "\n\n".join(parts) + "\n"


def _elaborates(
    task: Task, target: str, prefix: str, timeout: float, trace: bool
) -> "tuple[bool, str, str]":
    """Compile one prefix. Returns (ok, goals, first error)."""
    source = _render(task, target, prefix, trace)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Probe.lean"
        path.write_text(source)
        try:
            proc = subprocess.run(
                [_lean_binary(), str(path)],
                capture_output=True, text=True, timeout=timeout,
                env=_lean_env(),
            )
        except subprocess.TimeoutExpired:
            return (False, "", f"timeout after {timeout}s")

    out = proc.stdout + proc.stderr
    # `declaration uses 'sorry'` is expected here and is not a failure.
    errors = [
        m.group(1).strip()
        for m in _ERROR_LINE.finditer(out)
        if "declaration uses 'sorry'" not in m.group(0)
    ]
    if errors:
        # A prefix that already discharged everything makes the appended
        # `trace_state`/`sorry` complain that there is nothing left to do.
        # That is success, and it is how a finished proof is recognised.
        if any(_NO_GOALS in e.lower() for e in errors):
            return (True, "", "")
        return (False, "", errors[0])

    # `trace_state` prints the goal bare on stdout, with no `information:`
    # prefix, mixed in with the preamble's `#print axioms` reports.
    goals = "\n".join(
        ln for ln in proc.stdout.splitlines()
        if ln.strip()
        and not _AXIOM_NOISE.search(ln)
        and not _DIAG.search(ln)
    ).strip()
    return (True, goals, "")


def replay(
    task: Task,
    target: str,
    block: str,
    *,
    timeout: float = 120.0,
    with_goals: bool = True,
) -> Trace:
    """Elaborate every prefix of `block`, recording goals and failures.

    Costs one `lean` invocation per tactic. Use `locate_failure` when only the
    first failing step is wanted.
    """
    tactics = split_tactics(block)
    steps: list[Step] = []
    failed = False
    for i in range(len(tactics)):
        if failed:
            steps.append(Step(index=i, tactic=tactics[i], ok=False,
                              error="not reached"))
            continue
        prefix = "\n".join(tactics[: i + 1])
        ok, goals, err = _elaborates(task, target, prefix, timeout, with_goals)
        steps.append(Step(index=i, tactic=tactics[i], ok=ok, goals=goals,
                          error=err))
        failed = not ok
    return Trace(target=target, steps=steps)


def locate_failure(
    task: Task, target: str, block: str, *, timeout: float = 120.0
) -> "int | None":
    """Index of the first tactic that fails to elaborate, by bisection.

    `None` when the whole block elaborates. Elaboration is monotone in the
    prefix -- if a prefix of length `k` fails, so does every longer one, since
    the failing tactic is still in it -- which is what licenses the bisection.
    """
    tactics = split_tactics(block)
    if not tactics:
        return None
    whole, _, _ = _elaborates(
        task, target, "\n".join(tactics), timeout, trace=False
    )
    if whole:
        return None

    lo, hi = 0, len(tactics) - 1  # invariant: prefix hi+1 fails
    while lo < hi:
        mid = (lo + hi) // 2
        ok, _, _ = _elaborates(
            task, target, "\n".join(tactics[: mid + 1]), timeout, trace=False
        )
        if ok:
            lo = mid + 1
        else:
            hi = mid
    return lo


# -- the diagnostic that decides whether interactive proving is worth it ----


def survival_histogram(
    rollouts, *, workers: int = 16, timeout: float = 120.0
) -> dict:
    """How far failed proofs get before they break.

    This is the measurement that should decide whether to build the interactive
    environment properly. Two very different worlds produce the same one-shot
    reward of `0.00`:

    *   failures concentrated at step 1, meaning the policy cannot start, and
        intermediate proof states would tell it nothing it does not already
        know from the prompt;
    *   failures spread through the proof, meaning rollouts routinely do real
        work before breaking, and every one of those partial trajectories is
        signal the bandit formulation throws away.

    The current reward cannot distinguish them. This can.
    """
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for r in rollouts:
        for target, verdict in r.shaped.per_target.items():
            block = r.blocks.get(target)
            if verdict == "proved" or not block or not block.strip():
                continue
            jobs.append((r.task, target, block))

    def one(job):
        task, target, block = job
        steps = split_tactics(block)
        if not steps:
            return None
        idx = locate_failure(task, target, block, timeout=timeout)
        return (len(steps), len(steps) if idx is None else idx)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [r for r in pool.map(one, jobs) if r is not None]

    if not results:
        return {"failed_targets": 0}

    buckets: dict[str, int] = {}
    for total, survived in results:
        frac = survived / total if total else 0.0
        key = ("0 (first tactic)" if survived == 0
               else "all (semantic)" if frac >= 1.0
               else f"{int(frac * 4) * 25}-{int(frac * 4) * 25 + 25}%")
        buckets[key] = buckets.get(key, 0) + 1

    mean = sum(s / t for t, s in results if t) / len(results)
    return {
        "failed_targets": len(results),
        "mean_fraction_survived": mean,
        "died_on_first_tactic": buckets.get("0 (first tactic)", 0),
        "buckets": dict(sorted(buckets.items())),
    }


def main() -> None:
    import argparse
    import random

    from .env import TaskSampler, all_families, split_families
    from .policy import LocalPolicy
    from .rollout import Grader

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--volume", type=int, default=3)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, _ = split_families(all_families(include_corpus=True))
    sampler = TaskSampler(
        families=train, depths=tuple(args.depths) or None,
        volume=args.volume or None, rng=random.Random(args.seed),
    )
    policy = LocalPolicy(
        model=args.model, base_url=args.base_url,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
    )
    grader = Grader(workers=args.workers)

    tasks = [t for t in sampler.batch(args.prompts) for _ in range(args.samples)]
    texts = policy.complete_many([t.prompt(format_example=True) for t in tasks])
    rollouts = grader.grade_batch(tasks, texts)

    stats = survival_histogram(rollouts, workers=args.workers)
    print(f"failed targets analysed  {stats['failed_targets']}")
    if not stats["failed_targets"]:
        return
    print(f"mean fraction elaborated {stats['mean_fraction_survived']:.2f}")
    print(f"died on first tactic     {stats['died_on_first_tactic']}")
    print()
    for bucket, n in stats["buckets"].items():
        share = n / stats["failed_targets"]
        print(f"  {bucket:<20}{n:>6}  {share:>6.1%}")
    print()
    first = stats["died_on_first_tactic"] / stats["failed_targets"]
    if first >= 0.7:
        print("VERDICT: most failures are immediate. Proof states would add "
              "little,\n         because the policy is not getting started at "
              "all. Fix the\n         prompt or the base model first.")
    else:
        print("VERDICT: failures are spread through the proof, so rollouts "
              "routinely\n         do real work before breaking. Every one of "
              "those partial\n         trajectories is signal the one-shot "
              "reward discards, which is\n         the case for building the "
              "interactive environment.")


if __name__ == "__main__":
    main()
