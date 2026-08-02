"""Grade a completed sweep and produce every number the paper reports.

One pass: grade, Wilson intervals per cell, the pooled comparison over tasks
that actually differ between conditions, and the figure. Keeping this in one
place is deliberate. The paper quotes eleven numbers from this sweep, and when
they were computed by separate ad-hoc scripts it was possible for a table and a
figure to disagree without anyone noticing.

Depth-1 cells are reported but excluded from the pooled test, because at depth 1
a rung has no ancestors and the two conditions render byte-identical prompts.
Including them would dilute a real effect with cells that cannot show one, while
their disagreement across arms is exactly the noise-floor estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .families import FAMILIES
from .grader import grade
from .ladder import Instance
from .task import Task, parse_blocks

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on [[a,b],[c,d]]."""
    def hyp(a: int, b: int, c: int, d: int) -> float:
        n = a + b + c + d
        return (math.comb(a + b, a) * math.comb(c + d, c)) / math.comb(n, a + c)

    obs = hyp(a, b, c, d)
    total = a + c
    p = 0.0
    for i in range(total + 1):
        j = total - i
        if i <= a + b and j <= c + d:
            pr = hyp(i, a + b - i, j, c + d - j)
            if pr <= obs + 1e-12:
                p += pr
    return p


def grade_all() -> list[dict]:
    specs = json.loads((RUNS / "_specs.json").read_text())
    fam = FAMILIES[specs[0]["family"]]
    rows = []
    for s in specs:
        out = RUNS / f"{s['task_id']}.out"
        if not out.exists():
            continue
        inst = Instance.sample(fam, s["seed"])
        task = Task(inst, tuple(s["targets"]), s["arm"])
        res = grade(task, parse_blocks(out.read_text(), task))
        rows.append({**s, "ok": res.ok, "verdict": res.verdict.value})
    return rows


def report(rows: list[dict], label: str) -> None:
    cells: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["k"])].append(r["ok"])

    print(f"{'depth':>5} {'compositional':>22} {'monolithic':>22}")
    for d in sorted({k for _, k in cells}):
        line = f"{d:>5}"
        for arm in ("compositional", "monolithic"):
            v = cells.get((arm, d), [])
            if not v:
                line += f"{'--':>22}"
                continue
            k, n = sum(v), len(v)
            lo, hi = wilson(k, n)
            line += f"{f'{k}/{n} ({k/n:.2f}) [{lo:.2f},{hi:.2f}]':>22}"
        print(line)

    deep = [r for r in rows if r["k"] >= 2]
    a = sum(r["ok"] for r in deep if r["arm"] == "compositional")
    b = len([r for r in deep if r["arm"] == "compositional"]) - a
    c = sum(r["ok"] for r in deep if r["arm"] == "monolithic")
    d_ = len([r for r in deep if r["arm"] == "monolithic"]) - c
    if (a + b) and (c + d_):
        p = fisher_exact(a, b, c, d_)
        clo, chi = wilson(a, a + b)
        mlo, mhi = wilson(c, c + d_)
        print(f"\npooled over depth >= 2 (conditions genuinely differ there)")
        print(f"  compositional {a}/{a+b} [{clo:.2f},{chi:.2f}]"
              f"   monolithic {c}/{c+d_} [{mlo:.2f},{mhi:.2f}]")
        print(f"  Fisher exact two-sided p = {p:.2e}")

    d1 = [r for r in rows if r["k"] == 1]
    if d1:
        byarm = defaultdict(list)
        for r in d1:
            byarm[r["arm"]].append(r["ok"])
        print("\nnoise floor from depth-1 rungs (prompts identical across arms)")
        for arm, v in sorted(byarm.items()):
            print(f"  {arm:<15} {sum(v)}/{len(v)} = {sum(v)/len(v):.2f}")

    print(f"\ngraded {len(rows)} tasks")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {RESULTS / f'{label}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="vsi-sonnet5")
    args = ap.parse_args()
    report(grade_all(), args.label)


if __name__ == "__main__":
    main()
