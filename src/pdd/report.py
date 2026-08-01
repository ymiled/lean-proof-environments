"""Grade a completed run directory and print the result table.

Reports pass rates by depth and arm, and -- because the headline question is
whether a decay is about dependency depth or about sheer output volume -- also
prints the per-rung detail with reference proof length alongside. A run where a
long proof succeeds and a short one fails is direct evidence against the volume
explanation, and that is invisible in an aggregated curve.

    uv run python -m pdd.report --label ni-sonnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .families import FAMILIES
from .grader import grade
from .ladder import Instance
from .task import Task, parse_blocks

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args()

    runs = Path(args.runs)
    specs = json.loads((RUNS / "_specs.json").read_text())
    fam = FAMILIES[specs[0]["family"]]

    rows = []
    for s in specs:
        inst = Instance.sample(fam, s["seed"])
        task = Task(inst, tuple(s["targets"]), s["arm"])
        out = runs / f"{s['task_id']}.out"
        if out.exists():
            r = grade(task, parse_blocks(out.read_text(), task))
            verdict, ok = r.verdict.value, r.ok
        else:
            verdict, ok = "missing", False
        rows.append({
            **s,
            "verdict": verdict,
            "ok": ok,
            "target": s["targets"][-1],
            "target_loc": len(inst.reference_proof_of(s["targets"][-1]).splitlines()),
            "total_loc": sum(
                len(inst.reference_proof_of(k).splitlines()) for k in s["targets"]
            ),
        })

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.label}.json").write_text(json.dumps(rows, indent=2))

    print(f"=== {args.label} :: {fam.name} ===\n")
    print(f"{'depth':>5} {'arm':>14} {'lemmas':>7} {'pass':>6} {'n':>3}")
    for d in sorted({r["k"] for r in rows}):
        for arm in ("compositional", "monolithic"):
            sel = [r for r in rows if r["k"] == d and r["arm"] == arm]
            if not sel:
                continue
            lem = f"{min(len(r['targets']) for r in sel)}-" \
                  f"{max(len(r['targets']) for r in sel)}"
            print(f"{d:>5} {arm:>14} {lem:>7} "
                  f"{sum(r['ok'] for r in sel) / len(sel):>6.2f} {len(sel):>3}")

    # Per-rung detail. The point of printing length next to outcome is that a
    # long success beside a short failure refutes the volume explanation
    # directly, without needing a regression nobody has power for.
    for arm in ("compositional", "monolithic"):
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        print(f"\n{arm} detail:")
        for r in sorted(sel, key=lambda x: (x["k"], x["target"])):
            mark = "ok  " if r["ok"] else "FAIL"
            print(f"  [{mark}] d{r['k']} {r['target']:<18} "
                  f"target={r['target_loc']:>2}L total={r['total_loc']:>3}L "
                  f"{r['verdict']}")


if __name__ == "__main__":
    main()
