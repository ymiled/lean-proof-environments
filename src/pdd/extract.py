"""Dependency extraction by necessity rather than by mention.

Why not read the proof term. The obvious way to get a true dependency graph is
to ask the Lean environment which constants a proof references. That is not
possible in Lean 4.32: theorem bodies are elaborated asynchronously and the term
is not retained, so `ConstantInfo.value?` returns `none` for every theorem, both
for locally elaborated declarations and for a compiled `.olean` loaded through
`withImportModules`. Only the statement's constants survive, which makes every
theorem look dependency-free. `docs/design-log.md` records the experiment.

What this does instead. An edge `T -> A` is asserted only when removing `A` from
the file actually breaks `T`'s proof. That is a stronger criterion than reading
the term, because a name can appear in a proof without being load-bearing, and
it is the criterion `selftest` already uses to catch decorative dependencies.

The subtlety is that removing `A` also breaks every supplied lemma that itself
needs `A`, and those failures would be attributed to `T`. Rungs are therefore
processed in source order, so by the time `T` is examined the graph for
everything before it is known, and the whole `A`-cone can be removed together.
A failure then means `T` needs something in that cone, which means `T`
transitively needs `A`. That yields the transitive relation directly; direct
edges are recovered afterwards by transitive reduction.

Cost is one compile per candidate edge, quadratic in the number of rungs. At the
scale of a single theory that is minutes, and the inner loop parallelises.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .families import FAMILIES
from .grader import grade_source
from .ladder import Family, Instance

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _render(inst: Instance, supplied: list[str], target: str) -> str:
    """A file containing the definitions, `supplied` with their proofs, and `target`."""
    parts = [inst.definitions]
    parts.extend(inst.render_rung(k) for k in supplied)
    parts.append(inst.render_rung(target))
    parts.append(f"#print axioms {inst.names[target]}")
    return "\n\n".join(parts) + "\n"


def necessity_closure(
    family: Family, inst: Instance, workers: int = 4, verbose: bool = True
) -> dict[str, set[str]]:
    """For each rung, the set of earlier rungs it transitively needs."""
    order = [r.key for r in family.rungs]
    closure: dict[str, set[str]] = {}

    for idx, target in enumerate(order):
        prior = order[:idx]
        if not prior:
            closure[target] = set()
            if verbose:
                print(f"  {target:<24} needs: -")
            continue

        def probe(a: str) -> tuple[str, bool]:
            # Remove `a` together with everything already known to need it, so a
            # failure cannot be blamed on a collaterally broken supplied lemma.
            cone = {a} | {b for b in prior if a in closure.get(b, set())}
            supplied = [k for k in prior if k not in cone]
            res = grade_source(
                _render(inst, supplied, target), [inst.names[target]]
            )
            return a, (not res.ok)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            needed = {a for a, req in pool.map(probe, prior) if req}

        closure[target] = needed
        if verbose:
            print(f"  {target:<24} needs: {sorted(needed) or '-'}")

    return closure


def transitive_reduction(closure: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """Direct edges: drop any `A` reachable from `T` through another dependency."""
    direct: dict[str, tuple[str, ...]] = {}
    for t, needs in closure.items():
        redundant = {a for b in needs for a in closure.get(b, set())}
        direct[t] = tuple(sorted(needs - redundant))
    return direct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    family = FAMILIES[args.family]
    inst = Instance.sample(family, args.seed)

    started = time.time()
    print(f"extracting {args.family} by necessity ({len(family.rungs)} rungs)\n")
    closure = necessity_closure(family, inst, workers=args.workers)
    direct = transitive_reduction(closure)
    elapsed = time.time() - started

    declared = {k: set(family.by_key[k].deps) for k in family.by_key}
    print("\n=== extracted vs declared ===")
    diffs = 0
    for k in [r.key for r in family.rungs]:
        ext, dec = set(direct[k]), declared[k]
        if ext == dec:
            print(f"  ok    {k:<24} {sorted(ext) or '-'}")
        else:
            diffs += 1
            print(f"  DIFF  {k:<24} extracted={sorted(ext)} declared={sorted(dec)}")

    def depth(k: str) -> int:
        return 1 + max((depth(d) for d in direct[k]), default=0)

    print(f"\n{diffs} discrepancies")
    print(f"max depth (necessity graph): {max(depth(k) for k in direct)}")
    print(f"extraction wall-clock: {elapsed:.1f}s "
          f"({sum(len(direct) and i for i in range(len(family.rungs)))} probes)")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"graph-{args.family}.json"
    out.write_text(json.dumps(
        {"family": args.family,
         "method": "necessity",
         "elapsed_s": round(elapsed, 1),
         "closure": {k: sorted(v) for k, v in closure.items()},
         "direct": {k: list(v) for k, v in direct.items()}},
        indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
