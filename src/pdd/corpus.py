"""The generated corpus: many IFC theories, each a ladder in its own right.

A theory enters the corpus only when two things hold. Its reference proofs all
compile, and its dependency graph has been extracted by necessity rather than
declared. A spec without an extracted graph is not loadable, because every rung
would appear to be depth 1 and any measurement taken on it would be an artifact.

Task counting distinguishes two quantities, and conflating them would overstate
the corpus badly:

*   a task **shape** is a (target, withheld-set) pair, which is a structurally
    distinct proof obligation;
*   a task **instance** is a shape rendered under a particular renaming seed.

Instances of one shape share their mathematics and differ only in identifiers.
They are genuinely distinct tasks to a policy, which is the point of renaming,
but they are not independent evidence about proving at depth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from .ladder import Family, family_from_spec

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "corpus"
GRAPHS = ROOT / "results"


def load_generated() -> list[Family]:
    """Every generated theory whose graph has been extracted."""
    out: list[Family] = []
    for spec_path in sorted(SPECS.glob("*.json")):
        spec = json.loads(spec_path.read_text())
        graph = GRAPHS / f"graph-{spec['name']}.json"
        if not graph.exists():
            continue
        data = json.loads(graph.read_text())
        if data.get("method") != "necessity":
            raise ValueError(f"{graph} was not produced by necessity extraction")
        out.append(family_from_spec(spec, data["direct"]))
    return out


def task_shapes(fam: Family, target: str) -> int:
    """Withheld sets for `target`: itself plus any up-closed subset of its cone.

    Up-closedness is required because a supplied lemma carries its own proof,
    which cites its own dependencies; withholding one of those would leave the
    supplied proof referring to a name that no longer exists.
    """
    anc = fam.transitive_deps(target)
    if len(anc) > 12:
        return 2 ** 12
    total = 0
    for r in range(len(anc) + 1):
        for sub in combinations(anc, r):
            withheld = set(sub) | {target}
            # Only the target's own cone is ever rendered. Lemmas above the
            # target are absent from the file entirely, so they impose no
            # constraint; checking them would wrongly rule out every rung that
            # has a dependent, which is most of them.
            supplied = [k for k in anc if k not in withheld]
            if all(not (withheld & set(fam.by_key[k].deps)) for k in supplied):
                total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="renaming seeds per generated theory")
    args = ap.parse_args()

    from .families import FAMILIES

    gen = load_generated()
    hand = [FAMILIES[n] for n in sorted(FAMILIES)]

    print(f"generated theories with extracted graphs: {len(gen)}")
    print(f"hand-written families: {len(hand)}\n")

    by_depth: Counter[int] = Counter()
    per_theory: list[tuple[str, int, int, int]] = []
    shapes_total = 0

    for fam in gen + hand:
        n = 0
        for k in fam.by_key:
            s = task_shapes(fam, k)
            n += s
            by_depth[fam.depth_of(k)] += s
        # renaming applies to generated theories and to `noninterference`;
        # `vsi` cannot be renamed without breaking its own imports.
        seeds = 1 if fam.name == "vsi" else args.seeds
        per_theory.append((fam.name, len(fam.rungs), n, n * seeds))
        shapes_total += n

    print(f"{'theory':<20}{'rungs':>6}{'shapes':>9}{'instances':>11}")
    for name, rungs, shapes, inst in per_theory:
        print(f"{name:<20}{rungs:>6}{shapes:>9}{inst:>11}")

    total_inst = sum(t[3] for t in per_theory)
    print(f"\n{'TOTAL':<20}{sum(t[1] for t in per_theory):>6}"
          f"{shapes_total:>9}{total_inst:>11}")

    print("\ntask shapes per depth")
    for d in sorted(by_depth):
        bar = "#" * max(1, by_depth[d] * 40 // max(by_depth.values()))
        print(f"  depth {d}: {by_depth[d]:>5}  {bar}")


if __name__ == "__main__":
    main()
