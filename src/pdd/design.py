"""Chain-versus-antichain contrasts: separating depth from volume.

The problem this solves. In any natural ladder, dependency depth, ancestor count
and total proof length move together -- measured at r > 0.96 in both shipped
families. So a pass@k-versus-depth curve cannot say whether success falls
because dependency chains *compound*, or merely because deep tasks require
*more output*. Those are different claims: under the second, compositional
verification helps for reasons unrelated to dependency structure, and any
chunking would do as well. Theorem's own reported difficulty cliff at "17+
marginal LoC" is a length effect, and in a ladder of uniform per-rung cost that threshold
falls between depth 3 and depth 4 -- so their length cliff predicts the same
curve with no depth effect existing at all.

The contrast. Reformulate a task as *prove this set of lemmas*, everything else
supplied. Then compare two task sets of the **same size** and comparable total
length:

*   **antichain** -- k mutually independent lemmas. Volume is k, chaining is
    absent.
*   **chain** -- k lemmas where each feeds the next. Volume is k, chaining is
    maximal.

Equal pass rates implicate volume; a gap implicates chaining. This maps directly
onto the wording under test: "three-fold reduction per additional dependency
solved" read as volume predicts decay in both arms, read as chaining predicts
decay only in the chain arm.

Two structural constraints, both discovered the hard way:

1.  **The withheld set must be up-closed.** A supplied lemma arrives with its
    reference proof, which cites its own dependencies, so those must be supplied
    too. Withholding a lemma therefore forces withholding everything above it.

2.  **The chain must be essential.** A purpose-built family with uniform lemma
    shape at every level was built for this and failed: because each rung was
    merely the previous rung composed once more, the chain could be bypassed
    entirely by unfolding the definitions and reusing the base lemma. Matching
    by construction requires uniformity, and uniformity destroys essential
    depth. `essentiality_report` checks this per family rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .ladder import Family, Instance


@dataclass(frozen=True)
class Contrast:
    """A matched pair of task sets at the same volume."""

    k: int
    chain: tuple[str, ...]
    antichain: tuple[str, ...]
    chain_loc: int
    antichain_loc: int

    @property
    def loc_gap(self) -> int:
        return abs(self.chain_loc - self.antichain_loc)


def _ancestors(family: Family) -> dict[str, set[str]]:
    return {k: set(family.transitive_deps(k)) for k in family.by_key}


def is_up_closed(family: Family, targets: tuple[str, ...]) -> bool:
    """No supplied lemma may depend on a withheld one.

    `targets` are withheld (the model must prove them); everything else is
    supplied with its reference proof. If a supplied lemma cited a target, its
    proof would reference a name that does not exist yet.
    """
    withheld = set(targets)
    for key, rung in family.by_key.items():
        if key in withheld:
            continue
        if withheld & set(rung.deps):
            return False
    return True


def is_chain(family: Family, group: tuple[str, ...]) -> bool:
    """Totally ordered under the ancestor relation."""
    anc = _ancestors(family)
    return all(
        (a in anc[b]) or (b in anc[a]) for a, b in combinations(group, 2)
    )


def is_antichain(family: Family, group: tuple[str, ...]) -> bool:
    """Pairwise incomparable: no member depends on any other, even indirectly."""
    anc = _ancestors(family)
    return all(
        (a not in anc[b]) and (b not in anc[a]) for a, b in combinations(group, 2)
    )


def find_contrasts(
    family: Family, instance: Instance, max_loc_gap: int = 4
) -> list[Contrast]:
    """Best LoC-matched chain/antichain pair at each achievable k.

    Only up-closed groups are eligible, and only k >= 2, since a single lemma is
    neither a chain nor an antichain in any meaningful sense.
    """
    keys = list(family.by_key)
    loc = lambda g: sum(  # noqa: E731
        len(instance.reference_proof_of(x).splitlines()) for x in g
    )

    out: list[Contrast] = []
    for k in range(2, len(keys) + 1):
        groups = [
            g for g in combinations(keys, k) if is_up_closed(family, g)
        ]
        chains = [g for g in groups if is_chain(family, g)]
        antis = [g for g in groups if is_antichain(family, g)]
        if not chains or not antis:
            continue
        best = min(
            ((c, a) for c in chains for a in antis),
            key=lambda p: abs(loc(p[0]) - loc(p[1])),
        )
        c, a = best
        if abs(loc(c) - loc(a)) > max_loc_gap:
            continue
        out.append(
            Contrast(k=k, chain=c, antichain=a, chain_loc=loc(c),
                     antichain_loc=loc(a))
        )
    return out


def essentiality_report(family: Family) -> dict[str, bool]:
    """Which rungs have a dependency that automation cannot route around.

    Structural precondition only: a rung whose dependencies are all supplied
    trivially, or which has none, cannot contribute essential depth. The real
    check is empirical and lives in `selftest`, which renders each rung with its
    ancestors removed and confirms the citing proof fails.

    Recorded here because a family can pass every compile check and still have
    a decorative chain -- the uniform purpose-built family did exactly that.
    """
    return {k: bool(r.deps) for k, r in family.by_key.items()}
