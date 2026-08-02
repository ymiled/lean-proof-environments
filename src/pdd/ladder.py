"""Ladder machinery: rungs, families, and per-seed renaming.

A **family** is a self-contained Lean theory plus a dependency DAG of lemmas
over it. Families live in `pdd.families`; nothing in this module knows any
mathematics, and nothing downstream of it knows which family it is handling.
Adding a theory is therefore one new module and no edits anywhere else.

Two invariants the rest of the package relies on:

1.  **Goals stay open in the recursion variable.** An early ladder
    stated its lemmas at a concrete point (`op_i one y = one`). Every such goal
    is a closed term, so `simp` does not reason about it, it evaluates it -- a
    depth-3 rung fell to `simp_all` with nothing in scope. Rungs must quantify
    over the variables the definitions recurse on so reduction is blocked and
    induction is genuinely required. `selftest` enforces this empirically by
    checking that each citing proof *fails* once its ancestors are removed.

2.  **Depth is derived, never declared.** `Family.depth_of` computes longest
    path through `deps`, so a rung cannot misreport its own difficulty.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property

#: Tactics the reference proofs stay inside. No Mathlib, no `decide`, no
#: `native_decide`: the environment stays self-contained and difficulty stays in
#: the proof structure rather than in library search.
TACTIC_VOCABULARY = (
    "induction", "simp", "rw", "rfl", "exact", "apply", "intro",
    "by_cases", "subst", "cases", "have",
)


@dataclass(frozen=True)
class Rung:
    """One lemma. `statement` and `proof` are format strings over the name map."""

    key: str
    role: str
    binders: str
    statement: str
    proof: str
    deps: tuple[str, ...] = ()
    raw: str = ""


@dataclass(frozen=True)
class Family:
    """A Lean theory plus its lemma DAG.

    `slots` maps each placeholder appearing in `definitions`/rungs to a pool
    kind, so renaming can draw collision-free identifiers of the right sort.
    """

    name: str
    definitions: str
    rungs: tuple[Rung, ...]
    slots: dict[str, str]

    @cached_property
    def by_key(self) -> dict[str, Rung]:
        return {r.key: r for r in self.rungs}

    def depth_of(self, key: str) -> int:
        rung = self.by_key[key]
        if not rung.deps:
            return 1
        return 1 + max(self.depth_of(d) for d in rung.deps)

    def transitive_deps(self, key: str) -> list[str]:
        """Ancestors of `key`, each appearing after its own dependencies."""
        seen: list[str] = []

        def visit(k: str) -> None:
            for d in self.by_key[k].deps:
                visit(d)
                if d not in seen:
                    seen.append(d)

        visit(key)
        return seen

    @property
    def depths(self) -> list[int]:
        return sorted({self.depth_of(k) for k in self.by_key})


# Identifier pools, kept free of any connotation that would hint at the
# underlying mathematics.
POOLS: dict[str, tuple[str, ...]] = {
    "type": ("Warp", "Quiver", "Strand", "Gleam", "Notch", "Spool", "Trellis",
             "Kern", "Plinth", "Vane"),
    "ctor": ("nil", "base", "root", "seed", "origin", "stem", "wisp", "flint",
             "cusp", "brim", "dart", "glint"),
    "fn": ("melt", "braid", "fuse", "weave", "bind", "knit", "forge", "spin",
           "twine", "clasp", "hew", "meld", "sift", "carve", "plait", "graft",
           "tamp", "hone", "quell", "drape"),
}


@dataclass
class Instance:
    """A renamed copy of a family, ready to render as Lean."""

    family: Family
    seed: int
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def sample(cls, family: Family, seed: int) -> "Instance":
        if family.name == "vsi":
            return cls(family=family, seed=seed,
                       names={r.key: r.key for r in family.rungs})
        rng = random.Random(seed)
        pools = {k: list(v) for k, v in POOLS.items()}
        for v in pools.values():
            rng.shuffle(v)

        names: dict[str, str] = {}
        for slot, kind in family.slots.items():
            names[slot] = pools[kind].pop()

        # Lemma names are opaque hex so the role leaks nothing. A model that
        # sees `add_comm` can match the goal to a remembered proof; `lem_a7f2`
        # forces it to read the statement.
        used: set[str] = set()
        for rung in family.rungs:
            while (nm := f"lem_{rng.randrange(16**4):04x}") in used:
                pass
            used.add(nm)
            names[rung.key] = nm

        return cls(family=family, seed=seed, names=names)

    @cached_property
    def definitions(self) -> str:
        if self.family.name == "vsi":
            return self.family.definitions
        return self.family.definitions.format(**self.names)

    def statement_of(self, key: str) -> str:
        r = self.family.by_key[key]
        if r.raw:
            return r.raw.split(":= by", 1)[0].rstrip()
        binders = r.binders.format(**self.names)
        space = " " if binders else ""
        return (
            f"theorem {self.names[key]}{space}{binders} : "
            f"{r.statement.format(**self.names)}"
        )

    def reference_proof_of(self, key: str) -> str:
        r = self.family.by_key[key]
        if r.raw:
            return r.raw.split(":= by", 1)[1].lstrip().rstrip()
        return r.proof.format(**self.names)

    def render_rung(self, key: str) -> str:
        if self.family.by_key[key].raw:
            return self.family.by_key[key].raw
        return f"{self.statement_of(key)} := by\n{self.reference_proof_of(key)}"

    def render_reference(self, key: str) -> str:
        """Full Lean file proving `key` with all ancestors, reference proofs throughout."""
        parts = [self.definitions]
        parts.extend(self.render_rung(k) for k in self.family.transitive_deps(key))
        parts.append(self.render_rung(key))
        parts.append(f"#print axioms {self.names[key]}")
        return "\n\n".join(parts) + "\n"


def family_from_spec(spec: dict, deps: dict[str, list[str]] | None = None) -> Family:
    """Build a `Family` from a JSON spec rather than a hand-written module.

    This is what lets a generated or imported theory become a ladder without
    anyone writing Python for it. The spec carries the Lean definitions, the
    rungs with their reference proofs, and the renaming slots; the dependency
    graph is supplied separately because it is *derived*, not authored. Passing
    `deps=None` means the graph has not been extracted yet, which is only valid
    while bootstrapping a new theory: every rung then looks depth 1, so
    `selftest` and any sweep would be meaningless until `pdd.extract` has run.

    Authoring dependencies by hand is exactly what produced four fabricated
    edges in an earlier family, so this function refuses to invent them.
    """
    required = {"name", "definitions", "rungs"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"family spec missing keys: {sorted(missing)}")

    rungs = []
    for r in spec["rungs"]:
        key = r["key"]
        rungs.append(Rung(
            key=key,
            role=r.get("role", ""),
            binders=r.get("binders", ""),
            statement=r.get("statement", ""),
            proof=r.get("proof", ""),
            deps=tuple(deps.get(key, ())) if deps is not None else (),
            raw=r.get("raw", ""),
        ))

    if deps is not None:
        known = {r.key for r in rungs}
        for k, ds in deps.items():
            if k not in known:
                raise ValueError(f"dependency graph names unknown rung {k!r}")
            for d in ds:
                if d not in known:
                    raise ValueError(f"rung {k!r} depends on unknown {d!r}")

    return Family(
        name=spec["name"],
        definitions=spec["definitions"],
        rungs=tuple(rungs),
        slots=spec.get("slots", {}),
    )
