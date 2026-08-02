---
title: "Lean 4 won't tell you what a proof used"
subtitle: "So I recovered dependency structure by deleting lemmas, and it caught four fabricated ones"
geometry: margin=1.15in
fontsize: 11pt
mainfont: "Palatino"
monofont: "Menlo"
colorlinks: true
---

I was building a benchmark that measures how well models prove lemmas that
depend on other lemmas. Difficulty is set by dependency depth, so the whole
thing rests on knowing which lemma actually needs which.

## The obvious approach doesn't work

Ask Lean. It just checked the proof, so it knows what the proof used.

It doesn't. Lean 4 elaborates theorem bodies asynchronously and does not retain
the term. Two lines after a theorem is declared, its proof is unreachable:

```lean
theorem foo (n : Nat) : n = n := rfl
theorem bar (n : Nat) : n = n := foo n     -- cites foo

env.find? `bar                  -- value?.isSome = false
env.toKernelEnv.find? `bar      -- value?.isSome = false
liftCoreM (getConstInfo `bar)   -- value?.isSome = false
liftCoreM (collectAxioms `bar)  -- []
```

Compiling to `.olean` and loading through `withImportModules` gives the same
answer, so it isn't staleness of an in-memory map. Only the *statement's*
constants survive. Run over my main development, extraction reported all 46
theorems as dependency-free, max depth 1.

## Delete it and see what breaks

So don't ask what a proof mentions. Assert `T → A` only when removing `A` makes
`T` fail to compile.

This is stronger than reading the proof term even where reading is possible: a
name can appear in a proof without being load-bearing, and what the benchmark
needs is necessity, not mention.

One subtlety makes it work. Removing `A` also breaks every supplied lemma that
itself needs `A`, and those failures would be blamed on `T`. So rungs are
processed in source order, the graph for everything earlier is already known,
and the entire `A`-cone is removed together. A failure then means `T` needs
something in that cone, hence transitively needs `A`. Direct edges follow by
transitive reduction.

## It caught four fabrications

The dependency graph had been written by hand. Running the extractor against it:

```
                    declared edge            closure        verdict
     lowEq_symm  ->  lowEq_refl                  []        REJECTED
    lowEq_trans  ->  lowEq_refl                  []        REJECTED
   Lvl.le_trans  ->  Lvl.le_refl                 []        REJECTED
     secure_sub  ->  Lvl.le_refl                 []        REJECTED

   sub_base_inv  ->  Lvl.le_refl            present        accepted
    confinement  ->  secure_sub             present        accepted
```

Four of fourteen edges were fiction. `lowEq_symm` claimed to need `lowEq_refl`,
and its entire proof is:

```lean
theorem lowEq_symm {G : Ctx} {s t : St} (h : lowEq G s t) : lowEq G t s := by
  intro x hx; exact (h x hx).symm
```

Reflexivity cannot help you swap two states. Deleting `lowEq_refl` from the file
entirely, `lowEq_symm` still compiles clean, no axioms. Reported max depth was
6; the true value is 5, so every depth-indexed number from that family had been
wrong.

## The part I'd want you to notice

There was already a check for exactly this. It had reported zero failures for
the family's entire life, and it was **structurally incapable of failing.**

It graded a multi-lemma task using a single-lemma solution, so the harness
filled the missing proofs with `sorry`, the axiom audit rejected the attempt,
and the rejection was read as "the dependency is real." It was asking *did this
fail?* when it needed to ask *did this fail because the dependency was needed?*
The failure branch was unreachable.

Nothing looked wrong. Everything compiled, the audit worked correctly, the
suite printed `0 failure(s)`. I found it by reading the dependency listing and
noticing that a one-line proof cannot plausibly need a helper lemma.

The rule I now apply, and the reason this writeup exists: **a check that has
never failed is not evidence that nothing is wrong.** It is an untested branch.
Every check in the repository is now exercised against a case it is supposed to
reject before it is trusted, and I corrupt the graph file on purpose to watch
both refusals fire.

---

Cost of the guarantee: one compile per candidate edge, 91 probes in 52 s for a
14-lemma development, parallelisable. Across the full corpus, 660 probes in
261 s. That is what makes machine-established dependency structure practical
rather than merely preferable.

Code, corpus, and a log of every other thing that went wrong:
`github.com/ymiled/proof-depth-decay`
