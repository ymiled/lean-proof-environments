# Lean proof environments

This repository builds a Lean 4 environment for studying how proof-task
difficulty depends on dependency structure. It combines a machine-checked
information-flow security development with a kernel-based grader, configurable
withheld dependencies, and one-shot model evaluation.

The main result is a reward-design correction. In small model runs, grading a
task all-or-nothing made performance appear to collapse with depth. Regrading
the same responses by per-target credit recovered substantial partial progress:
on the active VSI corpus, a depth-4 cell changed from `0.00` to `0.46`. The
result suggests that a proof kernel provides finer feedback than a binary task
reward exposes, while the small sample limits claims about population-level
scaling.

The formal corpus is a 784-line Lean 4 Volpano–Smith–Irvine-style security type
system with subtyping, loops, and functions. Its soundness theorem is audited
by Lean to depend only on `propext` and `Quot.sound`. Dependencies are recovered
semantically by deletion tests rather than declared by hand.

The project also tests compositional structure: in the monolithic condition,
the model must reconstruct withheld ancestors; in the compositional condition,
they are supplied as trusted interfaces. A separate synthetic family supports
identifier renaming and contamination controls. The earlier Peano-arithmetic
ladder is archived and is not part of the active benchmark.

## Artifacts

- [Paper](docs/result.pdf)
- [Design history](docs/design-log.md)
- [Lean formalization](lean/README.md)
- [Results](results/)

## Verification

```bash
uv sync
uv run python -m pdd.selftest --family vsi
uv run python -m pdd.selftest --family noninterference
```

Both self-tests should report zero failures before running a sweep. The grader
requires successful compilation, an allowed Lean axiom set, and no banned
compiler-trusting tactics.

The active benchmark contains the `vsi` and `noninterference` families. Larger
model sweeps and within-depth identification experiments remain future work.
