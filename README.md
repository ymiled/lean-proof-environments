# Lean proof environments

A Lean 4 reinforcement learning environment for program verification. A candidate proof is checked by type-checking against the kernel. Task difficulty is set by how much of a lemma's dependency cone is withheld from the model. We define a lemma's depth as the longest dependency chain ending at it: 1 with no dependencies, otherwise one more than the deepest lemma it needs.

## The result

Grading a task all-or-nothing, distorts the difficulty curve. Under binary grading one
model's pass rate fell `1.00, 0.75, 0.33, 0.00` across depths 1 to 4, a decay of
1.74x per level. Regrading the identical responses by how many of each task's
lemmas individually check gives `1.00, 0.92, 0.73, 0.73`, a decay of 1.12x.

The compositional effect this predicts (supplying a task's dependencies helps,
withholding them hurts) holds up at scale. Pooled over depths 2-4 on 134 tasks, 
supplying dependencies vs. withholding them reaches
`p = 1.6e-3` after deduplicating identical policy responses, `p = 5.2e-4`
counting them as independent.

## The formalization

A Volpano-Smith-Irvine security type system with subtyping, while loops, and extended with
functions. 784 lines of Lean 4, `sorry`-free, soundness audited to depend on
`propext` and `Quot.sound`. The corpus is fourteen theories, 158 lemmas, 2,607 distinct tasks.

## The environment

- **The reward is the kernel.** Compilation, with an allowed axiom set under
  `#print axioms`. The axiom check is not
  redundant with compilation because a lemma could fail, and Lean could still declare the theorem
  anyway, and everything citing it could still compiled.
- **Dependencies are measured.** An edge enters the graph by deleting a lemma and
  observing that the proof fails.
- **The model cannot change the question.** The harness emits the theorem
  statement; the model supplies only the tactic block. A weaker statement cannot
  be substituted.

## Training

GRPO against the kernel reward, LoRA on `DeepSeek-Prover-V2-7B`, 4-bit,
colocated vLLM. Cold-started with expert iteration, then a curriculum of
depths (1,2) -> (2,3) -> (3,4) -> all, each stage measured by held-out
evaluation before the next stage begins. `theory` split: `vsi`, the written
formalization, is held out entirely, so nothing in evaluation shares a
definition or proof with anything trained on.

| depth | pass@1 | per-target score |
| --- | --- | --- |
| 1 | 0.221 | 0.753 |
| 2 | 0.000 | 0.510 |
| 3 | 0.000 | 0.310 |
| 4 | 0.000 | 0.171 |
| 5 | 0.000 | 0.044 |
| all | 0.044 | 0.358 |

We find the same distortion the calibration result predicts: pass@1 is zero
at every depth past 1. But per-target score has a different trend, decaying
smoothly through depth 4 rather than collapsing.


## Layout

- [`lean/`](lean/) formalization and the extraction procedure
- [`src/pdd/`](src/pdd/) environment, grader, reward, training loop
- [`corpus/`](corpus/) generated theory specifications
- [`results/`](results/) dependency graphs and recorded runs

Everything except `pdd.train_grpo` runs on CPU.
