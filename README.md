# Lean proof environments

A Lean 4 reinforcement learning environment for program verification. A candidate proof is checked by type-checking against tae kernel. Task difficulty is set by how much of a lemma's dependency cone is withheld from the model. We define a lemma's depth as the longest dependency chain ending at it: 1 with no dependencies, otherwise one more than the deepest lemma it needs.

## The result

Grading a task all-or-nothing, distorts the difficulty curve. Under binary grading one
model's pass rate fell `1.00, 0.75, 0.33, 0.00` across depths 1 to 4, a decay of
1.74x per level. Regrading the identical responses by how many of each task's
lemmas individually check gives `1.00, 0.92, 0.73, 0.73`, a decay of 1.12x. 

## The formalization

A Volpano-Smith-Irvine security type system with subtyping, while loops, and
functions. 784 lines of Lean 4, no `sorry`, soundness audited to depend on
`propext` and `Quot.sound`. The corpus
is fourteen theories, 158 lemmas, 2,607 distinct tasks.

## The environment

- **The reward is the kernel.** Compilation, an allowed axiom set under
  `#print axioms`. The axiom check is not
  redundant with compilation because a lemma could fail, and Lean could still declare the theorem
  anyway, and everything citing it could still compiled.
- **Dependencies are measured.** An edge enters the graph by deleting a lemma and
  observing that the proof fails, establishing necessity rather than mention.
- **The model cannot change the question.** The harness emits the theorem
  statement; the model supplies only the tactic block. A weaker statement cannot
  be substituted.

## Training

GRPO against the kernel reward, LoRA on a 7B prover, one 24 GB GPU, about four
hours. Training saw depths 1 and 2. Evaluation is on four theories held out
by theory, so nothing in the evaluation shares a definition or a proof with
anything trained on.

| depth | pass@1 | per-target score |
| --- | --- | --- |
| 1 | 0.021 -> 0.083 | 0.056 -> 0.368 |
| 2 | 0.000 -> 0.000 | 0.049 -> 0.188 |
| 3 | 0.000 -> 0.000 | 0.076 -> 0.188 |
| 4 | 0.000 -> 0.000 | 0.028 -> 0.069 |
| all | 0.005 -> 0.021 | 0.052 -> 0.203 |

Depths 3 and 4 improved without being trained on.


## Layout

- [`lean/`](lean/) formalization and the extraction procedure
- [`src/pdd/`](src/pdd/) environment, grader, reward, training loop
- [`corpus/`](corpus/) generated theory specifications
- [`results/`](results/) dependency graphs and recorded runs

Everything except `pdd.train_grpo` runs on CPU.
