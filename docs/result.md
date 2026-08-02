---
title: "A proof kernel gives you an honest reward, not a learnable one"
subtitle: "What I found building an RL environment for verification"
geometry: margin=1.15in
fontsize: 11pt
mainfont: "Palatino"
monofont: "Menlo"
colorlinks: true
---

The case for verification as an RL target is that a proof kernel is ground truth
rather than a proxy: unit tests degrade as a signal because coverage does not
track correctness, whereas a kernel cannot be fooled about whether a proof is a
proof.

I built the environment. That argument is half right, and the missing half is
the part that decides whether you can train on it.

## The environment

A ladder is a set of lemmas with a dependency graph. Difficulty is one knob: how
much of a lemma's dependency cone you withhold. Withhold nothing and the model
proves one lemma with everything beneath it supplied as trusted interfaces;
withhold the cone and it must rebuild the whole subtree. Reward is kernel
acceptance plus an audit of which axioms the result depends on.

14 theories, 158 lemmas, 2,607 distinct task shapes. Every dependency edge was
established by deleting a lemma and checking that the proof breaks, because Lean
4 discards proof terms after checking and will not tell you what a proof used.

## The kernel does not give you dense signal

Same environment, same reward function, three configurations:

| ladder | policy | pass rate by depth |
|---|---|---|
| arithmetic | Sonnet 5 | 1.00, 1.00, 1.00, 1.00 |
| security types (v1) | Sonnet 5 | 1.00, 0.00, 0.00, 0.00 |
| arithmetic | Haiku 4.5 | 1.00, 0.75, 0.33, 0.00 |

The first two produce **zero gradient**. Reward is constant, so there is nothing
to learn from, and both look perfectly healthy from outside: every task is
well-formed, every reward is correct, the suite is green. A dashboard cannot
tell you that an environment is teaching nothing.

Only the third has signal, and getting there was not a matter of picking a
harder theory. It took a weaker policy, or a rebuilt ladder.

**Dense feedback is not conferred by the proof assistant. It is a property of a
task distribution matched to a particular policy.** The kernel guarantees the
reward is *honest*. Nothing about it makes the reward *informative*. That is the
gap between having a verifier and having an environment, and it is the part that
fails silently.

## Difficulty is tunable, and the knob is decomposition

The failure mode is fixable, and the fix says something about compositional
verification directly.

The security ladder floored because its soundness theorem was one 37-line proof
with nothing between it and the trivial lemmas. Splitting it into two
intermediate lemmas, so the target became 22 lines, took that theorem from
**unprovable to proved** for the same policy at the same one-shot budget.

Nothing about the mathematics changed. Only the granularity of the trusted
interfaces did. That is the compositional-verification argument reproduced as a
property of the environment: the same corpus is unlearnable or learnable
depending on how finely you cut it.

Measured across the whole ladder, supplying a lemma's dependencies rather than
withholding them takes the pass rate from 4/14 to 18/27 (Fisher exact
$p=0.026$). Real, but thin, and I would not lead with it.

## The environment measures its own noise for free

A depth-1 lemma has no dependencies, so "supply its dependencies" and "withhold
them" render **byte-identical prompts**. I verified this with `diff`. They are
still dispatched to independent policy instances under different labels.

Identical input, different draws: 8/8 in one arm and 6/8 in the other. About
**25% run-to-run variance.**

That is a free noise-floor estimate on every sweep, and any environment of this
shape can build one in. It matters more than it sounds: without it I would have
read several single-sample cells as effects. With it, most individual cells in
my own results are visibly not significant.

## What I would tell someone building one of these

The grader, the task generator, the dependency machinery, the renaming, all of
it worked. None of it was the bottleneck.

The bottleneck was calibration, and it has no natural alarm. A task that always
succeeds and one that always fails both produce zero gradient, which is the same
sparse-reward failure usually attributed to unit tests, arriving by a different
route. If you are planning to scale RL on verification, the kernel is the easy
part. You need a calibration story, instrumentation to detect ceiling and floor,
and a difficulty knob fine enough to move between them.

---

Environment, corpus, extractor, and a log of everything that broke on the way:
`github.com/ymiled/proof-depth-decay`
