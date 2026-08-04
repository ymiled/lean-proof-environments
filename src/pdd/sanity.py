"""The check to run before renting a GPU.

GRPO learns by comparing rollouts sampled from one prompt. If every rollout in
a group scores the same, the group's advantages are all zero and it contributes
nothing to the update -- and a model that fails every attempt produces exactly
that. So the question that decides whether training is worth starting is not
"what is the pass rate", it is **"what fraction of rollout groups carry any
spread at all"**.

This script measures that, at the sampling settings and group size the trainer
will use, against a served local model. It reports four things:

*   mean shaped reward, and mean fraction of targets proved;
*   the share of groups that are degenerate, which is the share of the batch a
    trainer would be silently throwing away;
*   the verdict histogram, which says *how* the model is failing -- a wall of
    `banned_syntax` is a prompting problem, not a capability problem, and is
    fixed for free;
*   grading throughput, which sets the step time of the real loop.

Usage:

    uv run python -m pdd.sanity --model Qwen/Qwen3-4B --depths 1 2 \\
        --volume 3 --prompts 16 --group 8

Interpretation. Degenerate rate at or near 1.00 means do not start training at
these settings: lower the depth, lower the volume, or raise the temperature
before spending anything. Anything below about 0.7 is a workable signal.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .env import TaskSampler, all_families, split_families
from .policy import LocalPolicy
from .reward import RewardConfig
from .rollout import Grader, group_report

RESULTS = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model name the server knows")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2],
                    help="allowed depths for the deepest target; empty = all")
    ap.add_argument("--volume", type=int, default=3,
                    help="lemmas withheld per task; 0 means the whole cone")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8,
                    help="rollouts per prompt, matching the trainer's group size")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel lean processes")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="parallel requests to the serving endpoint")
    ap.add_argument("--corpus", action="store_true",
                    help="include the generated theories, not just vsi/noninterference")
    ap.add_argument("--format-example", action="store_true",
                    help="append a worked example of the output shape; use the "
                         "same setting here as in training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the raw record here")
    args = ap.parse_args()

    families = all_families(include_corpus=args.corpus)
    train, _ = split_families(families)
    sampler = TaskSampler(
        families=train,
        depths=tuple(args.depths) or None,
        volume=args.volume or None,
        rng=random.Random(args.seed),
    )

    policy = LocalPolicy(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    grader = Grader(workers=args.workers, config=RewardConfig())

    # Each prompt repeated `group` times, contiguously, so `group_report` can
    # slice groups back out by position.
    prompts = sampler.batch(args.prompts)
    tasks = [t for t in prompts for _ in range(args.group)]
    texts = policy.complete_many(
        [t.prompt(format_example=args.format_example) for t in tasks]
    )

    empty = sum(1 for t in texts if not t.strip())
    if empty == len(texts):
        raise SystemExit(
            f"every request to {args.base_url} came back empty -- "
            "is the server up and is --model the name it serves?"
        )

    rollouts = grader.grade_batch(tasks, texts)
    report = group_report(rollouts, args.group)

    verdicts: Counter[str] = Counter()
    for r in rollouts:
        verdicts.update(r.shaped.per_target.values())

    mean_score = sum(r.shaped.score for r in rollouts) / len(rollouts)
    solved = sum(1 for r in rollouts if r.shaped.binary)
    prompt_chars = sum(
        len(t.prompt(format_example=args.format_example)) for t in prompts
    ) / len(prompts)
    unparsed = sum(1 for r in rollouts if not r.shaped.parsed) / len(rollouts)

    print(f"model            {policy.name}")
    print(f"tasks            {args.prompts} prompts x {args.group} rollouts, "
          f"depths {args.depths or 'all'}, volume {args.volume or 'cone'}")
    print(f"mean prompt      {prompt_chars:,.0f} chars "
          f"(~{prompt_chars / 3.6:,.0f} tokens)")
    print(f"empty responses  {empty}/{len(texts)}")
    print()
    print(f"mean reward      {report.mean_reward:+.3f}")
    print(f"mean targets     {mean_score:.3f} proved")
    print(f"fully solved     {solved}/{len(rollouts)}")
    print(f"degenerate       {report.degenerate}/{report.groups} groups "
          f"({report.degenerate_rate:.0%})")
    print()
    print("per-target verdicts")
    total = sum(verdicts.values())
    for verdict, n in verdicts.most_common():
        print(f"  {verdict:<16}{n:>6}  {n / total:>6.1%}")
    print()
    print(f"grading          {grader.stats}")
    print()
    # Order matters. A low degenerate rate is necessary for a group-relative
    # method to learn *something*, but it says nothing about learning the right
    # thing: rollouts that all fail still differ in reward, because a missing
    # block and a compile error score differently. Reward spread among uniformly
    # failing rollouts teaches formatting hygiene, not proving. So the count of
    # proved targets is checked first and overrides everything below it.
    if mean_score == 0.0:
        print("VERDICT: zero targets proved in the entire sample. GRPO cannot")
        print("         work here: what spread the groups have comes from")
        print("         *kinds of failure*, so the policy would learn to avoid")
        print("         penalties rather than to prove. Add worked proof")
        print("         examples to the prompt, lower --depths, or bootstrap")
        print("         with SFT. Note that pdd.expert_iter also needs a")
        print("         non-zero rate -- at exactly zero it mines nothing.")
    elif unparsed >= 0.5 and not args.format_example:
        print(f"VERDICT: {unparsed:.0%} of responses carried no "
              "`-- PROOF <name>` marker, so they")
        print("         could not be split across the task's targets and were "
              "scored")
        print("         as format failures. That is a prompting problem, not a "
              "proving")
        print("         problem. Re-run with --format-example before "
              "concluding anything")
        print("         about capability, and train with the same flag.")
    elif solved == 0:
        print(f"VERDICT: individual targets are proved ({mean_score:.1%} of "
              "them) but no task")
        print("         was ever completed. Expert iteration will mine the")
        print("         per-target successes; GRPO on whole-task reward will")
        print("         still see mostly flat groups. Bootstrap first.")
    elif report.degenerate_rate >= 0.95:
        print("VERDICT: no usable signal. Reduce --depths or --volume, or raise")
        print("         --temperature, before starting a training run.")
    elif report.degenerate_rate >= 0.7:
        print("VERDICT: thin signal. Trainable, but start the curriculum here")
        print("         rather than deeper, and expect slow early progress.")
    else:
        print("VERDICT: usable signal at these settings.")

    if args.out:
        record = {
            "model": policy.name,
            "settings": vars(args),
            "mean_reward": report.mean_reward,
            "mean_score": mean_score,
            "fully_solved": solved,
            "rollouts": len(rollouts),
            "degenerate_rate": report.degenerate_rate,
            "unparseable_rate": unparsed,
            "verdicts": dict(verdicts),
            "lean_calls": grader.stats.lean_calls,
            "seconds": grader.stats.seconds,
        }
        path = Path(args.out)
        if not path.is_absolute():
            path = RESULTS / path
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
