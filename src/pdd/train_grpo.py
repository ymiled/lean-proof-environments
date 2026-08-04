"""GRPO against the Lean kernel, with Unsloth + TRL.

Run this on a CUDA machine. Unsloth is not available on Apple silicon, so the
intended split is: develop and sanity-check the environment locally, train on a
rented GPU. Nothing else in this package needs a GPU.

    uv run python -m pdd.sanity --model <served model> --depths 1 2   # local
    uv run python -m pdd.train_grpo --model unsloth/Qwen3-4B-Instruct # remote

## Why the loop is shaped like this

**Curriculum by stage, not by callback.** TRL fixes the dataset when the trainer
is built, so changing task difficulty mid-run means fighting the trainer. Each
stage instead builds its own dataset, trains, saves the adapter, and hands it to
the next stage. Stages are cheap; adapters are small.

**Reward comes from `rollout.Grader`, not from a fresh grader per call.** The
grader caches per-target verdicts across the whole run and pools Lean processes
across the entire batch of completions TRL hands it at once. Both matter: early
in training a group of eight rollouts is often five distinct responses, and a
verdict already computed is a verdict that does not need computing again.

**Prompts too long for the context are dropped, not truncated.** TRL truncates
prompts on the left, which on this task removes the definitions and leaves the
goals -- a task that cannot be solved from what remains, scored as if it could.
That is a silent corruption of the reward signal, so oversized tasks are
excluded at dataset build time instead.

**The degenerate-group rate is logged every step.** It is the honest measure of
whether the run is learning anything; see `rollout.group_report`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .env import TaskSampler, all_families, split_families
from .evaluate import evaluate_held_out
from .reward import RewardConfig
from .rollout import Grader, group_report
from .task import Task

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Stage:
    """One curriculum step: which tasks, and how long to stay on them.

    `steps` is a ceiling, not a plan. A stage the policy has already mastered is
    spent re-proving what it can prove, and a stage it cannot touch is spent on
    all-zero groups that produce no gradient. Both are detectable from the
    reward stream, so both end the stage early -- see `curriculum_callback`.
    """

    depths: tuple[int, ...] | None
    #: Lemmas withheld per task. `None` withholds the whole ancestor cone.
    volume: int | None
    steps: int
    #: Distinct prompts drawn for this stage.
    prompts: int = 512
    #: Advance once mean reward over the trailing window clears this.
    advance_reward: float = 0.55
    #: Abort if the trailing window is this degenerate: the groups are flat, so
    #: the updates are noise and a harder stage can only be worse.
    stall_degenerate: float = 0.97
    #: Reward batches in the trailing window, and the minimum before either
    #: rule may fire. Early batches are dominated by the policy learning the
    #: output format, which is not evidence about proving.
    window: int = 12
    min_batches: int = 24


#: Start where the reward is dense and the prompts are short, then deepen.
#: Volume is kept at 3 or more throughout: a single-target task scores in
#: {0, 1} no matter how it is graded, which is the degenerate case the shaped
#: reward exists to avoid.
DEFAULT_CURRICULUM = (
    Stage(depths=(1, 2), volume=3, steps=150),
    Stage(depths=(2, 3), volume=3, steps=200),
    Stage(depths=(3, 4), volume=4, steps=250),
    Stage(depths=None, volume=4, steps=300),
)


class Stalled(RuntimeError):
    """Raised when a stage produced no usable signal for long enough to stop."""


def curriculum_callback(log: "list[dict]", stage: Stage, state: dict):
    """A `TrainerCallback` that ends a stage on evidence rather than on a count.

    Defined inside a function because `transformers` is a training-only
    dependency, and importing it at module scope would stop this file being
    importable on a machine that only runs the environment.

    Window units are *reward batches*, not optimizer steps: gradient
    accumulation means the reward function is called more than once per step,
    and the reward stream is what is being judged.
    """
    from transformers import TrainerCallback

    class Curriculum(TrainerCallback):
        def on_step_end(self, args, state_, control, **kwargs):
            if len(log) < stage.min_batches:
                return control
            window = log[-stage.window:]
            if len(window) < stage.window:
                return control
            reward = sum(w["mean_reward"] for w in window) / len(window)
            degenerate = sum(w["degenerate_rate"] for w in window) / len(window)
            state["reward"] = reward
            state["degenerate"] = degenerate

            if degenerate >= stage.stall_degenerate:
                state["stalled"] = True
                control.should_training_stop = True
                return control
            if reward >= stage.advance_reward:
                state["advanced"] = True
                control.should_training_stop = True
            return control

    return Curriculum()


def build_dataset(
    sampler: TaskSampler,
    stage: Stage,
    tokenizer,
    max_prompt_tokens: int,
    format_example: bool = True,
):
    """Draw `stage.prompts` tasks and return (dataset, tasks).

    The dataset carries an index rather than the task itself: a `Task` holds its
    family's entire Lean source and is not something to serialise into an Arrow
    table.
    """
    from datasets import Dataset

    sampler.depths = stage.depths
    sampler.volume = stage.volume

    tasks: list[Task] = []
    rows: list[dict] = []
    oversized = 0
    attempts = 0
    while len(tasks) < stage.prompts and attempts < stage.prompts * 20:
        attempts += 1
        task = sampler.sample()
        prompt = task.prompt(format_example=format_example)
        if len(tokenizer(prompt)["input_ids"]) > max_prompt_tokens:
            oversized += 1
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "task_index": len(tasks),
            }
        )
        tasks.append(task)

    if not tasks:
        raise SystemExit(
            f"every sampled task exceeded {max_prompt_tokens} prompt tokens; "
            "raise --max-prompt-tokens or lower the stage's depth"
        )
    if oversized:
        print(f"  dropped {oversized} tasks over {max_prompt_tokens} prompt tokens")
    return Dataset.from_list(rows), tasks


def make_reward_fn(
    grader: Grader, tasks: "list[Task]", group_size: int, logs: "list[list]"
):
    """Adapt `Grader` to the callable TRL expects.

    TRL calls this once per step with every completion in the batch, which is
    exactly the batching `Grader` wants -- one pooled pass over all of them.
    """

    def reward_fn(completions, task_index, **kwargs) -> list[float]:
        texts = [
            c if isinstance(c, str) else c[0]["content"] for c in completions
        ]
        batch = [tasks[i] for i in task_index]
        rollouts = grader.grade_batch(batch, texts)
        report = group_report(rollouts, group_size)
        entry = {
            "mean_reward": report.mean_reward,
            "degenerate_rate": report.degenerate_rate,
            "mean_score": sum(r.shaped.score for r in rollouts) / len(rollouts),
            "solved": sum(1 for r in rollouts if r.shaped.binary),
            "n": len(rollouts),
        }
        for log in logs:
            log.append(entry)
        print(f"    reward {report} | {grader.stats}")
        return [r.reward for r in rollouts]

    reward_fn.__name__ = "lean_kernel_reward"
    return reward_fn


def _shim_vllm_guided_decoding() -> None:
    """Restore `vllm.sampling_params.GuidedDecodingParams` for trl.

    Unsloth caps trl at 0.24, and trl 0.24 imports that name unconditionally at
    module scope. vllm has since renamed the class to `StructuredOutputsParams`,
    so `from trl import GRPOTrainer` raises before any of our code runs. No pair
    of released versions satisfies both packages, which makes this a shim rather
    than a pin.

    It is safe because trl dereferences the name in exactly one place, guarded
    by `guided_decoding_regex` -- a knob this project never sets, since the
    output format is enforced by the grader rather than by constrained decoding.
    Aliasing to the successor class is still the right binding if that branch is
    ever taken.

    Deliberately narrow: if neither name exists, fail loudly rather than paper
    over a vllm that has moved further than this shim understands.
    """
    import vllm.sampling_params as sampling_params

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return
    successor = getattr(sampling_params, "StructuredOutputsParams", None)
    if successor is None:
        raise SystemExit(
            "vllm exposes neither GuidedDecodingParams nor "
            "StructuredOutputsParams; trl's vllm integration cannot be "
            "imported. Check the installed vllm and trl versions."
        )
    sampling_params.GuidedDecodingParams = successor


def _vllm_completer(model, tokenizer, args, adapter_dir: Path):
    """Sample from the current policy through Unsloth's colocated vLLM engine.

    Evaluation reuses the training engine rather than standing up a second one:
    a 4-bit base model loaded twice will not fit alongside the optimizer state.
    Unsloth serves an adapter to that engine by path, not from memory, so the
    adapter is written out first -- which the stage loop wants to do anyway.

    Sampled at the training temperature rather than greedily, because pass@k at
    temperature 0 is pass@1 repeated k times and says nothing new.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_completion_tokens,
    )
    lora = model.load_lora(str(adapter_dir))

    def complete(prompts: "list[str]") -> "list[str]":
        chats = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        outputs = model.fast_generate(
            chats, sampling_params=params, lora_request=lora
        )
        return [o.outputs[0].text for o in outputs]

    return complete


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default=str(ROOT / "runs" / "grpo"))
    ap.add_argument("--max-seq", type=int, default=10240)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-completion-tokens", type=int, default=768)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--group", type=int, default=8, help="rollouts per prompt")
    ap.add_argument("--batch", type=int, default=2, help="prompts per device step")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02, help="KL coefficient")
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel lean processes; roughly the machine's core count")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    ap.add_argument("--corpus", action="store_true", default=True,
                    help="train on the generated theories as well")
    ap.add_argument("--eval-tasks", type=int, default=12,
                    help="held-out tasks per depth, evaluated after each stage")
    ap.add_argument("--eval-group", type=int, default=4,
                    help="samples per held-out task, for pass@k")
    ap.add_argument("--no-format-example", action="store_true",
                    help="drop the output-shape example from training prompts; "
                         "matches the measurement prompt exactly, and costs a "
                         "small model most of its early reward")
    ap.add_argument("--binary-reward", action="store_true",
                    help="ablation: train against all-or-nothing task reward")
    ap.add_argument("--stages", default=None,
                    help="JSON list of {depths, volume, steps, prompts}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - CUDA only
        raise SystemExit(
            "unsloth is not installed or not supported here. It requires CUDA; "
            "Apple silicon is not supported. Use pdd.sanity locally and run "
            "this on a GPU host."
        ) from exc
    _shim_vllm_guided_decoding()
    from trl import GRPOConfig, GRPOTrainer

    if args.stages:
        defaults = Stage(depths=None, volume=3, steps=0)
        stages = tuple(
            Stage(
                depths=tuple(s["depths"]) if s.get("depths") else None,
                volume=s.get("volume"),
                steps=s["steps"],
                prompts=s.get("prompts", defaults.prompts),
                advance_reward=s.get("advance_reward", defaults.advance_reward),
                stall_degenerate=s.get("stall_degenerate",
                                       defaults.stall_degenerate),
                window=s.get("window", defaults.window),
                min_batches=s.get("min_batches", defaults.min_batches),
            )
            for s in json.loads(args.stages)
        )
    else:
        stages = DEFAULT_CURRICULUM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,  # vLLM backend, colocated with training
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    families = all_families(include_corpus=args.corpus)
    train_families, eval_families = split_families(families)
    print(f"train theories: {sorted(train_families)}")
    print(f"held out:       {sorted(eval_families)}")

    sampler = TaskSampler(
        families=train_families, rng=random.Random(args.seed)
    )
    # The ablation arm: zero everywhere, one point for a task fully proved.
    # Expressed through the same weights rather than a separate code path, so
    # the two arms differ in exactly the quantity under test.
    config = (
        RewardConfig(
            proved=0.0, all_bonus=1.0, compile_error=0.0, bad_axioms=0.0,
            banned=0.0, timeout=0.0, missing=0.0, unparseable=0.0,
        )
        if args.binary_reward
        else RewardConfig()
    )
    # One grader for the whole run, so its verdict cache spans every stage.
    grader = Grader(workers=args.workers, config=config)
    log: list[dict] = []
    evals: list[dict] = []

    for i, stage in enumerate(stages):
        print(f"\n=== stage {i}: depths={stage.depths} volume={stage.volume} "
              f"steps={stage.steps} ===")
        dataset, tasks = build_dataset(
            sampler,
            stage,
            tokenizer,
            args.max_prompt_tokens,
            format_example=not args.no_format_example,
        )
        trainer_config = GRPOConfig(
            output_dir=str(out / f"stage{i}"),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            num_generations=args.group,
            max_prompt_length=args.max_prompt_tokens,
            max_completion_length=args.max_completion_tokens,
            max_steps=stage.steps,
            temperature=args.temperature,
            beta=args.beta,
            logging_steps=1,
            save_steps=max(stage.steps // 4, 1),
            report_to="none",
            use_vllm=True,
            seed=args.seed,
        )
        # One log slice per stage, so the callback judges this stage's reward
        # stream rather than one inherited from an easier stage.
        stage_log: list[dict] = []
        state: dict = {}
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                make_reward_fn(grader, tasks, args.group, [log, stage_log])
            ],
            args=trainer_config,
            train_dataset=dataset,
        )
        trainer.add_callback(curriculum_callback(stage_log, stage, state))
        trainer.train()

        adapter_dir = out / f"stage{i}" / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        model.save_lora(str(adapter_dir))

        if state.get("advanced"):
            print(f"  advancing early: reward {state['reward']:+.3f} over the "
                  f"last {stage.window} batches")
        elif state.get("stalled"):
            print(f"  STALLED: {state['degenerate']:.0%} of groups degenerate "
                  f"over the last {stage.window} batches")

        # Held-out evaluation, by depth, on theories this run never trained on.
        # Training reward rises when a policy memorises the corpus just as
        # readily as when it learns to prove; this is the number that separates
        # the two, and it is the same curve the benchmark reports.
        report = evaluate_held_out(
            _vllm_completer(model, tokenizer, args, adapter_dir),
            grader,
            volume=stage.volume or 3,
            tasks_per_depth=args.eval_tasks,
            group=args.eval_group,
            seed=args.seed,
            format_example=not args.no_format_example,
            label=f"stage{i}",
        )
        print(report.table())
        evals.append(report.as_dict())
        (out / "log.json").write_text(
            json.dumps({"reward": log, "eval": evals}, indent=2) + "\n"
        )

        if state.get("stalled"):
            raise Stalled(
                f"stage {i} (depths={stage.depths}, volume={stage.volume}) "
                "produced no usable gradient. A harder stage cannot help. "
                "Lower the depth or volume, raise --temperature, or bootstrap "
                "with pdd.expert_iter first."
            )

    print(f"\ndone. adapters, reward log and held-out evals under {out}")


if __name__ == "__main__":
    main()
