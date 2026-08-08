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

**Expert iteration runs first, in this process.** GRPO cannot bootstrap itself
from a policy that never succeeds; see `pdd.bootstrap`. `--ei-rounds 0` skips it
and reproduces the original cold-start behaviour.

**The dataset is rebuilt every `--refresh-every` steps.** TRL fixes its dataset
at construction, so dynamic sampling -- keeping prompts that produce spread and
dropping ones that do not -- happens at cycle boundaries instead of inside a
step. See `pdd.dynamic`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .dynamic import Ledger, PromptPool, curate
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


def dataset_from_tasks(tasks: "list[Task]", format_example: bool = True):
    """Wrap tasks as the Arrow table TRL wants.

    The row carries an index rather than the task itself: a `Task` holds its
    family's entire Lean source and is not something to serialise into an Arrow
    table.
    """
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user",
                     "content": task.prompt(format_example=format_example)}
                ],
                "task_index": i,
            }
            for i, task in enumerate(tasks)
        ]
    )


def build_dataset(
    sampler: TaskSampler,
    stage: Stage,
    tokenizer,
    max_prompt_tokens: int,
    format_example: bool = True,
    *,
    pool: "PromptPool | None" = None,
    rng: "random.Random | None" = None,
    oversample: int = 4,
    explore: float = 0.35,
    count: "int | None" = None,
):
    """Draw `stage.prompts` tasks under dynamic sampling, and return (ds, tasks).

    With a fresh pool every prompt is unknown and this reduces to drawing at
    random, which is what it did before dynamic sampling existed. With a pool
    carried across cycles it becomes DAPO's filter: prompts observed to produce
    no spread are retired in favour of ones that do, and a reserved share of the
    dataset still goes to prompts nobody has tried.

    The pool is pinned to this stage's difficulty. Changing depth or volume
    invalidates its contents -- the prompts are the wrong difficulty even though
    the ledger's verdicts about them remain true -- so the pool is cleared and
    refilled, while the ledger carries over.
    """
    sampler.depths = stage.depths
    sampler.volume = stage.volume
    pool = pool if pool is not None else PromptPool()
    if pool.reconfigure((stage.depths, stage.volume)):
        print(f"  pool: reset for depths={stage.depths} volume={stage.volume}")

    result = curate(
        sampler,
        pool,
        count=count if count is not None else stage.prompts,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        format_example=format_example,
        oversample=oversample,
        explore=explore,
        rng=rng,
    )
    if not result.tasks:
        raise SystemExit(
            f"every sampled task exceeded {max_prompt_tokens} prompt tokens; "
            "raise --max-prompt-tokens or lower the stage's depth"
        )
    print(f"  dataset: {result}")
    return dataset_from_tasks(result.tasks, format_example), result.tasks


def make_reward_fn(
    grader: Grader,
    tasks: "list[Task]",
    group_size: int,
    logs: "list[list]",
    *,
    trainer_ref: "dict | None" = None,
    tokenizer=None,
    scale_advantages: bool = True,
    buffer=None,
    format_example: bool = True,
    ledger: "Ledger | None" = None,
):
    """Adapt `Grader` to the callable TRL expects.

    TRL calls this once per step with every completion in the batch, which is
    exactly the batching `Grader` wants -- one pooled pass over all of them.

    When `trainer_ref` is supplied it is a one-key box holding the trainer, and
    this function additionally computes per-token advantages and parks them on
    it for the loss to pick up. Doing it here rather than in the trainer is what
    keeps the kernel verdicts, which only exist at this point, attached to the
    completions they came from.
    """
    factored_on = trainer_ref is not None and tokenizer is not None

    def reward_fn(completions, task_index, **kwargs) -> list[float]:
        texts = [
            c if isinstance(c, str) else c[0]["content"] for c in completions
        ]
        batch = [tasks[i] for i in task_index]
        rollouts = grader.grade_batch(batch, texts)
        report = group_report(rollouts, group_size)
        if ledger is not None:
            # Recorded here rather than in a callback because this is the only
            # point where the graded rollouts and the prompts they came from
            # are both in hand. The next cycle's dataset is built from it.
            ledger.observe(rollouts, group_size)
        entry = {
            "mean_reward": report.mean_reward,
            "degenerate_rate": report.degenerate_rate,
            "mean_score": sum(r.shaped.score for r in rollouts) / len(rollouts),
            "solved": sum(1 for r in rollouts if r.shaped.binary),
            "n": len(rollouts),
        }

        if buffer is not None:
            # Free: these rollouts are already sampled and already graded, so
            # keeping the verified ones costs no Lean calls and no GPU time.
            fresh = buffer.harvest(rollouts, format_example)
            entry["buffer"] = len(buffer)
            entry["buffer_fresh"] = fresh

        if factored_on:
            rows, stats = _token_advantages(
                rollouts, group_size, tokenizer, scale_advantages
            )
            trainer = trainer_ref.get("trainer")
            if trainer is not None:
                trainer.pending_advantages = rows
            entry.update(stats)
            print(
                f"    factored rescued {stats['rescued_groups']}/"
                f"{stats['groups']} flat groups, span coverage "
                f"{stats['coverage']:.0%}"
            )

        for log in logs:
            log.append(entry)
        print(f"    reward {report} | {grader.stats}")
        return [r.reward for r in rollouts]

    reward_fn.__name__ = "lean_kernel_reward"
    return reward_fn


def distil_buffer(model, tokenizer, buffer, *, steps: int, lr: float,
                  batch: int, max_seq: int, out_dir) -> "dict | None":
    """One supervised pass over the verified proofs collected so far.

    Run between curriculum stages rather than inside a stage, because splicing
    a second optimiser into TRL's loop mid-stage fights its scheduler and its
    reference-model bookkeeping for no benefit that a stage boundary does not
    already give.

    Every example here was produced by the policy and accepted by the kernel, so
    no reference proof enters the corpus and there is nothing to leak. The
    reason to do it at all is that policy gradient can only reweight sequences
    the policy already samples, while this can move mass onto proofs it found
    once and would otherwise forget.
    """
    from .replay import to_dataset

    dataset = to_dataset(buffer)
    if dataset is None or len(dataset) < 32:
        print(f"  distil: skipped, only {len(buffer)} verified proofs banked")
        return None

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        print("  distil: trl SFTTrainer unavailable, skipped")
        return None

    print(f"  distil: {buffer.summary()}")
    common = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=1,
        learning_rate=lr,
        max_steps=steps,
        logging_steps=max(1, steps // 5),
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    # trl renamed this field between versions; try the current name first.
    try:
        config = SFTConfig(max_length=max_seq, **common)
    except TypeError:
        config = SFTConfig(max_seq_length=max_seq, **common)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
    )
    result = trainer.train()
    return {
        "examples": len(dataset),
        "steps": steps,
        "train_loss": float(result.training_loss),
        "by_source": buffer.by_source,
        "by_depth": buffer.by_depth,
    }


def _config(cls, **kwargs):
    """Build a TRL config, dropping fields this trl does not have.

    The DAPO knobs -- `epsilon_high`, `scale_rewards`, `loss_type` -- have
    appeared, been renamed and changed type across trl releases, and unsloth
    pins the version from its side. Passing an unknown one is a `TypeError`
    thrown after the model is loaded, which on a rented GPU is money. Filtering
    against the dataclass's own fields turns that into a printed line, and the
    printed line is what tells you a correction silently did not apply.
    """
    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(kwargs) - valid)
    if dropped:
        print(f"  note: this trl has no {dropped}; those settings are inactive")
    return cls(**{k: v for k, v in kwargs.items() if k in valid})


def _token_advantages(rollouts, group_size, tokenizer, scale):
    """Per-token advantage rows for a batch, plus what to log about them.

    `rescued_groups` is the number the whole idea rests on: groups whose scalar
    rewards are identical, so GRPO's advantage is zero throughout, but whose
    per-target outcomes differ and therefore still carry a usable gradient.
    """
    from .factored import coverage, factor_batch, per_token_advantages

    groups = factor_batch(rollouts, group_size, scale=scale)
    rows: list[list[float]] = []
    rescued = 0
    covered: list[float] = []

    for gi, group in enumerate(groups):
        chunk = rollouts[gi * group_size:(gi + 1) * group_size]
        rewards = [r.reward for r in chunk]
        flat = len(chunk) > 1 and max(rewards) - min(rewards) <= 1e-9
        if flat and group.usable:
            rescued += 1
        for credits, rollout in zip(group.credits, chunk):
            covered.append(coverage(credits))
            enc = tokenizer(
                rollout.text, add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]
            rows.append(per_token_advantages(credits, offsets))

    return rows, {
        "groups": len(groups),
        "rescued_groups": rescued,
        "coverage": sum(covered) / len(covered) if covered else 0.0,
    }


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


def _vllm_completer(model, tokenizer, args, adapter_dir: Path,
                    temperature: "float | None" = None):
    """Sample from the current policy through Unsloth's colocated vLLM engine.

    Evaluation reuses the training engine rather than standing up a second one:
    a 4-bit base model loaded twice will not fit alongside the optimizer state.
    Unsloth serves an adapter to that engine by path, not from memory, so the
    adapter is written out first -- which the stage loop wants to do anyway.

    Sampled at the training temperature rather than greedily, because pass@k at
    temperature 0 is pass@1 repeated k times and says nothing new. `temperature`
    overrides it, which expert iteration uses: that stage wants coverage rather
    than a faithful picture of what the policy usually does.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=args.temperature if temperature is None else temperature,
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
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL coefficient. Zero by default: the KL term pulls "
                         "the policy toward a base model that cannot do this "
                         "task, and at beta=0 trl also stops materialising a "
                         "reference model, which is most of a 4090's spare "
                         "memory")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="lower clip bound of the PPO ratio")
    ap.add_argument("--epsilon-high", type=float, default=0.28,
                    help="upper clip bound, held above --epsilon on purpose "
                         "(DAPO's clip-higher). The symmetric bound caps how "
                         "fast a rare token's probability can rise, and the "
                         "tactics worth learning here start rare, so a "
                         "symmetric clip collapses entropy before they are "
                         "found. Set equal to --epsilon to disable")
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
    ap.add_argument("--factored", default="on", choices=("on", "off"),
                    help="give each target its own advantage over its own "
                         "tokens, instead of one scalar per rollout. 'off' is "
                         "stock trl and the fallback if a trl upgrade breaks "
                         "the loss override")
    ap.add_argument("--scale-advantages", action="store_true",
                    help="divide centred advantages by the group standard "
                         "deviation. Off by default (the Dr. GRPO correction): "
                         "the division upweights low-variance groups, and "
                         "variance here tracks difficulty, so it systematically "
                         "overweights the prompts that taught the least")
    ap.add_argument("--distil", default="on", choices=("on", "off"),
                    help="bank every kernel-verified proof produced during "
                         "training and fine-tune on them between stages. Free "
                         "to collect: the rollouts are already graded")
    ap.add_argument("--distil-steps", type=int, default=60,
                    help="SFT steps per between-stage distillation pass")
    ap.add_argument("--distil-lr", type=float, default=1e-5)
    ap.add_argument("--buffer-capacity", type=int, default=20_000)
    ap.add_argument("--stages", default=None,
                    help="JSON list of {depths, volume, steps, prompts}")
    ap.add_argument("--split", default="binops",
                    choices=("binops", "theory", "both"),
                    help="which theories are held out. 'binops' is the "
                         "robustness check and keeps numbers comparable with "
                         "earlier runs; 'theory' holds out the hand-written "
                         "formalization and is the one to quote for transfer")
    ap.add_argument("--seed", type=int, default=0)

    ei = ap.add_argument_group(
        "expert iteration",
        "Cold start. GRPO's update is proportional to the reward spread inside "
        "a rollout group, so a policy that never succeeds produces flat groups "
        "and zero gradient however long it runs. This stage needs a success "
        "somewhere in the corpus, not several on one prompt.",
    )
    ei.add_argument("--ei-rounds", type=int, default=3,
                    help="sample/grade/mine/fine-tune rounds before GRPO. "
                         "0 skips the bootstrap entirely")
    ei.add_argument("--ei-tasks", type=int, default=192,
                    help="distinct tasks per round")
    ei.add_argument("--ei-samples", type=int, default=8,
                    help="rollouts per task. More is cheap relative to its "
                         "value here: one success anywhere yields an example, "
                         "and vLLM shares the prompt prefix across them")
    ei.add_argument("--ei-depths", type=int, nargs="*", default=[1, 2],
                    help="depths to mine. Deep tasks a cold policy cannot "
                         "touch yield nothing and cost the same to grade")
    ei.add_argument("--ei-volume", type=int, default=3)
    ei.add_argument("--ei-temperature", type=float, default=1.2,
                    help="above the training temperature deliberately: this "
                         "stage wants coverage, and every sample is "
                         "kernel-checked before it can become training data")
    ei.add_argument("--ei-sft-steps", type=int, default=120)
    ei.add_argument("--ei-sft-lr", type=float, default=1e-5)
    ei.add_argument("--ei-stop-score", type=float, default=0.55,
                    help="stop bootstrapping once the mean per-target score "
                         "clears this; past it the groups GRPO draws already "
                         "have spread")

    dyn = ap.add_argument_group(
        "dynamic sampling",
        "DAPO's filter, applied at the dataset because TRL fixes its dataset "
        "when the trainer is built. See pdd.dynamic.",
    )
    dyn.add_argument("--refresh-every", type=int, default=50,
                     help="rebuild the dataset from the difficulty ledger "
                          "every this many optimizer steps. A stage runs as a "
                          "sequence of such cycles; 0 disables refreshing and "
                          "runs each stage as one fixed dataset")
    dyn.add_argument("--oversample", type=int, default=4,
                     help="pool size as a multiple of the dataset size. This "
                          "is the slack dynamic sampling has: at 1 the pool is "
                          "the dataset, and retiring a prompt can only be "
                          "answered with a fresh unknown one")
    dyn.add_argument("--explore", type=float, default=0.35,
                     help="share of each dataset reserved for prompts the "
                          "ledger has never seen. Without it a curriculum "
                          "cannot advance: every prompt at a new depth is "
                          "unknown, so a purely exploitative filter would "
                          "never draw one")
    dyn.add_argument("--on-stall", default="rescue",
                     choices=("rescue", "abort"),
                     help="what to do when a stage's groups go flat. 'rescue' "
                          "runs an expert-iteration round at that stage's "
                          "depths and retries, which is what an unattended "
                          "paid run wants; 'abort' is the old behaviour")
    dyn.add_argument("--stall-retries", type=int, default=1)
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

    factored = args.factored == "on"
    if factored:
        from .factored_trainer import build, _check_trl

        _check_trl()
        trainer_cls = build(GRPOTrainer)
        print("advantages: factored per target over their own token spans")
    else:
        trainer_cls = GRPOTrainer
        print("advantages: one scalar per rollout (stock trl)")

    buffer = None
    if args.distil == "on":
        from .replay import ProofBuffer

        buffer = ProofBuffer(capacity=args.buffer_capacity)
        print("distillation: banking verified proofs, SFT between stages")

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
    train_families, eval_families = split_families(families, args.split)
    print(f"train theories: {sorted(train_families)}")
    print(f"held out:       {sorted(eval_families)} (split={args.split})")

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
    distils: list[dict] = []
    # One pool for the whole run, holding one ledger. It is what makes a cycle
    # boundary mean something: the next dataset is built from what the last one
    # did, and the prompts persist long enough for that to be true.
    pool = PromptPool()
    ledger = pool.ledger
    rng = random.Random(args.seed + 1)
    format_example = not args.no_format_example

    def sft(buf, steps, lr, out_dir):
        return distil_buffer(
            model, tokenizer, buf, steps=steps, lr=lr,
            batch=args.batch, max_seq=args.max_seq, out_dir=out_dir,
        )

    def completer_factory(adapter_dir, temperature=None):
        return _vllm_completer(
            model, tokenizer, args, adapter_dir, temperature=temperature
        )

    def run_expert_iteration(rounds, depths, volume, tag) -> dict:
        from .bootstrap import bootstrap

        return bootstrap(
            model, tokenizer, sampler, grader,
            completer_factory=lambda d: completer_factory(
                d, temperature=args.ei_temperature
            ),
            sft=sft,
            out=out / tag,
            rounds=rounds,
            tasks=args.ei_tasks,
            samples=args.ei_samples,
            depths=depths,
            volume=volume,
            max_prompt_tokens=args.max_prompt_tokens,
            format_example=format_example,
            buffer=buffer,
            sft_steps=args.ei_sft_steps,
            sft_lr=args.ei_sft_lr,
            stop_score=args.ei_stop_score,
        )

    bootstrap_record = None
    if args.ei_rounds > 0:
        print(f"\n=== expert iteration: {args.ei_rounds} rounds at depths "
              f"{args.ei_depths}, volume {args.ei_volume} ===")
        bootstrap_record = run_expert_iteration(
            args.ei_rounds,
            tuple(args.ei_depths) if args.ei_depths else None,
            args.ei_volume,
            "bootstrap",
        )
    else:
        print("\nexpert iteration skipped (--ei-rounds 0); GRPO starts cold")

    def train_cycles(stage, index: int, stage_log: list, state: dict) -> None:
        """Run one stage as a sequence of dataset refreshes.

        Rebuilding the trainer per cycle is the price of dynamic sampling under
        TRL, and it is a small one: the LoRA weights live on `model` and
        survive, only the optimizer state resets. At `--beta 0` there is no
        reference model to rebuild either, which is what keeps this affordable
        on one 24 GB card.
        """
        import gc

        import torch

        done = 0
        cycle = 0
        span = args.refresh_every or stage.steps
        # TRL's `per_device_train_batch_size` counts completions, so a step
        # consumes `batch * accum / group` prompts -- typically one. Building
        # `stage.prompts` of them per cycle would tokenize thousands to use
        # dozens; twice what the cycle can reach leaves room to shuffle and
        # nothing more. `stage.prompts` stays the ceiling.
        per_step = max(1, (args.batch * args.accum) // args.group)
        while done < stage.steps:
            steps = min(span, stage.steps - done)
            count = min(stage.prompts, max(64, steps * per_step * 2))
            print(f"  -- cycle {cycle}: steps {done}..{done + steps}, "
                  f"{count} prompts --")
            dataset, tasks = build_dataset(
                sampler, stage, tokenizer, args.max_prompt_tokens,
                format_example=format_example,
                pool=pool, rng=rng, count=count,
                oversample=args.oversample, explore=args.explore,
            )
            trainer_config = _config(
                GRPOConfig,
                output_dir=str(out / f"stage{index}" / f"cycle{cycle}"),
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch,
                gradient_accumulation_steps=args.accum,
                num_generations=args.group,
                max_prompt_length=args.max_prompt_tokens,
                max_completion_length=args.max_completion_tokens,
                max_steps=steps,
                temperature=args.temperature,
                beta=args.beta,
                epsilon=args.epsilon,
                epsilon_high=args.epsilon_high,
                # Dr. GRPO: the per-group standard deviation in the denominator
                # upweights whichever prompts had the least spread, which here
                # are the ones that taught the least.
                scale_rewards=args.scale_advantages,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                use_vllm=True,
                seed=args.seed + cycle,
            )
            # The reward function needs the trainer to park per-token
            # advantages on, and the trainer needs the reward function to
            # construct. A box breaks the cycle.
            box: dict = {}
            trainer = trainer_cls(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[
                    make_reward_fn(
                        grader, tasks, args.group, [log, stage_log],
                        trainer_ref=box if factored else None,
                        tokenizer=tokenizer if factored else None,
                        scale_advantages=args.scale_advantages,
                        buffer=buffer,
                        format_example=format_example,
                        ledger=ledger,
                    )
                ],
                args=trainer_config,
                train_dataset=dataset,
            )
            box["trainer"] = trainer
            trainer.add_callback(curriculum_callback(stage_log, stage, state))
            trainer.train()

            done += steps
            cycle += 1
            print(f"  ledger: {ledger.summary()}")
            del trainer, box
            gc.collect()
            torch.cuda.empty_cache()

            if state.get("advanced") or state.get("stalled"):
                return

    for i, stage in enumerate(stages):
        print(f"\n=== stage {i}: depths={stage.depths} volume={stage.volume} "
              f"steps={stage.steps} ===")
        # One log slice per stage, so the callback judges this stage's reward
        # stream rather than one inherited from an easier stage.
        stage_log: list[dict] = []
        state: dict = {}
        train_cycles(stage, i, stage_log, state)

        # A stall means every group went flat, which no amount of further
        # policy-gradient steps can fix -- the gradient is zero, not small. The
        # thing that can fix it is more verified data at this difficulty, which
        # is what expert iteration produces.
        attempts = 0
        while (state.get("stalled") and args.on_stall == "rescue"
               and attempts < args.stall_retries):
            attempts += 1
            print(f"  STALLED at {state['degenerate']:.0%} degenerate; "
                  f"rescue {attempts}/{args.stall_retries} by expert iteration "
                  f"at depths {stage.depths}")
            run_expert_iteration(
                1, stage.depths, stage.volume, f"rescue{i}.{attempts}"
            )
            stage_log.clear()
            state.clear()
            train_cycles(stage, i, stage_log, state)

        # Distil the verified proofs banked during this stage before the
        # adapter is written, so the saved adapter is the one just evaluated.
        if buffer is not None:
            record = distil_buffer(
                model, tokenizer, buffer,
                steps=args.distil_steps,
                lr=args.distil_lr,
                batch=args.batch,
                max_seq=args.max_seq,
                out_dir=out / f"stage{i}" / "distil",
            )
            if record is not None:
                distils.append({"stage": i, **record})
            buffer.save(out / f"stage{i}" / "verified.jsonl")

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
            split=args.split,
            volume=stage.volume or 3,
            tasks_per_depth=args.eval_tasks,
            group=args.eval_group,
            seed=args.seed,
            format_example=format_example,
            label=f"stage{i}",
        )
        print(report.table())
        evals.append(report.as_dict())
        (out / "log.json").write_text(
            json.dumps(
                {
                    "reward": log,
                    "eval": evals,
                    "distil": distils,
                    "bootstrap": bootstrap_record,
                    "ledger": ledger.summary(),
                    "args": vars(args),
                },
                indent=2,
            ) + "\n"
        )

        if state.get("stalled"):
            raise Stalled(
                f"stage {i} (depths={stage.depths}, volume={stage.volume}) "
                "produced no usable gradient, and rescue did not recover it. "
                "Lower the depth or volume, raise --temperature, or raise "
                "--ei-rounds."
            )

    print(f"\ndone. adapters, reward log and held-out evals under {out}")


if __name__ == "__main__":
    main()
