"""A `GRPOTrainer` that accepts one advantage per token instead of per sequence.

`pdd.factored` computes an advantage per (rollout, target). Delivering it to the
optimiser needs the trainer to stop broadcasting one scalar across a whole
completion. TRL's loss does

    per_token_loss1 = coef_1 * advantages.unsqueeze(1)

with `advantages` of shape `(B,)` and `coef_1` of shape `(B, T)`. Handing it a
`(B, T)` advantage does not work: `unsqueeze(1)` would make it `(B, 1, T)`,
which broadcasts against `(B, T)` to `(B, B, T)` and yields a number that is
neither an error nor the loss anyone wanted. So the loss is overridden rather
than fed.

**This file reimplements a piece of TRL's internals and is therefore the most
version-fragile thing in the package.** It is written against trl 0.24, checks
what it can at runtime, and refuses loudly rather than training on a silently
wrong objective. `--factored off` avoids it entirely and is the fallback if a
future trl moves the ground.
"""

from __future__ import annotations

import warnings

#: The trl series this was written against. A different minor version is not
#: automatically wrong, but it is unverified.
KNOWN_TRL = ("0.24", "0.23", "0.22")


def _check_trl() -> str:
    import trl

    version = getattr(trl, "__version__", "unknown")
    if not any(version.startswith(v) for v in KNOWN_TRL):
        warnings.warn(
            f"pdd.factored_trainer was written against trl {KNOWN_TRL[0]}, "
            f"found {version}. The token-level loss override may no longer "
            "match trl's own. Verify one step against --factored off before "
            "trusting a run, or use --factored off.",
            RuntimeWarning,
            stacklevel=2,
        )
    return version


#: Private trl methods this override depends on. Checked once, at construction,
#: because discovering a rename halfway through a paid GPU run is the expensive
#: way to find out.
REQUIRED = ("_generate_and_score_completions", "_compute_loss")
#: Names trl has used for the per-token log-probability helper across versions.
LOGP_METHODS = (
    "_get_per_token_logps",
    "_get_per_token_logps_and_entropies",
)


def preflight(base_cls) -> str:
    """Fail now, with instructions, rather than during training."""
    missing = [m for m in REQUIRED if not hasattr(base_cls, m)]
    logp = next((m for m in LOGP_METHODS if hasattr(base_cls, m)), None)
    if missing or logp is None:
        raise RuntimeError(
            "this trl exposes a different internal API than "
            "pdd.factored_trainer overrides "
            f"(missing: {missing or []}"
            f"{'' if logp else ', no per-token logp helper'}). "
            "Re-run with --factored off, which uses stock trl and is "
            "unaffected."
        )
    return logp


def build(base_cls):
    """Return a subclass of `base_cls` using per-token advantages.

    Taken as an argument rather than imported so that this module can be loaded
    on a machine without trl, which is where most of the development happens.
    """
    import torch

    logp_method = preflight(base_cls)

    class FactoredGRPOTrainer(base_cls):
        """GRPO with advantages resolved per token.

        `advantages` may arrive as `(B,)`, in which case behaviour is
        unchanged and this is exactly the parent trainer, or as `(B, T)`, in
        which case each token carries the advantage of the target whose span
        it falls in.
        """

        #: Set by the reward function each step: a list of per-token advantage
        #: rows, one per completion, in batch order. Consumed and cleared by
        #: `_prepare_factored`.
        pending_advantages: "list[list[float]] | None" = None

        def _prepare_factored(self, inputs):
            """Widen `advantages` from `(B,)` to `(B, T)` where alignment holds.

            The per-token rows are built by re-tokenizing the completion text,
            which is not guaranteed to reproduce the exact token sequence the
            model generated. A mismatch would shift every span and train on
            credit attached to the wrong tokens, which is worse than not
            factoring at all and would leave no trace in any metric.

            So each row is checked against the true completion length, and a row
            that does not line up keeps the scalar advantage TRL already
            computed for it. Degrading per row rather than per batch means one
            odd completion costs one completion's worth of resolution.
            """
            rows = self.pending_advantages
            self.pending_advantages = None
            if rows is None:
                return inputs

            mask = inputs.get("completion_mask")
            ids = inputs.get("completion_ids")
            scalar = inputs.get("advantages")
            if mask is None or ids is None or scalar is None:
                warnings.warn(
                    "trainer inputs lack completion_mask/completion_ids/"
                    "advantages; keeping scalar advantages",
                    RuntimeWarning, stacklevel=2,
                )
                return inputs
            if scalar.dim() != 1:
                return inputs

            batch, width = ids.shape
            if len(rows) != batch:
                raise RuntimeError(
                    f"factored advantages cover {len(rows)} completions but "
                    f"the batch holds {batch}. The reward function and the "
                    "trainer disagree about batch order, which would train on "
                    "misattributed credit."
                )

            lengths = mask.sum(dim=1).tolist()
            dense = scalar.detach().float().unsqueeze(1).repeat(1, width)
            aligned = 0
            for i, row in enumerate(rows):
                true_len = int(lengths[i])
                # The generated sequence usually carries a trailing EOS that
                # decodes to nothing, so one token shorter is expected and fine.
                if not row or not (true_len - 1 <= len(row) <= true_len):
                    continue
                dense[i, :len(row)] = torch.tensor(
                    row, dtype=dense.dtype, device=dense.device
                )
                if len(row) < width:
                    dense[i, len(row):] = 0.0
                aligned += 1

            self.factored_alignment = aligned / batch if batch else 0.0
            if self.factored_alignment < 0.5:
                warnings.warn(
                    f"only {self.factored_alignment:.0%} of completions "
                    "re-tokenized to their generated length, so most rollouts "
                    "fell back to scalar advantages. Factoring is not doing "
                    "much; check the tokenizer or use --factored off.",
                    RuntimeWarning, stacklevel=2,
                )
            inputs["advantages"] = dense * mask.to(dense.dtype)
            return inputs

        def _generate_and_score_completions(self, generation_batch):
            inputs = super()._generate_and_score_completions(generation_batch)
            return self._prepare_factored(inputs)

        def _compute_loss(self, model, inputs):
            adv = inputs.get("advantages")
            if adv is None or adv.dim() == 1:
                # Nothing factored this step: the parent's own loss is correct
                # and is preferred, since it is the version that is tested.
                return super()._compute_loss(model, inputs)
            return self._token_level_loss(model, inputs)

        # -- the overridden objective -------------------------------------

        def _token_level_loss(self, model, inputs):
            """GRPO's clipped surrogate, with `(B, T)` advantages.

            Mirrors trl's own loss. Kept deliberately small: KL and the
            entropy-masking refinements are not reimplemented, so a run that
            needs them should use `--factored off`.
            """
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            helper = getattr(self, logp_method)
            per_token_logps = helper(
                model, input_ids, attention_mask, logits_to_keep
            )
            # The newer helper returns (logps, entropies).
            if isinstance(per_token_logps, tuple):
                per_token_logps = per_token_logps[0]

            old = inputs.get("old_per_token_logps")
            if old is None:
                old = per_token_logps.detach()

            advantages = inputs["advantages"]
            coef_1 = torch.exp(per_token_logps - old)
            eps_low = getattr(self, "epsilon_low", 0.2)
            eps_high = getattr(self, "epsilon_high", 0.2)
            coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)

            loss1 = coef_1 * advantages
            loss2 = coef_2 * advantages
            per_token_loss = -torch.min(loss1, loss2)

            beta = getattr(self, "beta", 0.0)
            ref = inputs.get("ref_per_token_logps")
            if beta and ref is not None:
                per_token_kl = (
                    torch.exp(ref - per_token_logps)
                    - (ref - per_token_logps)
                    - 1
                )
                per_token_loss = per_token_loss + beta * per_token_kl

            denom = completion_mask.sum().clamp(min=1.0)
            return (per_token_loss * completion_mask).sum() / denom

    FactoredGRPOTrainer.__name__ = "FactoredGRPOTrainer"
    return FactoredGRPOTrainer


def make(trl_module=None):
    """Convenience: check the version and subclass `trl.GRPOTrainer`."""
    _check_trl()
    if trl_module is None:
        from trl import GRPOTrainer as base
    else:
        base = trl_module.GRPOTrainer
    return build(base)
