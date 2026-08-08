"""A buffer of kernel-verified proofs, filled from rollouts GRPO already paid for.

GRPO is on-policy: it samples a batch, computes advantages, takes one step, and
discards everything. In this environment that throws away the scarcest thing
produced, a proof that the Lean kernel accepted. A 400-step run at 8 prompts and
8 rollouts sees 25,600 completions and roughly a fifth of their targets check,
so several thousand verified proofs are generated and then deleted.

Keeping them costs nothing. The rollouts are already graded, `expert_iter.mine`
already turns graded rollouts into supervised examples, and the only missing
piece is somewhere to put them and a decision about when to train on them.

Why it is worth doing, in one sentence: policy-gradient methods reweight what
the policy already does, whereas supervised fine-tuning on verified proofs can
move probability onto sequences the policy currently almost never samples. The
pass@1-versus-pass@k pattern in the first measured run is what sharpening looks
like, and this is the standard remedy.

The buffer deduplicates and caps. Deduplication matters more here than usual:
early in training the policy emits the same one-line proof for the same easy
lemma hundreds of times, and a buffer that counted each would be almost entirely
`rfl`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .expert_iter import Example, mine


@dataclass
class ProofBuffer:
    """Deduplicated, capped store of verified (prompt, completion) pairs.

    Eviction is oldest-first among the *most common* kinds rather than plain
    FIFO, so that a late run does not consist entirely of the easy lemmas the
    policy has learned to emit reliably. `by_key` counts what is in there, which
    is the diagnostic worth printing.
    """

    capacity: int = 20_000
    examples: list[Example] = field(default_factory=list)
    _seen: set = field(default_factory=set)
    added: int = 0
    rejected_duplicates: int = 0

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def by_source(self) -> "dict[str, int]":
        return dict(Counter(e.source for e in self.examples))

    @property
    def by_depth(self) -> "dict[int, int]":
        return dict(sorted(Counter(e.depth for e in self.examples).items()))

    def extend(self, examples: "list[Example]") -> int:
        """Add new examples, returning how many were actually novel."""
        fresh = 0
        for ex in examples:
            key = (ex.prompt, ex.completion)
            if key in self._seen:
                self.rejected_duplicates += 1
                continue
            self._seen.add(key)
            self.examples.append(ex)
            fresh += 1
        self.added += fresh
        self._evict()
        return fresh

    def harvest(self, rollouts, format_example: bool = True) -> int:
        """Mine a batch of graded rollouts straight into the buffer.

        Called from the training reward function, where the rollouts already
        exist and have already been graded, so this adds no Lean calls.
        """
        return self.extend(mine(rollouts, format_example))

    def _evict(self) -> None:
        if len(self.examples) <= self.capacity:
            return
        # Drop from whichever (kind, depth) bucket is over-represented, oldest
        # first, until back under capacity. Keeps rare deep proofs.
        while len(self.examples) > self.capacity:
            counts = Counter((e.source, e.depth) for e in self.examples)
            fattest = counts.most_common(1)[0][0]
            for i, ex in enumerate(self.examples):
                if (ex.source, ex.depth) == fattest:
                    victim = self.examples.pop(i)
                    self._seen.discard((victim.prompt, victim.completion))
                    break

    # -- persistence -------------------------------------------------------

    def load(self, path: Path) -> int:
        """Read back a corpus this class wrote. Returns how many were novel.

        Deduplication is what makes this safe to call on top of a live buffer:
        proofs already present are counted as duplicates rather than repeated.
        Nothing is re-verified, and nothing needs to be. Every row was accepted
        by the kernel when it was written, the kernel is deterministic, and the
        toolchain is pinned, so acceptance does not expire.
        """
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"no proof corpus at {path}")
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            messages = d["messages"]
            rows.append(
                Example(
                    prompt=messages[0]["content"],
                    completion=messages[1]["content"],
                    source=d.get("source", "whole"),
                    depth=d.get("depth", 0),
                    family=d.get("family", "?"),
                    targets=d.get("targets", 1),
                )
            )
        return self.extend(rows)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for ex in self.examples:
                fh.write(json.dumps(ex.as_row()) + "\n")
        return path

    def summary(self) -> str:
        return (
            f"{len(self)} verified proofs "
            f"({self.added} added, {self.rejected_duplicates} duplicates), "
            f"sources {self.by_source}, depths {self.by_depth}"
        )


def to_dataset(buffer: "ProofBuffer", max_rows: "int | None" = None):
    """Build a `datasets.Dataset` of prompt/completion rows for SFT.

    Returns `None` when there is nothing worth training on. A handful of
    examples would overfit the adapter to whatever the policy already does,
    which is the opposite of the point.
    """
    from datasets import Dataset

    rows = [
        {"prompt": e.prompt, "completion": e.completion}
        for e in buffer.examples
    ]
    if max_rows is not None:
        rows = rows[-max_rows:]
    if not rows:
        return None
    return Dataset.from_list(rows)
