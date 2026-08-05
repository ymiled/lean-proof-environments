"""Policies: things that emit a tactic block given a task.

The name is deliberate. In MDP terms these are $\\pi_\\theta$, and the rest of
the package is the environment. The measurement experiments hold these frozen;
`env.py` and `rollout.py` expose what a learner needs to stop holding them
frozen.

`LocalPolicy` exists for that second use. Before spending anything on a GPU it
is worth knowing whether the small model you intend to train scores strictly
above zero on the easiest tasks in the corpus, because a policy that never
succeeds cannot be improved by a method that learns from its successes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .task import Task, _clean_block, parse_blocks

class Policy(Protocol):
    name: str

    def act(self, task: Task) -> "dict[str, str]":
        """Return one tactic block per target."""
        ...


def _strip_fences(text: str) -> str:
    """Models add code fences despite being asked not to. Tolerate it.

    Being lenient here is correct: fence-stripping is a formatting concern, and
    charging a model a failure for it would confound presentation with proving
    ability, which is the thing we are trying to measure.

    This used to run over the whole response before it was split across
    targets, which quietly kept only the *first* fenced block and threw the
    rest away -- fine for a single goal, destructive for a model that opens one
    fence per lemma. Splitting now happens first and the cleanup runs per
    section, so this is only a thin alias over the shared implementation.
    """
    return _clean_block(text)


#: Public name for the same thing, kept because callers outside this module
#: import it.
strip_fences = _strip_fences


class ReferencePolicy:
    """Oracle. Always emits the known-good proof.

    Used to certify the benchmark and to provide the ceiling line on plots.
    """

    name = "reference"

    def act(self, task: Task) -> "dict[str, str]":
        return task.reference_solution()


class EmptyPolicy:
    """Floor. Emits a proof that never works.

    Gives the plot a zero line and catches graders that accept garbage.
    """

    name = "empty"

    def act(self, task: Task) -> "dict[str, str]":
        return {k: "  rfl" for k in task.targets}


class AnthropicPolicy:
    """A frozen Claude model, sampled at fixed temperature."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"{model}@T{temperature}"

    def act(self, task: Task) -> "dict[str, str]":
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": task.prompt()}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return parse_blocks(text, task)


class LocalPolicy:
    """Any OpenAI-compatible chat endpoint: vLLM, Ollama, LM Studio, TGI.

    Deliberately spoken over plain HTTP rather than through a client library.
    The three servers a small open-weights model is likely to sit behind all
    expose the same route, and depending on none of them means the sanity check
    runs wherever the model happens to be.

    `complete_many` exists because the check that matters is a pass rate over
    many samples, and a served model batches those far better than a loop does.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 1.0,
        max_tokens: int = 768,
        top_p: float = 0.95,
        api_key: str | None = None,
        timeout: float = 300.0,
        concurrency: int = 16,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        self.timeout = timeout
        self.concurrency = concurrency
        self.name = f"{model}@T{temperature}"

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(
                f"no OpenAI-compatible server at {self.base_url}: {exc}"
            ) from exc
        return payload["choices"][0]["message"]["content"] or ""

    def complete_many(self, prompts: "list[str]") -> list[str]:
        """Sample every prompt, keeping order. A failed request yields ""."""

        def one(p: str) -> str:
            try:
                return self.complete(p)
            except Exception:  # a serving failure is data, not a crash
                return ""

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(one, prompts))

    def act(self, task: Task) -> "dict[str, str]":
        return parse_blocks(self.complete(task.prompt()), task)
