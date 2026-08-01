"""Plot pass@k against dependency depth and fit the monolithic decay constant.

The quantity of interest is the per-depth decay factor in the monolithic arm.
Theorem report "roughly a 3x reduction in successful problem count for every
additional dependency solved"; fitting

    log(pass@k) = log(a) - d * log(r)

gives an independent estimate of r on a different corpus.

Usage:  uv run python -m pdd.plot results/claude-opus-5@T1.0.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"monolithic": "#c1440e", "compositional": "#1f6f8b"}


def _series(summary: dict, condition: str) -> tuple[list[int], list[float]]:
    pts = sorted(
        (int(k.split(":")[1]), v["pass_at_k"])
        for k, v in summary.items()
        if k.startswith(condition + ":")
    )
    return [d for d, _ in pts], [p for _, p in pts]


def fit_decay(depths: list[int], rates: list[float]) -> tuple[float, float] | None:
    """Least-squares fit of log(pass) vs depth. Returns (rate_per_depth, r2).

    Zero-valued points are dropped rather than clamped: a floored zero would
    dominate a log-space fit and manufacture an arbitrarily steep slope.
    """
    pts = [(d, math.log(p)) for d, p in zip(depths, rates) if p > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(d for d, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((d - mx) ** 2 for d, _ in pts)
    if sxx == 0:
        return None
    slope = sum((d - mx) * (y - my) for d, y in pts) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * d + intercept)) ** 2 for d, y in pts)
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return math.exp(-slope), r2


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/reference.json")
    data = json.loads(path.read_text())
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for condition in ("compositional", "monolithic"):
        depths, rates = _series(summary, condition)
        if not depths:
            continue
        ax.plot(
            depths,
            rates,
            marker="o",
            color=COLORS[condition],
            label=condition,
            linewidth=2,
        )

    mono_d, mono_p = _series(summary, "monolithic")
    caption = ""
    fit = fit_decay(mono_d, mono_p) if mono_d else None
    if fit:
        rate, r2 = fit
        caption = f"monolithic decay: {rate:.2f}x per depth  (R²={r2:.2f})"
        ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)

    ax.set_xlabel("dependency depth")
    ax.set_ylabel(f"pass@{data['k']}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sorted(set(mono_d)) or [1])
    ax.set_title(f"{data['policy']}   (k={data['k']}, seeds={data['seeds']})")
    if caption:
        ax.text(
            0.98,
            0.94,
            caption,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#c1440e",
        )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = path.with_suffix(".png")
    fig.savefig(out)
    print(f"wrote {out}")
    if fit:
        print(caption)
        print("Theorem's reported figure for comparison: ~3x per dependency")


if __name__ == "__main__":
    main()
