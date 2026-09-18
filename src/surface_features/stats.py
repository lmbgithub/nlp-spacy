"""Effect sizes, because a significant difference is not necessarily a real one.

With tens of thousands of comments almost any difference in means clears a
significance test while remaining far too small to build on. Cohen's *d*
answers the question that actually matters: how large is the gap relative to
the spread inside each group?
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Conventional Cohen bands, with an extra "negligible" floor below 0.1. Under
# roughly 0.1 the distributions overlap so heavily that no classifier can
# exploit the difference, which is a more useful verdict than "small".
_BANDS: tuple[tuple[float, str], ...] = (
    (0.1, "negligible"),
    (0.2, "small"),
    (0.5, "modest"),
    (0.8, "medium"),
)


@dataclass(frozen=True, slots=True)
class Effect:
    """One feature's separation between the two groups."""

    feature: str
    positive_mean: float
    negative_mean: float
    cohens_d: float

    @property
    def magnitude(self) -> str:
        return magnitude(self.cohens_d)

    @property
    def difference(self) -> float:
        return self.positive_mean - self.negative_mean


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Sample variance (ddof=1); NaN for fewer than two observations."""
    n = len(values)
    if n < 2:
        return float("nan")
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (n - 1)


def cohens_d(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Standardised mean difference using the pooled standard deviation.

    Returns NaN rather than raising when the statistic is undefined: a group of
    fewer than two observations has no sample variance, and two constant groups
    have zero pooled spread. NaN propagates into the report as a blank cell,
    which is honest; a sentinel of 0.0 would read as "measured, no effect".
    """

    n_pos, n_neg = len(positive), len(negative)
    if n_pos < 2 or n_neg < 2:
        return float("nan")

    var_pos, var_neg = variance(positive), variance(negative)
    pooled_var = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
    if pooled_var <= 0:
        return float("nan")

    return (mean(positive) - mean(negative)) / math.sqrt(pooled_var)


def magnitude(d: float) -> str:
    """Label a *d* value, using its absolute size: direction is reported separately."""
    if math.isnan(d):
        return "undefined"
    size = abs(d)
    for threshold, label in _BANDS:
        if size < threshold:
            return label
    return "large"


def effect_table(
    columns: dict[str, tuple[Sequence[float], Sequence[float]]],
) -> list[Effect]:
    """Build the per-feature effect table, largest absolute effect first.

    NaN effects sort last rather than arbitrarily: an undefined statistic is
    not a strong result and must never head the ranking.
    """

    effects = [
        Effect(
            feature=name,
            positive_mean=mean(list(pos)),
            negative_mean=mean(list(neg)),
            cohens_d=cohens_d(list(pos), list(neg)),
        )
        for name, (pos, neg) in columns.items()
    ]
    return sorted(
        effects,
        key=lambda e: (
            math.isnan(e.cohens_d),
            -abs(e.cohens_d) if not math.isnan(e.cohens_d) else 0.0,
        ),
    )


def format_table(effects: Sequence[Effect]) -> str:
    """Render the effect table as fixed-width text for a terminal transcript."""
    header = f"{'feature':<24}{'hostile':>10}{'other':>10}{'cohen_d':>10}  magnitude"
    lines = [header, "-" * len(header)]
    for e in effects:
        lines.append(
            f"{e.feature:<24}{e.positive_mean:>10.4f}{e.negative_mean:>10.4f}"
            f"{e.cohens_d:>10.4f}  {e.magnitude}"
        )
    return "\n".join(lines)
