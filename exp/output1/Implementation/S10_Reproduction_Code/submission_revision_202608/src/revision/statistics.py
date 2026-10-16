"""Exploratory paired-statistics utilities without additional dependencies.

Used only for month-level SA/GA comparisons across six months; small-sample outputs are explicitly marked exploratory."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _normal_two_sided_p(z: float) -> float:
    """Calculate a two-sided p value using a standard-normal approximation."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _wilcoxon_normal_approx(differences: list[float]) -> dict[str, Any]:
    """Implement Wilcoxon signed ranks with zero-difference removal, average ranks, and a normal approximation."""
    nonzero = [value for value in differences if abs(value) > 1e-12]
    if len(nonzero) < 2:
        return {"n": len(nonzero), "statistic": None, "z": None, "p_value": None}
    indexed = sorted(enumerate(nonzero), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(nonzero)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and abs(abs(indexed[end][1]) - abs(indexed[cursor][1])) <= 1e-12:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for pos in range(cursor, end):
            ranks[indexed[pos][0]] = average_rank
        cursor = end
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    n = len(nonzero)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    z = (min(positive, negative) - mean) / math.sqrt(max(variance, 1e-12))
    return {"n": n, "statistic": min(positive, negative), "z": z, "p_value": _normal_two_sided_p(z)}


def _holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Apply Holm multiplicity correction, preserving missing p values."""
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None]
    valid.sort(key=lambda item: item[1])
    adjusted: list[float | None] = [None] * len(p_values)
    previous = 0.0
    m = len(valid)
    for rank, (index, value) in enumerate(valid):
        candidate = min(1.0, float(value) * (m - rank))
        previous = max(previous, candidate)
        adjusted[index] = previous
    return adjusted


def exploratory_paired_comparison(frame: pd.DataFrame, metric: str, method_a: str, method_b: str) -> dict[str, Any]:
    """Report exploratory SA/GA Wilcoxon and Holm comparisons using months as paired instances."""
    pivot = frame.pivot(index="month_id", columns="method", values=metric).dropna(subset=[method_a, method_b])
    differences = (pivot[method_a] - pivot[method_b]).astype(float).tolist()
    wilcoxon = _wilcoxon_normal_approx(differences)
    adjusted = _holm_adjust([wilcoxon["p_value"]])[0]
    return {
        "analysis_label": "exploratory_paired_comparison",
        "metric": metric,
        "methods": [method_a, method_b],
        "paired_months": [str(value) for value in pivot.index.tolist()],
        "difference_definition": f"{method_a} - {method_b}",
        "differences": differences,
        "wilcoxon_signed_rank_normal_approximation": wilcoxon,
        "holm_adjusted_p_value": adjusted,
        "interpretation_note": 'Only six paired months are available, limiting statistical power; nonsignificance does not establish equivalence.',
    }
