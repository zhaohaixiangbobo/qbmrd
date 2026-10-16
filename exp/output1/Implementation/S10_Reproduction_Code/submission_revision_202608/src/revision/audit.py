"""Rule auditing, feasibility summaries, and submission metrics.

Formal ConstraintManager checks are conservatively classified as hard constraints unless the violation magnitude lies within numerical tolerance. Soft planning preferences are reported separately and are never conflated with violations."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


HARD_CONSTRAINTS = {
    "demand",
    "price",
    "volume",
    "fixed_allocation",
    "first_round_supply",
    "demand_based_priority",
    "price_based",
    "c_type",
}


def _violation_magnitude(item: dict[str, Any]) -> tuple[float, float]:
    """Extract violation magnitudes and comparison scales from backend records for numerical-tolerance classification."""
    if "excess" in item:
        return abs(float(item["excess"])), max(1.0, abs(float(item.get("available_supply", 0.0))))
    # Price and round-total records contain both lower and upper bounds; select the bound corresponding to the actual triggering direction,
    # otherwise a lower-bound deviation would be misreported as the distance across the full interval to the upper bound.
    violation_type = str(item.get("violation_type", "")).lower()
    if "actual_price" in item and "upper_limit" in item and "lower_limit" in item:
        limit_key = "lower_limit" if violation_type == "lower" else "upper_limit"
        limit_value = float(item[limit_key])
        return abs(float(item["actual_price"]) - limit_value), max(1.0, abs(limit_value))
    if "actual_volume" in item and "upper_limit" in item and "lower_limit" in item:
        limit_key = "lower_limit" if violation_type == "lower" else "upper_limit"
        limit_value = float(item[limit_key])
        return abs(float(item["actual_volume"]) - limit_value), max(1.0, abs(limit_value))
    pairs = [
        ("actual_total", "limit"),
        ("actual_ratio", "limit"),
        ("actual_rounds", "limit"),
    ]
    for actual, limit in pairs:
        if actual in item and limit in item:
            actual_value, limit_value = float(item[actual]), float(item[limit])
            return abs(actual_value - limit_value), max(1.0, abs(limit_value))
    return 1.0, 1.0


def count_applicable_checks(month: Any, results: dict[str, Any]) -> int:
    """Estimate the checks actually performed from data dimensions to provide a transparent denominator for normalized flag rates."""
    product = month.product_data
    rounds = list(month.rounds)
    n_products = int(len(product))
    enabled = set(results.get("summary", {}).get("enabled_constraints", []))
    checks = 0
    if "demand" in enabled:
        checks += n_products
    if "fixed_allocation" in enabled:
        checks += sum(int((series > 0).sum()) for series in month.data_loader.get_existing_allocations().values())
    if "first_round_supply" in enabled:
        checks += n_products
    if "volume" in enabled or "price" in enabled:
        checks += len(rounds) * int("volume" in enabled) + len(rounds) * int("price" in enabled)
    if "demand_based_priority" in enabled:
        checks += int(product.get('Demand_Based', pd.Series(dtype=str)).fillna("").astype(str).str.contains('Demand_Based').sum())
    if "price_based" in enabled:
        checks += len(rounds)
    if "c_type" in enabled:
        checks += len(rounds) * 3
    return max(1, checks)


def classify_validation(
    month: Any,
    validation: Any,
    relative_tolerance: float,
    absolute_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Separate backend validation into hard violations, numerical tolerances, soft rules, and review flags."""
    raw = validation.violations
    constraint_results = raw.get("constraint_results", {}) if isinstance(raw, dict) else {}
    hard_count = numerical_count = review_count = 0
    max_magnitude = 0.0
    rows: list[dict[str, Any]] = []
    for name, result in constraint_results.items():
        for index, item in enumerate(result.get("violations", []) or []):
            item_dict = item if isinstance(item, dict) else {"raw": item}
            magnitude, scale = _violation_magnitude(item_dict)
            max_magnitude = max(max_magnitude, magnitude)
            category = "hard_constraint_violation" if name in HARD_CONSTRAINTS else "post_solution_review_flag"
            # LP results undergo demand-preserving postprocessing; decimal residuals for case quantities and mean price per case can use
            # a reporting-precision tolerance. Dimensionless rules such as proportions and round counts still use only relative tolerance to avoid relaxing business ratios.
            is_quantity_or_price = any(
                key in item_dict for key in ("excess", "actual_volume", "actual_price", "actual_total")
            )
            tolerance = max(float(absolute_tolerance), relative_tolerance * scale) if is_quantity_or_price else relative_tolerance * scale
            if magnitude <= tolerance:
                category = "numerical_tolerance_flag"
                numerical_count += 1
            elif category == "hard_constraint_violation":
                hard_count += 1
            else:
                review_count += 1
            rows.append(
                {
                    "constraint_name": name,
                    "violation_index": index,
                    "category": category,
                    "magnitude": magnitude,
                    "scale": scale,
                    "detail": item_dict,
                }
            )

    applicable = count_applicable_checks(month, raw)
    return {
        "hard_constraint_violations": hard_count,
        "soft_rule_slack_activations": 0,
        "post_solution_review_flags": review_count,
        "numerical_tolerance_flags": numerical_count,
        "applicable_rule_checks": applicable,
        "normalized_review_flag_rate": float((hard_count + review_count + numerical_count) / applicable),
        "max_violation_magnitude": max_magnitude,
        "details": rows,
    }


def fill_rate(allocation: pd.DataFrame, product: pd.DataFrame, rounds: list[str]) -> float:
    """Calculate total allocation divided by total demand as an operational complement to GNAE."""
    total_allocated = float(allocation.loc[:, rounds].fillna(0.0).astype(float).to_numpy().sum())
    demand = pd.to_numeric(product.get('Demand', pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    return float(total_allocated / demand) if float(demand) > 0 else 0.0


def normalized_l1_change(current: pd.DataFrame, baseline: pd.DataFrame, rounds: list[str]) -> float:
    """Calculate normalized L1 change from the baseline solution without mechanically comparing perturbed inputs with historical labels."""
    delta = float((current.loc[:, rounds].fillna(0.0).astype(float) - baseline.loc[:, rounds].fillna(0.0).astype(float)).abs().to_numpy().sum())
    total = float(baseline.loc[:, rounds].fillna(0.0).astype(float).to_numpy().sum())
    return float(delta / max(1.0, 2.0 * total))


def allocation_metrics(
    allocation: pd.DataFrame,
    month: Any,
    objective: dict[str, Any],
    include_historical_gnae: bool = True,
) -> dict[str, float]:
    """Calculate stability and fill rate consistently; calculate historical-plan GNAE only when requested."""
    from bench.metrics import (
        compute_round_volume_deviation,
        compute_sku_concentration_metric,
        compute_sku_stability_metric,
        compute_total_stability_metric,
    )

    rounds = list(month.rounds)
    metrics = {
        "fill_rate": fill_rate(allocation, month.product_data, rounds),
        "round_volume_deviation": float(compute_round_volume_deviation(allocation, rounds)),
        "adjacent_round_fluctuation": float(compute_total_stability_metric(allocation, rounds)),
        "sku_level_fluctuation": float(compute_sku_stability_metric(allocation, rounds)),
        "sku_concentration": float(compute_sku_concentration_metric(allocation, rounds)),
    }

    # The input perturbation experiment does not use the historical input environment of the human plan, so this option is disabled for that experiment.
    if include_historical_gnae and month.label_data is not None:
        from bench.io import standardize_allocation_matrix
        from bench.metrics import compute_gnae

        gnae = np.nan
        actual = standardize_allocation_matrix(allocation, rounds)
        label = standardize_allocation_matrix(month.label_data, rounds)
        if 'Code' in actual.columns and 'Code' in label.columns:
            merged = actual.merge(label, on='Code', suffixes=("_a", "_b"))
            a = merged[['Code'] + [f"{round_name}_a" for round_name in rounds]].rename(columns={f"{round_name}_a": round_name for round_name in rounds})
            b = merged[['Code'] + [f"{round_name}_b" for round_name in rounds]].rename(columns={f"{round_name}_b": round_name for round_name in rounds})
            gnae = compute_gnae(a, b, rounds)
        else:
            gnae = compute_gnae(actual, label, rounds)
        metrics["gnae_normalized_l1_to_historical_plan"] = float(gnae) if not pd.isna(gnae) else np.nan
    return metrics
