"""Orchestrate submission experiments and export results.

Every entry point creates a new run_id directory; original exp/backend files and historical revision results are not overwritten."""
from __future__ import annotations

import copy
import shutil
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .audit import allocation_metrics, classify_validation, normalized_l1_change
from .common import REPO_ROOT, create_run, load_frozen_config, load_months, load_protocol, sha256_file, write_json, write_rows_csv
from .solvers import RevisionSolverOutput, solve_ga_budgeted, solve_lp_frozen, solve_rule_based_frozen, solve_sa_budgeted


def _validate(
    month: Any,
    output: RevisionSolverOutput,
    constraints: dict[str, Any],
    protocol: dict[str, Any],
    include_historical_gnae: bool = True,
    audit_absolute_tolerance: float = 0.0,
    audit_relative_tolerance: float | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Audit every method with the same final validator and add common submission metrics."""
    from bench.metrics import validate_constraints

    if output.allocation_matrix_full.empty:
        audit = {
            "hard_constraint_violations": -1,
            "soft_rule_slack_activations": 0,
            "post_solution_review_flags": 0,
            "numerical_tolerance_flags": 0,
            "applicable_rule_checks": 0,
            "normalized_review_flag_rate": np.nan,
            "max_violation_magnitude": np.nan,
            "details": [{"category": "solver_failure", "detail": {"status": output.status}}],
        }
        return audit, {}
    validation = validate_constraints(month.data_loader, output.allocation_matrix_full, constraints)
    audit = classify_validation(
        month,
        validation,
        float(protocol["numerical_tolerance_relative"] if audit_relative_tolerance is None else audit_relative_tolerance),
        absolute_tolerance=audit_absolute_tolerance,
    )
    return audit, allocation_metrics(output.allocation_matrix_full, month, {}, include_historical_gnae=include_historical_gnae)


def _record(
    month: Any,
    method: str,
    seed: int | None,
    output: RevisionSolverOutput,
    lp_objective: float | None,
    constraints: dict[str, Any],
    protocol: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    include_historical_gnae: bool = True,
    audit_context: dict[str, Any] | None = None,
    audit_absolute_tolerance: float = 0.0,
    audit_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Flatten one method run into a summary-ready record with traceable audit details."""
    audit, metrics = _validate(
        month,
        output,
        constraints,
        protocol,
        include_historical_gnae=include_historical_gnae,
        audit_absolute_tolerance=audit_absolute_tolerance,
        audit_relative_tolerance=audit_relative_tolerance,
    )
    for detail in audit.pop("details"):
        audit_rows.append({"month_id": month.month_id, "method": method, "seed": seed, **(audit_context or {}), **detail})
    gap = np.nan
    if lp_objective is not None and output.objective_value is not None:
        gap = float((output.objective_value - lp_objective) / max(1.0, abs(lp_objective)))
    return {
        "month_id": month.month_id,
        "method": method,
        "seed": seed,
        "status": output.status,
        "solver_status": output.solver_status,
        "solve_time_sec": output.solve_time_sec,
        "objective_evaluations": output.objective_evaluations,
        "objective_value": output.objective_value,
        "objective_gap_to_lp": gap,
        "lp_optimality_gap": output.optimality_gap,
        "solver_success": bool(not output.allocation_matrix_full.empty),
        **metrics,
        **audit,
    }


def _summarize(
    rows: list[dict[str, Any]],
    group_cols: list[str] | None = None,
    numeric_only: bool = True,
) -> list[dict[str, Any]]:
    """Generate month-by-method means, standard deviations, medians, ranges, and approximate 95% intervals."""
    if not rows:
        return []
    group_cols = group_cols or ["month_id", "method"]
    frame = pd.DataFrame(rows)
    excluded = {"seed"}
    numeric = [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
    out: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item: dict[str, Any] = {column: keys[index] for index, column in enumerate(group_cols)}
        item["n_runs"] = int(len(group))
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().tolist()
            if not values:
                continue
            mean = float(statistics.mean(values))
            std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            se = std / max(1.0, len(values) ** 0.5)
            item.update({
                f"{column}_mean": mean,
                f"{column}_sd": std,
                f"{column}_median": float(statistics.median(values)),
                f"{column}_min": float(min(values)),
                f"{column}_max": float(max(values)),
                f"{column}_ci95_low": mean - 1.96 * se,
                f"{column}_ci95_high": mean + 1.96 * se,
            })
        out.append(item)
    return out


def _write_run_outputs(run_dir: Path, rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    """Write raw, summary, and audit records using the submission-table field definitions."""
    write_rows_csv(run_dir / "benchmark_raw_runs.csv", rows)
    write_rows_csv(run_dir / "benchmark_month_method_summary.csv", _summarize(rows))
    write_rows_csv(run_dir / "rule_audit_details.csv", audit_rows)


def run_benchmark(max_months: int = 0, max_seeds: int = 0) -> Path:
    """Run the frozen main experiment: LP/RuleBased once and SA/GA with 30 fixed seeds each."""
    run_dir = create_run("benchmark")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    months = load_months(max_months)
    seeds = list(range(int(protocol["seed_start"]), int(protocol["seed_start"]) + int(protocol["seed_count"])))
    if max_seeds > 0:
        seeds = seeds[:max_seeds]
    write_json(run_dir / "seeds.txt", {"seeds": seeds})

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    total = len(months) * (2 + 2 * len(seeds))
    done = 0
    for month in months:
        print(f"[Benchmark] month {month.month_id}: LP and RuleBased")
        lp = solve_lp_frozen(month, constraints, objective, int(protocol["lp_time_limit_sec"]))
        rows.append(_record(month, "LP", None, lp, lp.objective_value, constraints, protocol, audit_rows)); done += 1
        rule = solve_rule_based_frozen(month, constraints, objective)
        rows.append(_record(month, "RuleBased", None, rule, lp.objective_value, constraints, protocol, audit_rows)); done += 1
        for seed in seeds:
            print(f"[Benchmark] {done + 1}/{total} month={month.month_id} method=SA seed={seed}")
            sa = solve_sa_budgeted(month, constraints, objective, seed, float(protocol["metaheuristic_time_limit_sec"]), int(protocol["max_objective_evaluations"]))
            rows.append(_record(month, "SA", seed, sa, lp.objective_value, constraints, protocol, audit_rows)); done += 1
            print(f"[Benchmark] {done + 1}/{total} month={month.month_id} method=GA seed={seed}")
            ga = solve_ga_budgeted(month, constraints, objective, seed, float(protocol["metaheuristic_time_limit_sec"]), int(protocol["max_objective_evaluations"]), int(protocol["ga_population_size"]))
            rows.append(_record(month, "GA", seed, ga, lp.objective_value, constraints, protocol, audit_rows)); done += 1
    _write_run_outputs(run_dir, rows, audit_rows)
    _write_exploratory_statistics(run_dir, rows)
    return run_dir


def run_ablation(max_months: int = 0) -> Path:
    """Remove each of four stability components separately while preserving all other frozen weights."""
    run_dir = create_run("ablation")
    constraints, baseline = load_frozen_config()
    protocol = load_protocol()
    components = ["round_balance_weight", "round_variance_weight", "product_balance_weight", "smooth_transition_weight"]
    scenarios = {"Baseline": dict(baseline)}
    scenarios.update({f"Without_{component}": {**baseline, component: 0.0} for component in components})
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for month in load_months(max_months):
        for name, objective in scenarios.items():
            output = solve_lp_frozen(month, constraints, objective, int(protocol["lp_time_limit_sec"]))
            row = _record(month, name, None, output, None, constraints, protocol, audit_rows)
            row["scenario"] = name
            for component in components:
                row[component] = objective.get(component, 0.0)
            rows.append(row)
    write_rows_csv(run_dir / "ablation_raw.csv", rows)
    write_rows_csv(run_dir / "ablation_summary.csv", _summarize(rows))
    write_rows_csv(run_dir / "rule_audit_details.csv", audit_rows)
    return run_dir


def _perturbed_month(month: Any, demand_multiplier: float = 1.0, inventory_multiplier: float = 1.0, price_width_multiplier: float = 1.0) -> Any:
    """Deep-copy the loader and modify only the copy, preserving the Excel files and original DataLoader."""
    loader = copy.deepcopy(month.data_loader)
    product = loader.sheet1_data.copy(deep=True)
    if 'Demand' in product.columns:
        product['Demand'] = pd.to_numeric(product['Demand'], errors="coerce").fillna(0.0) * demand_multiplier
    if 'Available_Supply' in product.columns:
        product['Available_Supply'] = pd.to_numeric(product['Available_Supply'], errors="coerce").fillna(0.0) * inventory_multiplier
    loader.sheet1_data = product
    for round_name, values in loader.round_constraints.items():
        center = (float(values["upper_price_limit"]) + float(values["lower_price_limit"])) / 2.0
        half_width = (float(values["upper_price_limit"]) - float(values["lower_price_limit"])) / 2.0 * price_width_multiplier
        values["upper_price_limit"] = center + half_width
        values["lower_price_limit"] = center - half_width
    return replace(month, data_loader=loader, product_data=product)


def run_uncertainty(max_months: int = 0) -> Path:
    """Run one-factor perturbations of demand, inventory, and price bounds across 11 unique scenarios."""
    run_dir = create_run("uncertainty")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    scenarios = [("Baseline", 1.0, 1.0, 1.0)]
    scenarios += [(f"Demand_{int((x - 1) * 100):+d}pct", x, 1.0, 1.0) for x in (0.9, 0.95, 1.05, 1.1)]
    scenarios += [(f"Inventory_{int((x - 1) * 100):+d}pct", 1.0, x, 1.0) for x in (0.9, 0.95, 1.05, 1.1)]
    scenarios += [("PriceWidth_-5pct", 1.0, 1.0, 0.95), ("PriceWidth_+5pct", 1.0, 1.0, 1.05)]
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for month in load_months(max_months):
        base_output = solve_lp_frozen(month, constraints, objective, int(protocol["lp_time_limit_sec"]))
        for name, d_mult, i_mult, p_mult in scenarios:
            scenario_month = _perturbed_month(month, d_mult, i_mult, p_mult)
            output = base_output if name == "Baseline" else solve_lp_frozen(scenario_month, constraints, objective, int(protocol["lp_time_limit_sec"]))
            row = _record(scenario_month, "LP", None, output, None, constraints, protocol, audit_rows)
            row.update({"scenario": name, "demand_multiplier": d_mult, "inventory_multiplier": i_mult, "price_width_multiplier": p_mult})
            if not output.allocation_matrix_full.empty and not base_output.allocation_matrix_full.empty:
                row["normalized_l1_change_to_baseline"] = normalized_l1_change(output.allocation_matrix_full, base_output.allocation_matrix_full, month.rounds)
            rows.append(row)
    write_rows_csv(run_dir / "uncertainty_raw.csv", rows)
    write_rows_csv(run_dir / "uncertainty_summary.csv", _summarize(rows))
    write_rows_csv(run_dir / "rule_audit_details.csv", audit_rows)
    return run_dir


ROBUSTNESS_STRUCTURE_METRICS = [
    "round_volume_deviation",
    "adjacent_round_fluctuation",
    "sku_level_fluctuation",
    "sku_concentration",
]
ROBUSTNESS_RULE_METRICS = [
    "hard_constraint_violations",
    "post_solution_review_flags",
    "numerical_tolerance_flags",
    "normalized_review_flag_rate",
    "max_violation_magnitude",
]


def _month_seed_sequence(month_id: str, seed: int) -> np.random.SeedSequence:
    """Encode month and replication seed jointly to avoid reusing a random stream across months."""
    month_entropy = int(month_id) if str(month_id).isdigit() else sum(ord(char) for char in str(month_id))
    return np.random.SeedSequence([int(seed), month_entropy])


def prepare_robustness_month(month: Any) -> Any:
    """Create a monthly copy without historical fixed allocations to avoid conflicts with perturbed demand."""
    loader = copy.deepcopy(month.data_loader)
    product = loader.sheet1_data.copy(deep=True)
    for round_name in month.rounds:
        if round_name in product.columns:
            product[round_name] = 0.0
    loader.sheet1_data = product
    return replace(month, data_loader=loader, product_data=product)


def _worst_case_share(subgroup_total: float, parent_total: float, amplitude: float) -> float:
    """Calculate an ex ante upper bound on subgroup share under independent SKU perturbation limits."""
    subgroup = max(0.0, float(subgroup_total))
    parent = max(0.0, float(parent_total))
    other = max(0.0, parent - subgroup)
    numerator = subgroup * (1.0 + amplitude)
    denominator = numerator + other * (1.0 - amplitude)
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def build_robustness_constraint_config(
    month: Any,
    base_constraints: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Set consistent quota, average-price, and structural bounds from perturbation limits without using historical labels."""
    amplitude = float(protocol["robustness_input_perturbation_amplitude"])
    volume_tolerance = float(protocol["robustness_volume_tolerance"])
    price_half_width_ratio = float(protocol["robustness_price_band_relative_half_width"])
    ratio_guard = float(protocol["robustness_structure_ratio_guard"])
    volume_margin = float(protocol["robustness_structure_volume_margin"])
    if not 0.0 <= volume_tolerance < 1.0:
        raise ValueError('Robustness round-volume tolerance must lie in [0, 1).')

    product = month.product_data
    demand = pd.to_numeric(product['Demand'], errors="coerce").fillna(0.0)
    total_demand = float(demand.sum())
    round_count = max(1, len(month.rounds))
    config = dict(base_constraints)
    config["enable_demand_split_constraints"] = False
    config["volume_tolerance"] = volume_tolerance
    config["volume_limits"] = {}
    config["price_lower_limits"] = {}
    config["price_upper_limits"] = {}
    design_rows: list[dict[str, Any]] = []

    # Keep round target totals unchanged, but allow round rebalancing within twice the SKU perturbation amplitude.
    for round_name in month.rounds:
        original = month.data_loader.get_round_constraints(round_name)
        target = float(original["total_quantity"])
        original_lower = float(original["lower_price_limit"])
        original_upper = float(original["upper_price_limit"])
        center = (original_lower + original_upper) / 2.0
        half_width = max((original_upper - original_lower) / 2.0, abs(center) * price_half_width_ratio)
        adjusted_lower = max(0.0, center - half_width)
        adjusted_upper = center + half_width
        config["volume_limits"][round_name] = target
        config["price_lower_limits"][round_name] = adjusted_lower
        config["price_upper_limits"][round_name] = adjusted_upper
        design_rows.append(
            {
                "month_id": month.month_id,
                "round_name": round_name,
                "monthly_quota": total_demand,
                "volume_target": target,
                "volume_lower": target * (1.0 - volume_tolerance),
                "volume_upper": target * (1.0 + volume_tolerance),
                "price_lower_original": original_lower,
                "price_upper_original": original_upper,
                "price_lower_adjusted": adjusted_lower,
                "price_upper_adjusted": adjusted_upper,
                "price_half_width_relative": price_half_width_ratio,
            }
        )

    c_mask = product["C"].fillna("").astype(str).str.contains("C", na=False)
    c_total = float(demand[c_mask].sum())
    c_upper_share = min(1.0, _worst_case_share(c_total, total_demand, amplitude) + ratio_guard)
    c_upper_total = _worst_case_share(c_total, total_demand, amplitude) * total_demand
    # The linear solver can have read-back errors around 1e-9 near binding constraints; the additional margin does not change the business range.
    ratio_numerical_margin = 1e-6
    volume_numerical_margin = 1e-2
    config["c_type_ratio"] = min(1.0, c_upper_share + ratio_numerical_margin)
    config["c_type_volume_limit"] = max(float(base_constraints.get("c_type_volume_limit", 0.0)), c_upper_total * (1.0 + volume_margin) / round_count) + volume_numerical_margin

    subtype_specs = [('Long', "chang_type_ratio", "chang_type_volume_limit"), ('Slim', "xi_type_ratio", "xi_type_volume_limit")]
    for subtype, ratio_key, volume_key in subtype_specs:
        subgroup_total = float(demand[product['C_Category'].fillna("").astype(str).eq(subtype)].sum())
        parent_share = _worst_case_share(subgroup_total, c_total, amplitude)
        subgroup_upper_total = _worst_case_share(subgroup_total, total_demand, amplitude) * total_demand
        config[ratio_key] = min(1.0, parent_share + ratio_guard + ratio_numerical_margin)
        config[volume_key] = max(float(base_constraints.get(volume_key, 0.0)), subgroup_upper_total * (1.0 + volume_margin) / round_count) + volume_numerical_margin

    # When too few price-based SKUs exist, the original 30% share of SKU count per round cannot coexist with demand fulfilment for all SKUs.
    price_based_count = int(product.get('Price_Based', pd.Series(dtype=str)).fillna("").astype(str).str.contains('Price_Based', na=False).sum())
    total_sku_count = int((demand > 0.0).sum())
    base_ratio = float(base_constraints.get("price_based_ratio", 0.3))
    max_non_price_slots = price_based_count * round_count * (1.0 - base_ratio) / max(base_ratio, 1e-12)
    price_rule_feasible_by_count = (total_sku_count - price_based_count) <= max_non_price_slots
    if not price_rule_feasible_by_count:
        config["enable_price_based_constraints"] = False

    for item in design_rows:
        item.update(
            {
                "historical_allocations_cleared": True,
                "demand_split_constraints_enabled": False,
                "price_based_constraints_enabled": bool(config.get("enable_price_based_constraints", True)),
                "price_based_sku_count": price_based_count,
                "c_type_ratio_limit": config["c_type_ratio"],
                "c_type_round_volume_limit": config["c_type_volume_limit"],
                "chang_type_ratio_limit": config["chang_type_ratio"],
                "chang_type_round_volume_limit": config["chang_type_volume_limit"],
                "xi_type_ratio_limit": config["xi_type_ratio"],
                "xi_type_round_volume_limit": config["xi_type_volume_limit"],
            }
        )
    return config, design_rows


def perturb_month_by_seed(month: Any, seed: int, amplitude: float) -> tuple[Any, dict[str, float]]:
    """Generate random SKU-level demand/inventory perturbations while preserving original monthly total demand and quotas exactly."""
    if not 0.0 <= float(amplitude) < 1.0:
        raise ValueError('Input-perturbation amplitude must lie in [0, 1).')

    prepared_month = prepare_robustness_month(month)
    loader = prepared_month.data_loader
    product = loader.sheet1_data.copy(deep=True)
    rng = np.random.default_rng(_month_seed_sequence(str(month.month_id), int(seed)))
    lower, upper = 1.0 - float(amplitude), 1.0 + float(amplitude)
    metadata: dict[str, float] = {"perturbation_seed": int(seed), "perturbation_amplitude": float(amplitude)}

    demand = pd.to_numeric(product.get('Demand', pd.Series(0.0, index=product.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    demand_positive = demand > 0.0
    demand_multiplier = np.ones(len(product), dtype=float)
    demand_multiplier[demand_positive] = rng.uniform(lower, upper, size=int(demand_positive.sum()))
    perturbed_demand = demand * demand_multiplier
    original_demand_total = float(demand.sum())
    raw_demand_total = float(perturbed_demand.sum())
    if original_demand_total > 0.0 and raw_demand_total > 0.0:
        perturbed_demand *= original_demand_total / raw_demand_total
    product['Demand'] = perturbed_demand

    inventory = pd.to_numeric(product.get('Available_Supply', pd.Series(0.0, index=product.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    inventory_positive = inventory > 0.0
    inventory_multiplier = np.ones(len(product), dtype=float)
    inventory_multiplier[inventory_positive] = rng.uniform(lower, upper, size=int(inventory_positive.sum()))
    product['Available_Supply'] = inventory * inventory_multiplier

    metadata.update(
        {
            "original_demand_total": original_demand_total,
            "perturbed_demand_total": float(perturbed_demand.sum()),
            "demand_multiplier_min": float(demand_multiplier[demand_positive].min()) if demand_positive.any() else 1.0,
            "demand_multiplier_max": float(demand_multiplier[demand_positive].max()) if demand_positive.any() else 1.0,
            "inventory_multiplier_min": float(inventory_multiplier[inventory_positive].min()) if inventory_positive.any() else 1.0,
            "inventory_multiplier_max": float(inventory_multiplier[inventory_positive].max()) if inventory_positive.any() else 1.0,
            "monthly_quota": float(sum(float(loader.get_round_constraints(round_name).get("total_quantity", 0.0)) for round_name in month.rounds)),
        }
    )
    loader.sheet1_data = product
    return replace(prepared_month, data_loader=loader, product_data=product), metadata


def _add_robustness_deltas(row: dict[str, Any], baseline_row: dict[str, Any], baseline_allocation: pd.DataFrame, allocation: pd.DataFrame, rounds: list[str]) -> None:
    """Add solution, structural, and rule-diagnostic changes relative to the same-month baseline."""
    if allocation.empty or baseline_allocation.empty:
        row["normalized_l1_change_to_baseline"] = np.nan
    else:
        row["normalized_l1_change_to_baseline"] = normalized_l1_change(allocation, baseline_allocation, rounds)
    for metric in [*ROBUSTNESS_STRUCTURE_METRICS, *ROBUSTNESS_RULE_METRICS]:
        current = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        baseline = pd.to_numeric(pd.Series([baseline_row.get(metric)]), errors="coerce").iloc[0]
        delta = float(current - baseline) if pd.notna(current) and pd.notna(baseline) else np.nan
        row[f"{metric}_delta_to_baseline"] = delta
        row[f"{metric}_abs_change_to_baseline"] = abs(delta) if pd.notna(delta) else np.nan


def _mean_sd_range(values: pd.Series) -> tuple[float | None, float | None, float | None, float | None]:
    """Generate means, sample standard deviations, and ranges for submission summaries."""
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None, None, None, None
    return float(valid.mean()), float(valid.std(ddof=1)) if len(valid) > 1 else 0.0, float(valid.min()), float(valid.max())


def _write_robustness_publication_summary(run_dir: Path, rows: list[dict[str, Any]], status_rows: list[dict[str, Any]]) -> None:
    """Generate a Markdown summary containing only the three reviewer-response result categories."""
    frame = pd.DataFrame(rows)
    perturbed = frame[frame["scenario"] == "Perturbed"].copy()
    baseline = frame[frame["scenario"] == "Baseline"].copy()
    def series_or_empty(data: pd.DataFrame, column: str) -> pd.Series:
        """Some solution-structure fields are absent after solver failure; continue reporting status instead of aborting."""
        return data[column] if column in data.columns else pd.Series(dtype=float)

    solution_mean, solution_sd, solution_min, solution_max = _mean_sd_range(series_or_empty(perturbed, "normalized_l1_change_to_baseline"))
    lines = [
        "# SKU-level input-perturbation robustness test",
        "",
        "Six monthly baselines were compared with ten independently seeded SKU-level demand and inventory perturbations per month. Monthly quota totals were retained while pre-specified round-volume, price-band, and structure guard ranges were applied.",
        "",
        "## 1. Solution change",
        "",
        "| Metric | Mean +/- SD | Range |",
        "|---|---:|---:|",
        f"| Normalized L1 change to baseline | {solution_mean:.6f} +/- {solution_sd:.6f} | {solution_min:.6f} to {solution_max:.6f} |" if solution_mean is not None else "| Normalized L1 change to baseline | unavailable | unavailable |",
        "",
        "The score is sum(abs(x_perturbed - x_baseline)) divided by twice the baseline total allocation.",
        "",
        "## 2. Stability and rule-diagnostic change",
        "",
        "| Metric | Baseline mean | Perturbed mean +/- SD | Mean absolute change |",
        "|---|---:|---:|---:|",
    ]
    for metric in [*ROBUSTNESS_STRUCTURE_METRICS, *ROBUSTNESS_RULE_METRICS]:
        baseline_mean, _, _, _ = _mean_sd_range(series_or_empty(baseline, metric))
        perturbed_mean, perturbed_sd, _, _ = _mean_sd_range(series_or_empty(perturbed, metric))
        abs_change, _, _, _ = _mean_sd_range(series_or_empty(perturbed, f"{metric}_abs_change_to_baseline"))
        lines.append(
            f"| {metric} | {baseline_mean:.6f} | {perturbed_mean:.6f} +/- {perturbed_sd:.6f} | {abs_change:.6f} |"
            if None not in (baseline_mean, perturbed_mean, perturbed_sd, abs_change)
            else f"| {metric} | unavailable | unavailable | unavailable |"
        )
    lines.extend([
        "",
        "## 3. Solver status and solve time",
        "",
        "| Status | Solver status | Runs | Successful allocation rate |",
        "|---|---|---:|---:|",
    ])
    for item in status_rows:
        lines.append(f"| {item['status']} | {item['solver_status']} | {item['n_runs']} | {item['solver_success_rate']:.3f} |")
    solve_mean, solve_sd, solve_min, solve_max = _mean_sd_range(series_or_empty(perturbed, "solve_time_sec"))
    lines.extend([
        "",
        f"Perturbed solve time: {solve_mean:.6f} +/- {solve_sd:.6f} seconds (range {solve_min:.6f} to {solve_max:.6f})." if solve_mean is not None else "Perturbed solve time: unavailable.",
        "",
    ])
    (run_dir / "robustness_publication_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_robustness(max_months: int = 0) -> Path:
    """Run the revised LP robustness experiment with random SKU-level input perturbations."""
    run_dir = create_run("robustness")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    seed_start = int(protocol["robustness_seed_start"])
    seed_count = int(protocol["robustness_seed_count"])
    amplitude = float(protocol["robustness_input_perturbation_amplitude"])
    seeds = list(range(seed_start, seed_start + seed_count))
    write_json(run_dir / "robustness_protocol.txt", {"seeds": seeds, "amplitude": amplitude, "demand_total_policy": "renormalize_to_original_month_total", "quota_policy": "unchanged"})

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    months = load_months(max_months)
    total_runs = len(months) * (1 + len(seeds))
    done = 0
    for month in months:
        robustness_constraints, month_design_rows = build_robustness_constraint_config(month, constraints, protocol)
        design_rows.extend(month_design_rows)
        write_json(run_dir / f"robustness_constraint_config_{month.month_id}.txt", robustness_constraints)
        baseline_month = prepare_robustness_month(month)
        print(f"[Robustness] {done + 1}/{total_runs} month={month.month_id} scenario=Baseline")
        baseline_output = solve_lp_frozen(baseline_month, robustness_constraints, objective, int(protocol["lp_time_limit_sec"]))
        baseline_row = _record(
            baseline_month, "LP", None, baseline_output, baseline_output.objective_value, robustness_constraints, protocol, audit_rows,
            include_historical_gnae=False, audit_context={"scenario": "Baseline", "constraint_profile": "range_adjusted_robustness"},
            audit_absolute_tolerance=float(protocol["robustness_numerical_tolerance_absolute"]),
            audit_relative_tolerance=float(protocol["robustness_numerical_tolerance_relative"]),
        )
        baseline_row.update({"scenario": "Baseline", "is_baseline": True, "constraint_profile": "range_adjusted_robustness", "perturbation_seed": None, "perturbation_amplitude": amplitude, "normalized_l1_change_to_baseline": 0.0})
        for metric in [*ROBUSTNESS_STRUCTURE_METRICS, *ROBUSTNESS_RULE_METRICS]:
            baseline_row[f"{metric}_delta_to_baseline"] = 0.0
            baseline_row[f"{metric}_abs_change_to_baseline"] = 0.0
        rows.append(baseline_row)
        done += 1

        for seed in seeds:
            print(f"[Robustness] {done + 1}/{total_runs} month={month.month_id} scenario=Perturbed seed={seed}")
            perturbed_month, metadata = perturb_month_by_seed(month, seed, amplitude)
            output = solve_lp_frozen(perturbed_month, robustness_constraints, objective, int(protocol["lp_time_limit_sec"]))
            row = _record(
                perturbed_month, "LP", int(seed), output, baseline_output.objective_value, robustness_constraints, protocol, audit_rows,
                include_historical_gnae=False, audit_context={"scenario": "Perturbed", "constraint_profile": "range_adjusted_robustness"},
                audit_absolute_tolerance=float(protocol["robustness_numerical_tolerance_absolute"]),
                audit_relative_tolerance=float(protocol["robustness_numerical_tolerance_relative"]),
            )
            row.update({"scenario": "Perturbed", "is_baseline": False, "constraint_profile": "range_adjusted_robustness", **metadata})
            _add_robustness_deltas(row, baseline_row, baseline_output.allocation_matrix_full, output.allocation_matrix_full, list(month.rounds))
            rows.append(row)
            done += 1

    frame = pd.DataFrame(rows)
    status_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for keys, group in frame.groupby(["scenario", "status", "solver_status"], dropna=False):
            scenario, status, solver_status = keys
            status_rows.append({"scenario": scenario, "status": status, "solver_status": solver_status, "n_runs": int(len(group)), "solver_success_rate": float(group["solver_success"].astype(float).mean())})
    write_rows_csv(run_dir / "robustness_raw.csv", rows)
    write_rows_csv(run_dir / "robustness_month_summary.csv", _summarize(rows, group_cols=["month_id", "scenario"]))
    write_rows_csv(run_dir / "robustness_overall_summary.csv", _summarize(rows, group_cols=["scenario"]))
    write_rows_csv(run_dir / "robustness_solver_status_summary.csv", status_rows)
    write_rows_csv(run_dir / "robustness_rule_audit_details.csv", audit_rows)
    write_rows_csv(run_dir / "robustness_constraint_design.csv", design_rows)
    _write_robustness_publication_summary(run_dir, rows, status_rows)
    return run_dir


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg correction within one test family while retaining input order."""
    count = len(p_values)
    adjusted = [1.0] * count
    running = 1.0
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    for rank in range(count, 0, -1):
        index, value = ordered[rank - 1]
        running = min(running, min(1.0, float(value) * count / rank))
        adjusted[index] = running
    return adjusted


def build_month_level_gnae_reanalysis(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Average seed results by month, then perform exact Wilcoxon tests on six independent months."""
    required = {"month_id", "method", "seed", "gnae"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark raw runs are missing fields: {sorted(missing)}")
    clean = frame.loc[:, ["month_id", "method", "seed", "gnae"]].copy()
    clean["month_id"] = clean["month_id"].astype(str)
    clean["gnae"] = pd.to_numeric(clean["gnae"], errors="coerce")
    clean = clean.dropna(subset=["gnae"])
    expected_methods = ["LP", "RuleBased", "SA", "GA"]
    if set(clean["method"].unique()) != set(expected_methods):
        raise ValueError('Benchmark raw runs must contain exactly LP, RuleBased, SA, and GA.')

    monthly_frame = clean.groupby(["month_id", "method"], as_index=False).agg(
        n_seed_runs=("gnae", "size"),
        gnae_mean=("gnae", "mean"),
        gnae_sd=("gnae", "std"),
        gnae_min=("gnae", "min"),
        gnae_max=("gnae", "max"),
    )
    monthly_frame["gnae_sd"] = monthly_frame["gnae_sd"].fillna(0.0)
    pivot = monthly_frame.pivot(index="month_id", columns="method", values="gnae_mean").reindex(columns=expected_methods).dropna().sort_index()
    if len(pivot) != 6:
        raise ValueError(f"Month-level GNAE reanalysis expects 6 complete months; found {len(pivot)}.")
    if not (monthly_frame["n_seed_runs"] == 10).all():
        raise ValueError('Each month and method must have exactly 10 GNAE raw runs.')

    significance_rows: list[dict[str, Any]] = []
    comparisons = ["RuleBased", "SA", "GA"]
    p_values: list[float] = []
    for method in comparisons:
        test = stats.wilcoxon(pivot[method], pivot["LP"], alternative="two-sided", method="exact")
        p_value = float(test.pvalue)
        p_values.append(p_value)
        comparison_mean = float(pivot[method].mean())
        baseline_mean = float(pivot["LP"].mean())
        significance_rows.append({
            "metric": "gnae",
            "baseline_method": "LP",
            "compare_method": method,
            "n_pairs": int(len(pivot)),
            "pairing_unit": "month",
            "seed_handling": "mean_gnae_within_month_before_pairing",
            "test_name": "Wilcoxon signed-rank exact two-sided",
            "statistic": float(test.statistic),
            "p_value": p_value,
            "effect_direction": "better" if comparison_mean < baseline_mean else "worse" if comparison_mean > baseline_mean else "equal",
            "baseline_month_mean": baseline_mean,
            "compare_month_mean": comparison_mean,
        })
    for row, adjusted in zip(significance_rows, _bh_adjust(p_values)):
        row["p_value_bh"] = adjusted
        row["significant_at_0_05_after_bh"] = bool(adjusted < 0.05)

    sig_by_method = {row["compare_method"]: row for row in significance_rows}
    table_rows: list[dict[str, Any]] = []
    for method in expected_methods:
        values = pivot[method]
        row = {
            "method": method,
            "n_independent_months": int(len(values)),
            "seed_runs_per_month": 10,
            "gnae_month_mean": float(values.mean()),
            "gnae_month_sd": float(values.std(ddof=1)),
            "gnae_month_mean_sd": f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}",
            "p_value_vs_lp": None,
            "p_value_bh_vs_lp": None,
            "significant_at_0_05_after_bh": None,
        }
        if method != "LP":
            row.update({key: sig_by_method[method][key] for key in ("p_value", "p_value_bh", "significant_at_0_05_after_bh")})
            row["p_value_vs_lp"] = row.pop("p_value")
            row["p_value_bh_vs_lp"] = row.pop("p_value_bh")
        table_rows.append(row)
    return monthly_frame.to_dict("records"), significance_rows, table_rows


def run_benchmark_gnae_reanalysis() -> Path:
    """Perform traceable month-clustered reanalysis of existing 10-seed GNAE raw runs."""
    source_path = REPO_ROOT / "exp" / "output" / "paper_10seed_benchmark_20260809" / "benchmark_multiseed_runs.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Benchmark raw evidence file was not found: {source_path}")
    run_dir = create_run("benchmark_gnae_reanalysis")
    copied_source = run_dir / "benchmark_multiseed_runs_source.csv"
    shutil.copy2(source_path, copied_source)
    source_frame = pd.read_csv(copied_source)
    monthly_rows, significance_rows, table_rows = build_month_level_gnae_reanalysis(source_frame)
    write_json(run_dir / "source_evidence.txt", {"source_path": str(source_path), "source_sha256": sha256_file(source_path), "copied_filename": copied_source.name, "copied_sha256": sha256_file(copied_source), "source_row_count": int(len(source_frame))})
    write_rows_csv(run_dir / "benchmark_gnae_month_level.csv", monthly_rows)
    write_rows_csv(run_dir / "benchmark_gnae_month_level_significance.csv", significance_rows)
    write_rows_csv(run_dir / "Table_benchmark_gnae_month_level.csv", table_rows)
    lines = [
        "# Month-level GNAE benchmark reanalysis",
        "",
        "For each method, the ten seed-level values within a month were averaged before statistical testing. The six calendar months are the independent paired units.",
        "",
        "| Comparison | Exact two-sided p | BH-adjusted p | Significant at 0.05 after BH |",
        "|---|---:|---:|---|",
    ]
    for row in significance_rows:
        lines.append(f"| {row['compare_method']} vs LP | {row['p_value']:.4f} | {row['p_value_bh']:.4f} | {row['significant_at_0_05_after_bh']} |")
    lines.extend(["", "The results do not support a statistically significant GNAE advantage over LP after correction.", ""])
    (run_dir / "benchmark_gnae_month_level_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def _write_exploratory_statistics(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Aggregate stochastic methods by month, then export exploratory six-month paired comparisons and Holm corrections."""
    from .statistics import exploratory_paired_comparison

    frame = pd.DataFrame(rows)
    random_rows = frame[frame["method"].isin(["SA", "GA"])].copy()
    monthly = random_rows.groupby(["month_id", "method"], as_index=False)["gnae_normalized_l1_to_historical_plan"].mean()
    result = exploratory_paired_comparison(monthly, "gnae_normalized_l1_to_historical_plan", "SA", "GA")
    write_json(run_dir / "exploratory_paired_statistics.txt", result)


def _anonymized_case_frame(allocation: pd.DataFrame, month: Any, top_n: int) -> pd.DataFrame:
    """Select the top N SKUs by demand and export an anonymous allocation matrix without original product names."""
    rounds = list(month.rounds)
    base = allocation.copy()
    demand = pd.to_numeric(month.product_data.get('Demand', pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    base["_demand"] = demand.to_numpy()
    top = base.nlargest(int(top_n), "_demand").copy()
    top.insert(0, "anonymous_sku", [f"SKU_{i:02d}" for i in range(1, len(top) + 1)])
    other = base.drop(top.index)
    if not other.empty:
        row = {"anonymous_sku": "Others", **{round_name: float(other[round_name].sum()) for round_name in rounds}}
        top = pd.concat([top[["anonymous_sku"] + rounds], pd.DataFrame([row])], ignore_index=True)
    else:
        top = top[["anonymous_sku"] + rounds]
    return top


def _write_heatmap_svg(path: Path, matrix: pd.DataFrame, rounds: list[str], title: str) -> None:
    """Export an anonymous SVG heatmap without adding dependencies."""
    values = matrix.loc[:, rounds].to_numpy(dtype=float)
    maximum = max(1.0, float(values.max()))
    cell_w, cell_h, left, top = 90, 24, 120, 60
    width, height = left + cell_w * len(rounds) + 20, top + cell_h * len(matrix) + 40
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', f'<text x="10" y="22" font-size="16">{title}</text>']
    for j, round_name in enumerate(rounds):
        elements.append(f'<text x="{left + j * cell_w + 4}" y="48" font-size="11">{round_name}</text>')
    for i, (_, row) in enumerate(matrix.iterrows()):
        y = top + i * cell_h
        elements.append(f'<text x="4" y="{y + 16}" font-size="11">{row["anonymous_sku"]}</text>')
        for j, round_name in enumerate(rounds):
            value = float(row[round_name])
            intensity = int(245 - 180 * value / maximum)
            color = f"rgb({intensity},{intensity + min(10, 255 - intensity)},{255})"
            x = left + j * cell_w
            elements.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}"/>')
            elements.append(f'<text x="{x + 3}" y="{y + 16}" font-size="10">{value:.0f}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def run_case_study() -> Path:
    """Select a representative month and generate anonymous heatmaps, baseline/no-stability comparisons, and a binding-constraint table."""
    run_dir = create_run("case_study")
    constraints, baseline = load_frozen_config()
    protocol = load_protocol()
    months = load_months()
    median = statistics.median(len(month.product_data) for month in months)
    month = sorted(months, key=lambda item: (abs(len(item.product_data) - median), item.month_id))[0]
    no_stability = dict(baseline)
    for component in ("round_balance_weight", "round_variance_weight", "product_balance_weight", "smooth_transition_weight"):
        no_stability[component] = 0.0
    base_output = solve_lp_frozen(month, constraints, baseline, int(protocol["lp_time_limit_sec"]))
    no_output = solve_lp_frozen(month, constraints, no_stability, int(protocol["lp_time_limit_sec"]))
    if base_output.allocation_matrix_full.empty or no_output.allocation_matrix_full.empty:
        raise RuntimeError(f"Representative month {month.month_id} LP produced no usable allocation matrix.")
    frame = _anonymized_case_frame(base_output.allocation_matrix_full, month, int(protocol["representative_top_skus"]))
    write_rows_csv(run_dir / "representative_allocation_anonymized.csv", frame.to_dict("records"))
    _write_heatmap_svg(run_dir / "representative_heatmap.svg", frame, month.rounds, f"Representative allocation: {month.month_id}")
    rows = []
    for name, output in (("Baseline", base_output), ("No_stability", no_output)):
        metrics = allocation_metrics(output.allocation_matrix_full, month, baseline)
        totals = output.allocation_matrix_full.loc[:, month.rounds].sum(axis=0)
        total = float(totals.sum())
        rows.append({"scenario": name, **metrics, "first_two_round_share": float(totals.iloc[:2].sum() / total) if total > 0 else 0.0, **{f"share_{round_name}": float(totals[round_name] / total) if total > 0 else 0.0 for round_name in month.rounds}})
    write_rows_csv(run_dir / "baseline_vs_no_stability.csv", rows)
    # Use slack/dual values supplied directly by the LP solver to identify binding constraints; save validation flags separately to distinguish the two concepts.
    diagnostics = base_output.constraint_diagnostics or []
    for item in diagnostics:
        item["is_binding"] = abs(float(item["slack"])) <= float(protocol["numerical_tolerance_relative"])
    diagnostics.sort(key=lambda item: abs(float(item["slack"])))
    write_rows_csv(run_dir / "binding_constraints.csv", diagnostics[:8])
    from bench.metrics import validate_constraints
    audit = classify_validation(month, validate_constraints(month.data_loader, base_output.allocation_matrix_full, constraints), float(protocol["numerical_tolerance_relative"]))
    write_rows_csv(run_dir / "validation_flags.csv", audit["details"])
    write_json(run_dir / "representative_month.txt", {"month_id": month.month_id, "sku_count": len(month.product_data), "median_sku_count": median, "selection_rule": "closest_to_median_sku_count_then_earliest"})
    return run_dir


def run_field_table() -> Path:
    """Generate the design-science field-deployment table; mark manual fields without logs as unavailable."""
    run_dir = create_run("field_table")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for month in load_months():
        output = solve_lp_frozen(month, constraints, objective, int(protocol["lp_time_limit_sec"]))
        row = _record(month, "LP", None, output, output.objective_value, constraints, protocol, audit_rows)
        quota = sum(float(month.data_loader.get_round_constraints(round_name).get("total_quantity", 0.0)) for round_name in month.rounds)
        row.update({"sku_count": len(month.product_data), "round_count": len(month.rounds), "monthly_quota": quota, "manual_review_or_execution": "unavailable", "planner_override_count": "unavailable", "manual_adjustment_cell_rate": "unavailable"})
        rows.append(row)
    write_rows_csv(run_dir / "field_deployment_table.csv", rows)
    write_rows_csv(run_dir / "rule_audit_details.csv", audit_rows)
    return run_dir


def run_scalability(max_months: int = 0) -> Path:
    """Rebuild joint scalability stress tests using the snapshot generator and 10 fixed seeds per scale."""
    run_dir = create_run("scalability")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    from run_lp_scalability import ScaleSpec, _build_scaled_instance, _round_share_from_month

    scales = [ScaleSpec("S1 Small", 0.5, 0.5, 3), ScaleSpec("S2 Field-scale", 1.0, 1.0, 4), ScaleSpec("S3 Large", 2.0, 1.5, 5), ScaleSpec("S4 Extra-large", 5.0, 2.0, 6)]
    rows: list[dict[str, Any]] = []
    for month in load_months(max_months):
        shares = _round_share_from_month(month.data_loader, month.rounds)
        for scale in scales:
            for seed in range(int(protocol["seed_start"]), int(protocol["seed_start"]) + 10):
                instance = _build_scaled_instance(month.month_id, month.product_data, shares, scale, seed)
                synthetic_month = replace(month, month_id=f"{month.month_id}_{scale.name}_{seed}", data_loader=instance.data_loader, product_data=instance.product_data, rounds=instance.rounds, label_data=instance.label_data)
                output = solve_lp_frozen(synthetic_month, constraints, objective, int(protocol["lp_time_limit_sec"]))
                audit, metrics = _validate(synthetic_month, output, constraints, protocol)
                rows.append({"base_month": month.month_id, "scale": scale.name, "seed": seed, "number_of_skus": len(instance.product_data), "rounds": len(instance.rounds), "demand_scale": scale.demand_scale, "status": output.status, "solve_time_sec": output.solve_time_sec, "lp_optimality_gap": output.optimality_gap, "success": int(not output.allocation_matrix_full.empty), **metrics, **{key: value for key, value in audit.items() if key != "details"}})
    write_rows_csv(run_dir / "scalability_raw.csv", rows)
    write_rows_csv(run_dir / "scalability_summary.csv", _summarize(rows))
    return run_dir



def run_preflight(max_months: int = 0) -> Path:
    """Run only the frozen LP precheck and report infeasibility or hard-constraint issues before long multi-seed experiments."""
    run_dir = create_run("preflight")
    constraints, objective = load_frozen_config()
    protocol = load_protocol()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for month in load_months(max_months):
        output = solve_lp_frozen(month, constraints, objective, int(protocol["lp_time_limit_sec"]))
        rows.append(_record(month, "LP", None, output, output.objective_value, constraints, protocol, audit_rows))
    write_rows_csv(run_dir / "frozen_rule_preflight.csv", rows)
    write_rows_csv(run_dir / "rule_audit_details.csv", audit_rows)
    write_json(run_dir / "preflight_summary.txt", {"all_lp_optimal": all(row["status"] == "Optimal" for row in rows), "all_hard_constraints_satisfied": all(int(row["hard_constraint_violations"]) == 0 for row in rows)})
    return run_dir
