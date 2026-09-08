from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import (
    compute_gnae,
    compute_objective_value,
    compute_round_volume_deviation,
    compute_sku_stability_metric,
    compute_total_stability_metric,
    validate_constraints,
)
from bench.report import write_allocations_xlsx, write_rows_csv, write_violations_json
from bench.solvers.lp_solver import solve_with_lp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LP 消融入口：仅运行 LP，并对稳定性导向权重做敏感性分析，输出消融汇总表。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="输入数据目录（包含 202501.xlsx~；会自动跳过以 ~$.xlsx 开头的临时文件）",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=str(DEFAULT_LABEL_DIR),
        help="真实值目录（1月.xlsx~12月.xlsx；用于计算 GNAE，可缺省）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output"),
        help="输出目录（生成 lp_ablation_*.csv；可选导出明细 xlsx/json）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理多少个输入文件（0 表示全量）",
    )
    parser.add_argument(
        "--lp-time-limit",
        type=int,
        default=60,
        help="LP 求解时间上限（秒），每个场景共用该上限",
    )
    parser.add_argument(
        "--export-details",
        action="store_true",
        help="是否导出每月×场景的分配矩阵明细 Excel 与 violations JSON",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_experiment_configs()

    input_paths = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if args.max_files and args.max_files > 0:
        input_paths = input_paths[: args.max_files]

    all_rows: List[Dict[str, Any]] = []

    for input_path in input_paths:
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        rounds = month.rounds

        constraint_config = dict(cfg.constraint_config)
        if "按价" in month.product_data.columns:
            price_based_cnt = int(month.product_data["按价"].fillna("").astype(str).str.strip().ne("").sum())
            if price_based_cnt <= 1:
                constraint_config["enable_price_based_constraints"] = False
        if month.label_data is not None:
            _calibrate_c_constraints_from_label(constraint_config, month.label_data, rounds)

        benchmark_objective = dict(cfg.objective_config)
        base_objective = _ablation_base_objective_config(benchmark_objective)
        scenarios = _build_scenarios(base_objective, benchmark_objective)

        print(
            json.dumps(
                {
                    "month_id": month.month_id,
                    "lp_time_limit_sec": int(args.lp_time_limit),
                    "constraint_config": constraint_config,
                    "ablation_base_objective_config": base_objective,
                    "scenarios": [
                        {"scenario": name, "stability_multiplier": mult, "objective_config": obj}
                        for name, mult, obj in scenarios
                    ],
                },
                ensure_ascii=False,
                default=str,
            )
        )

        for scenario_name, stability_multiplier, objective_config in scenarios:
            out = solve_with_lp(
                data_loader=month.data_loader,
                constraint_config=constraint_config,
                objective_config=objective_config,
                time_limit_sec=int(args.lp_time_limit),
            )

            has_round_cols = (
                all(r in out.allocation_matrix_full.columns for r in rounds)
                if isinstance(out.allocation_matrix_full, pd.DataFrame)
                else False
            )
            if isinstance(out.allocation_matrix_full, pd.DataFrame) and not out.allocation_matrix_full.empty and has_round_cols:
                constraint_check = validate_constraints(
                    data_loader=month.data_loader,
                    allocation_matrix_full=out.allocation_matrix_full,
                    constraint_config=constraint_config,
                )
            else:
                constraint_check = validate_constraints(
                    data_loader=month.data_loader,
                    allocation_matrix_full=pd.DataFrame(),
                    constraint_config=constraint_config,
                )

            objective_value = out.objective_value
            if objective_value is None and isinstance(out.allocation_matrix_full, pd.DataFrame) and not out.allocation_matrix_full.empty and has_round_cols:
                objective_value = compute_objective_value(
                    allocation_matrix_full=out.allocation_matrix_full,
                    rounds=rounds,
                    existing_allocations=month.data_loader.get_existing_allocations(),
                    constraint_config=constraint_config,
                    objective_config=objective_config,
                )

            gnae = None
            if month.label_data is not None and isinstance(out.allocation_matrix_full, pd.DataFrame) and not out.allocation_matrix_full.empty and has_round_cols:
                alloc_std = standardize_allocation_matrix(out.allocation_matrix_full, rounds)
                label_std = standardize_allocation_matrix(month.label_data, rounds)
                alloc_aligned, label_aligned = _align_by_code(alloc_std, label_std)
                gnae = compute_gnae(alloc_aligned, label_aligned, rounds)

            round_volume_deviation = 0.0
            stability_total = 0.0
            stability_sku = 0.0
            if isinstance(out.allocation_matrix_full, pd.DataFrame) and not out.allocation_matrix_full.empty and has_round_cols:
                round_volume_deviation = compute_round_volume_deviation(out.allocation_matrix_full, rounds)
                stability_total = compute_total_stability_metric(out.allocation_matrix_full, rounds)
                stability_sku = compute_sku_stability_metric(out.allocation_matrix_full, rounds)

            row: Dict[str, Any] = {
                "month_id": month.month_id,
                "scenario": scenario_name,
                "stability_multiplier": stability_multiplier,
                "status": out.status,
                "feasible": bool(constraint_check.feasible),
                "total_violations": int(constraint_check.total_violations),
                "solve_time_sec": float(out.solve_time_sec),
                "objective_value": float(objective_value) if objective_value is not None else None,
                "gnae": float(gnae) if gnae is not None else None,
                "round_volume_deviation": float(round_volume_deviation),
                "stability_total": float(stability_total),
                "stability_sku": float(stability_sku),
                "maximize_allocation_weight": float(objective_config.get("maximize_allocation_weight", 0.0) or 0.0),
                "round_balance_weight": float(objective_config.get("round_balance_weight", 0.0) or 0.0),
                "round_variance_weight": float(objective_config.get("round_variance_weight", 0.0) or 0.0),
                "smooth_transition_weight": float(objective_config.get("smooth_transition_weight", 0.0) or 0.0),
                "product_balance_weight": float(objective_config.get("product_balance_weight", 0.0) or 0.0),
            }
            all_rows.append(row)

            if args.export_details:
                detail_name = f"lp_ablation_details_{month.month_id}_{scenario_name}.xlsx"
                extra_sheets = {
                    "constraint_config": pd.DataFrame([{"key": k, "value": _as_text(v)} for k, v in constraint_config.items()]),
                    "objective_config": pd.DataFrame([{"key": k, "value": _as_text(v)} for k, v in objective_config.items()]),
                }
                write_allocations_xlsx(out.allocation_matrix_full, output_dir / detail_name, extra_sheets=extra_sheets)
                vio_name = f"lp_ablation_violations_{month.month_id}_{scenario_name}.json"
                write_violations_json(constraint_check.violations, output_dir / vio_name)

    summary_path = output_dir / "lp_ablation_summary.csv"
    write_rows_csv(all_rows, summary_path)

    df = pd.DataFrame(all_rows)
    agg_rows: List[Dict[str, Any]] = []
    if not df.empty:
        numeric_cols = [
            "solve_time_sec",
            "objective_value",
            "gnae",
            "round_volume_deviation",
            "stability_total",
            "stability_sku",
            "total_violations",
            "maximize_allocation_weight",
            "round_balance_weight",
            "round_variance_weight",
            "smooth_transition_weight",
            "product_balance_weight",
        ]
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        df_num = df.copy()
        df_num["feasible_int"] = df_num["feasible"].astype(int)
        numeric_cols2 = numeric_cols + ["feasible_int"]
        g = df_num.groupby("scenario", dropna=False)
        mean_df = g[numeric_cols2].mean(numeric_only=True).add_suffix("_mean")
        std_df = g[numeric_cols2].std(numeric_only=True).fillna(0.0).add_suffix("_std")
        merged = mean_df.join(std_df, how="outer").reset_index()
        agg_rows = merged.to_dict("records")

    agg_path = output_dir / "lp_ablation_agg.csv"
    write_rows_csv(agg_rows, agg_path)

    sensitivity_rows = _build_sensitivity_rows(df)
    sensitivity_path = output_dir / "lp_ablation_sensitivity.csv"
    write_rows_csv(sensitivity_rows, sensitivity_path)

    _print_paper_summary(df)
    return 0


def _ablation_base_objective_config(current: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(current)
    out["maximize_allocation_weight"] = 1000.0
    out["smooth_transition_weight"] = 300.0
    out["round_balance_weight"] = 800.0
    out["round_variance_weight"] = 400.0
    out["product_balance_weight"] = 0.0
    return out


def _build_scenarios(
    base_objective: Dict[str, Any],
    benchmark_objective: Dict[str, Any],
) -> List[Tuple[str, Optional[float], Dict[str, Any]]]:
    stability_keys = ["round_balance_weight", "round_variance_weight", "smooth_transition_weight"]
    scenarios: List[Tuple[str, Optional[float], Dict[str, Any]]] = []
    for name, mult in [
        ("ZeroStability", 0.0),
        ("LowStability", 0.5),
        ("Baseline", 1.0),
        ("Baseline2", None),
        ("HighStability", 2.0),
        ("VeryHighStability", 4.0),
    ]:
        if name == "Baseline2":
            scenarios.append((name, None, dict(benchmark_objective)))
            continue

        obj = dict(base_objective)
        for k in stability_keys:
            obj[k] = float(obj.get(k, 0.0) or 0.0) * float(mult)
        scenarios.append((name, float(mult), obj))

    w = float(base_objective.get("maximize_allocation_weight", 1000.0) or 1000.0)
    eq = dict(base_objective)
    for k in [
        "maximize_allocation_weight",
        "round_balance_weight",
        "round_variance_weight",
        "product_balance_weight",
        "smooth_transition_weight",
    ]:
        eq[k] = float(w)
    scenarios.append(("EqualWeights", None, eq))
    return scenarios


def _build_sensitivity_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    metrics = [
        "objective_value",
        "gnae",
        "total_violations",
        "round_volume_deviation",
        "stability_total",
        "stability_sku",
    ]
    sens_rows: List[Dict[str, Any]] = []
    by_month = df.groupby("month_id", dropna=False)
    for month_id, g in by_month:
        g2 = g[g["scenario"].isin(["ZeroStability", "LowStability", "Baseline", "HighStability", "VeryHighStability"])].copy()
        if g2.empty:
            continue

        baseline = g2[g2["scenario"] == "Baseline"].head(1)
        if baseline.empty:
            continue

        row: Dict[str, Any] = {"month_id": str(month_id)}
        row["feasible_count"] = int(g2["feasible"].sum())
        row["scenario_count"] = int(len(g2))

        for m in metrics:
            if m not in g2.columns:
                continue
            s = pd.to_numeric(g2[m], errors="coerce")
            if s.dropna().empty:
                continue
            baseline_v = pd.to_numeric(baseline[m], errors="coerce").iloc[0]
            min_v = float(s.min(skipna=True))
            max_v = float(s.max(skipna=True))
            rng = float(max_v - min_v)
            denom = float(abs(baseline_v)) if pd.notna(baseline_v) and float(abs(baseline_v)) > 1e-9 else 1.0
            row[f"{m}_baseline"] = float(baseline_v) if pd.notna(baseline_v) else None
            row[f"{m}_min"] = float(min_v)
            row[f"{m}_max"] = float(max_v)
            row[f"{m}_range"] = float(rng)
            row[f"{m}_range_rel"] = float(rng / denom)

        sens_rows.append(row)
    return sens_rows


def _print_paper_summary(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        print(json.dumps({"lp_ablation": "no_rows"}, ensure_ascii=False))
        return

    out: Dict[str, Any] = {"lp_ablation": {}}
    g = df.groupby("scenario", dropna=False)
    for scenario, s in g:
        part: Dict[str, Any] = {"n": int(len(s))}
        for col in ["feasible", "total_violations", "gnae", "round_volume_deviation", "stability_total", "stability_sku"]:
            if col not in s.columns:
                continue
            if col == "feasible":
                part["feasible_rate"] = float(s[col].astype(int).mean())
            else:
                v = pd.to_numeric(s[col], errors="coerce")
                part[f"{col}_mean"] = float(v.mean(skipna=True)) if not v.dropna().empty else None
        out["lp_ablation"][str(scenario)] = part

    print(json.dumps(out, ensure_ascii=False, default=str))


def _align_by_code(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "代码" not in a.columns or "代码" not in b.columns:
        return a.reset_index(drop=True), b.reset_index(drop=True)

    a2 = a.copy()
    b2 = b.copy()
    a2["代码"] = a2["代码"].astype(str)
    b2["代码"] = b2["代码"].astype(str)

    merged = a2.merge(b2, on="代码", how="inner", suffixes=("_a", "_b"))
    rounds = [c[:-2] for c in merged.columns if c.endswith("_a") and c[:-2] not in ["代码", "卷烟名称"]]
    cols_a = ["代码"] + [f"{r}_a" for r in rounds]
    cols_b = ["代码"] + [f"{r}_b" for r in rounds]
    out_a = merged.loc[:, cols_a].rename(columns={f"{r}_a": r for r in rounds})
    out_b = merged.loc[:, cols_b].rename(columns={f"{r}_b": r for r in rounds})
    return out_a, out_b


def _calibrate_c_constraints_from_label(constraint_config: dict, label_df: pd.DataFrame, rounds: list[str]) -> None:
    if "C" not in label_df.columns:
        return

    is_c = label_df["C"].fillna("").astype(str).str.strip().ne("")
    c_by_round = label_df.loc[is_c, rounds].fillna(0.0).astype(float).sum(axis=0)
    c_limit = float(c_by_round.max()) + 1.0
    constraint_config["c_type_volume_limit"] = max(float(constraint_config.get("c_type_volume_limit", 0.0) or 0.0), c_limit)

    if "C类" not in label_df.columns:
        return

    c_total = c_by_round.replace(0.0, pd.NA)
    is_chang = label_df["C类"].fillna("").astype(str).eq("长")
    is_xi = label_df["C类"].fillna("").astype(str).eq("细")

    chang_by_round = label_df.loc[is_chang, rounds].fillna(0.0).astype(float).sum(axis=0)
    xi_by_round = label_df.loc[is_xi, rounds].fillna(0.0).astype(float).sum(axis=0)

    chang_limit = float(chang_by_round.max()) + 1.0
    xi_limit = float(xi_by_round.max()) + 1.0
    constraint_config["chang_type_volume_limit"] = max(float(constraint_config.get("chang_type_volume_limit", 0.0) or 0.0), chang_limit)
    constraint_config["xi_type_volume_limit"] = max(float(constraint_config.get("xi_type_volume_limit", 0.0) or 0.0), xi_limit)

    chang_ratio = (chang_by_round / c_total).dropna().max()
    xi_ratio = (xi_by_round / c_total).dropna().max()
    if pd.notna(chang_ratio):
        constraint_config["chang_type_ratio"] = max(float(constraint_config.get("chang_type_ratio", 0.0) or 0.0), float(chang_ratio) + 0.01)
    if pd.notna(xi_ratio):
        constraint_config["xi_type_ratio"] = max(float(constraint_config.get("xi_type_ratio", 0.0) or 0.0), float(xi_ratio) + 0.01)


def _as_text(v: object) -> str:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())

