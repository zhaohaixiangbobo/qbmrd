from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, compute_objective_value, validate_constraints
from bench.report import write_rows_csv
from bench.solvers.ga import solve_with_ga
from bench.solvers.lp_solver import solve_with_lp
from bench.solvers.rule_based import solve_with_rule_based
from bench.solvers.sa import solve_with_simulated_annealing


def main() -> int:
    """
    运行多随机种子 benchmark，并输出 run-level、聚合统计与显著性分析结果。
    """
    parser = argparse.ArgumentParser(
        description="多种子对比实验入口：批量运行 LP / RuleBased / SA / GA 的多随机种子实验，并输出均值、方差、95%CI 与显著性结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="输入数据目录（包含 202501.xlsx~202506.xlsx；会自动跳过以 ~$.xlsx 开头的临时文件）",
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
        default=str(Path(__file__).resolve().parent / "output" / "benchmark_multiseed"),
        help="输出目录（生成 benchmark_multiseed_*.csv）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理多少个输入文件（0 表示全量）",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="每个方法的随机种子数量",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="起始随机种子（实际会生成连续整数种子序列）",
    )
    parser.add_argument(
        "--lp-time-limit",
        type=int,
        default=60,
        help="LP 求解时间上限（秒）",
    )
    parser.add_argument(
        "--sa-time",
        type=float,
        default=300.0,
        help="SA 最大运行时长（秒）",
    )
    parser.add_argument(
        "--sa-iters",
        type=int,
        default=50000,
        help="SA 最大迭代次数",
    )
    parser.add_argument(
        "--ga-time",
        type=float,
        default=600.0,
        help="GA 最大运行时长（秒）。若传 0，则按数据规模自适应且下限为 60 秒。",
    )
    parser.add_argument(
        "--ga-pop",
        type=int,
        default=0,
        help="GA 种群大小（0 表示按数据规模自适应）",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_experiment_configs()
    seeds = [int(args.base_seed) + i for i in range(max(1, int(args.n_seeds)))]

    input_paths = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if args.max_files and args.max_files > 0:
        input_paths = input_paths[: args.max_files]

    run_rows: List[Dict[str, Any]] = []
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

        print(
            json.dumps(
                {
                    "month_id": month.month_id,
                    "n_seeds": len(seeds),
                    "lp_time_limit_sec": int(args.lp_time_limit),
                    "sa_time_sec": float(args.sa_time),
                    "ga_time_sec": float(args.ga_time),
                    "constraint_config": constraint_config,
                    "objective_config": cfg.objective_config,
                },
                ensure_ascii=False,
                default=str,
            )
        )

        auto_ga_time, auto_ga_pop = _auto_ga_params(n_products=len(month.product_data), n_rounds=len(rounds))
        for seed in seeds:
            outputs = {
                "LP": solve_with_lp(
                    data_loader=month.data_loader,
                    constraint_config=constraint_config,
                    objective_config=cfg.objective_config,
                    time_limit_sec=int(args.lp_time_limit),
                ),
                "RuleBased": solve_with_rule_based(
                    data_loader=month.data_loader,
                    constraint_config=constraint_config,
                ),
                "SA": solve_with_simulated_annealing(
                    data_loader=month.data_loader,
                    constraint_config=constraint_config,
                    objective_config=cfg.objective_config,
                    max_time_sec=float(args.sa_time),
                    max_iters=int(args.sa_iters),
                    seed=int(seed),
                ),
                "GA": solve_with_ga(
                    data_loader=month.data_loader,
                    constraint_config=constraint_config,
                    objective_config=cfg.objective_config,
                    max_time_sec=float(args.ga_time) if float(args.ga_time) > 0 else float(auto_ga_time),
                    pop_size=int(args.ga_pop) if int(args.ga_pop) > 0 else int(auto_ga_pop),
                    seed=int(seed),
                ),
            }

            for method, out in outputs.items():
                row = _evaluate_method_output(
                    month_id=month.month_id,
                    method=method,
                    seed=int(seed),
                    out=out,
                    rounds=rounds,
                    month=month,
                    constraint_config=constraint_config,
                    objective_config=cfg.objective_config,
                )
                run_rows.append(row)

    write_rows_csv(run_rows, output_dir / "benchmark_multiseed_runs.csv")
    summary_rows = _build_summary_rows(pd.DataFrame(run_rows), group_cols=["month_id", "method"])
    overall_rows = _build_summary_rows(pd.DataFrame(run_rows), group_cols=["method"])
    significance_rows = _build_significance_rows(pd.DataFrame(run_rows))
    write_rows_csv(summary_rows, output_dir / "benchmark_multiseed_summary.csv")
    write_rows_csv(overall_rows, output_dir / "benchmark_multiseed_overall.csv")
    write_rows_csv(significance_rows, output_dir / "benchmark_multiseed_significance.csv")

    print(
        json.dumps(
            {
                "输出文件说明": {
                    "benchmark_multiseed_runs.csv": "每月×方法×seed 的 run-level 结果",
                    "benchmark_multiseed_summary.csv": "每月×方法的 mean/std/95%CI/feasible_rate 汇总",
                    "benchmark_multiseed_overall.csv": "跨全部月份的 method 级整体统计",
                    "benchmark_multiseed_significance.csv": "默认相对 LP 的 Wilcoxon 显著性结果（含 BH 校正）",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _evaluate_method_output(
    month_id: str,
    method: str,
    seed: int,
    out: object,
    rounds: List[str],
    month: object,
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将单次方法输出整理成 run-level 结果行。
    """
    has_round_cols = all(r in out.allocation_matrix_full.columns for r in rounds) if isinstance(out.allocation_matrix_full, pd.DataFrame) else False
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

    return {
        "month_id": str(month_id),
        "method": str(method),
        "seed": int(seed),
        "status": str(out.status),
        "feasible": bool(constraint_check.feasible),
        "total_violations": int(constraint_check.total_violations),
        "solve_time_sec": float(out.solve_time_sec),
        "objective_value": float(objective_value) if objective_value is not None else None,
        "gnae": float(gnae) if gnae is not None else None,
    }


def _build_summary_rows(df: pd.DataFrame, group_cols: List[str]) -> List[Dict[str, Any]]:
    """
    按给定分组列生成 mean/std/95%CI/feasible_rate 汇总。
    """
    if df.empty:
        return []

    out_rows: List[Dict[str, Any]] = []
    metric_specs = {
        "gnae": "gnae",
        "total_violations": "violations",
        "solve_time_sec": "solve_time",
        "objective_value": "objective",
    }

    grouped = df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: keys[i] for i, col in enumerate(group_cols)}
        row["n_runs"] = int(len(g))
        row["feasible_rate"] = float(g["feasible"].astype(int).mean()) if "feasible" in g.columns else None
        for col, prefix in metric_specs.items():
            stats_row = _series_stats(pd.to_numeric(g[col], errors="coerce"))
            row[f"{prefix}_mean"] = stats_row["mean"]
            row[f"{prefix}_std"] = stats_row["std"]
            row[f"{prefix}_ci95_low"] = stats_row["ci95_low"]
            row[f"{prefix}_ci95_high"] = stats_row["ci95_high"]
        out_rows.append(row)
    return out_rows


def _series_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """
    计算单个指标序列的 mean/std/95%CI。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}

    values = s.astype(float).tolist()
    n = len(values)
    mean = float(s.mean())
    if n <= 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}

    std = float(s.std(ddof=1))
    if std <= 0:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}

    sem = std / math.sqrt(n)
    try:
        ci_low, ci_high = stats.t.interval(confidence=0.95, df=n - 1, loc=mean, scale=sem)
    except TypeError:
        ci_low, ci_high = stats.t.interval(alpha=0.95, df=n - 1, loc=mean, scale=sem)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


def _build_significance_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    构造默认相对 LP 的显著性比较结果。
    """
    if df.empty:
        return []

    metrics = ["gnae", "total_violations", "solve_time_sec", "objective_value"]
    compare_methods = ["RuleBased", "SA", "GA"]
    raw_rows: List[Dict[str, Any]] = []

    for metric in metrics:
        base_df = df[df["method"] == "LP"].loc[:, ["month_id", "seed", metric]].rename(columns={metric: "baseline_value"})
        for compare_method in compare_methods:
            cmp_df = df[df["method"] == compare_method].loc[:, ["month_id", "seed", metric]].rename(columns={metric: "compare_value"})
            merged = base_df.merge(cmp_df, on=["month_id", "seed"], how="inner")
            merged["baseline_value"] = pd.to_numeric(merged["baseline_value"], errors="coerce")
            merged["compare_value"] = pd.to_numeric(merged["compare_value"], errors="coerce")
            merged = merged.dropna(subset=["baseline_value", "compare_value"])

            row = {
                "metric": metric,
                "baseline_method": "LP",
                "compare_method": compare_method,
                "n_pairs": int(len(merged)),
                "test_name": "Wilcoxon",
                "statistic": None,
                "p_value": None,
                "p_value_bh": None,
                "effect_direction": None,
                "baseline_mean": float(merged["baseline_value"].mean()) if not merged.empty else None,
                "compare_mean": float(merged["compare_value"].mean()) if not merged.empty else None,
                "significant_at_0_05": None,
                "note": "",
            }

            if merged.empty:
                row["note"] = "insufficient pairs"
                raw_rows.append(row)
                continue

            diffs = merged["compare_value"] - merged["baseline_value"]
            if bool((diffs.abs() <= 1e-12).all()):
                row["statistic"] = 0.0
                row["p_value"] = 1.0
                row["effect_direction"] = "equal"
                row["significant_at_0_05"] = False
                row["note"] = "all paired differences are zero"
                raw_rows.append(row)
                continue

            try:
                test = stats.wilcoxon(
                    merged["compare_value"],
                    merged["baseline_value"],
                    zero_method="wilcox",
                    alternative="two-sided",
                    correction=False,
                    mode="auto",
                )
                row["statistic"] = float(test.statistic)
                row["p_value"] = float(test.pvalue)
                row["effect_direction"] = _effect_direction(float(merged["compare_value"].mean()), float(merged["baseline_value"].mean()), metric)
                row["significant_at_0_05"] = bool(float(test.pvalue) < 0.05)
            except ValueError as exc:
                row["p_value"] = 1.0
                row["effect_direction"] = _effect_direction(float(merged["compare_value"].mean()), float(merged["baseline_value"].mean()), metric)
                row["significant_at_0_05"] = False
                row["note"] = str(exc)
            raw_rows.append(row)

    p_values = [r["p_value"] for r in raw_rows if r.get("p_value") is not None]
    if p_values:
        _, pvals_bh, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        idx = 0
        for row in raw_rows:
            if row.get("p_value") is None:
                continue
            row["p_value_bh"] = float(pvals_bh[idx])
            idx += 1
    return raw_rows


def _effect_direction(compare_mean: float, baseline_mean: float, metric: str) -> str:
    """
    根据指标口径判断 compare 方法相对 baseline 的方向。
    """
    if abs(compare_mean - baseline_mean) <= 1e-12:
        return "equal"
    lower_better = {"gnae", "total_violations", "solve_time_sec", "objective_value"}
    if metric in lower_better:
        return "better" if compare_mean < baseline_mean else "worse"
    return "higher" if compare_mean > baseline_mean else "lower"


def _align_by_code(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于“代码”字段对齐两份分配矩阵，确保 GNAE 的逐元素比较一致。
    """
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


def _calibrate_c_constraints_from_label(
    constraint_config: dict,
    label_df: pd.DataFrame,
    rounds: List[str],
) -> None:
    """
    基于 label 的实际投放结构，对 C 类相关阈值做最小化校准，避免因阈值过紧导致模型不可行。
    """
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


def _auto_ga_params(n_products: int, n_rounds: int) -> tuple[float, int]:
    """
    按问题规模给出 GA 的默认运行时长与种群大小。
    """
    n_products = max(0, int(n_products))
    n_rounds = max(1, int(n_rounds))
    complexity = float(n_products * n_rounds)
    time_sec = max(60.0, min(240.0, 60.0 + 0.04 * complexity))
    pop = int(max(20, min(120, 20 + 0.03 * n_products + 5 * n_rounds)))
    return float(time_sec), int(pop)


if __name__ == "__main__":
    raise SystemExit(main())

