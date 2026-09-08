from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, compute_objective_value, validate_constraints
from bench.report import (
    MonthMethodReportRow,
    violations_to_count_rows,
    violations_to_detail_rows,
    write_allocations_xlsx,
    write_summary_csv,
    write_violation_details_csv,
    write_violation_counts_csv,
    write_violations_json,
)
from bench.solvers.lp_solver import solve_with_lp
from bench.solvers.ga import solve_with_ga
from bench.solvers.rule_based import solve_with_rule_based
from bench.solvers.sa import solve_with_simulated_annealing


def main() -> int:
    """
    批量跑 202501~202506 的对比实验，并输出汇总报表与可选明细。
    """
    parser = argparse.ArgumentParser(
        description="对比实验入口：批量读取本地 exp/data1/input 下的月份文件，运行 LP / RuleBased / SA / GA，并输出汇总与约束违背明细。",
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
        default=str(Path(__file__).resolve().parent / "output"),
        help="输出目录（生成 benchmark_summary*.csv、constraint_violation_*.csv 等）",
    )
    parser.add_argument(
        "--export-details",
        action="store_true",
        help="是否导出每月×方法的分配矩阵明细 Excel 与 violations JSON",
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
        help="LP 求解时间上限（秒）",
    )
    parser.add_argument(
        "--sa-time",
        type=float,
        default=30.0,
        help="SA 最大运行时长（秒）。SA 同时受 --sa-iters 限制，先达到者先停。",
    )
    parser.add_argument(
        "--sa-iters",
        type=int,
        default=50000,
        help="SA 最大迭代次数（建议设大，否则会出现时间没跑满就提前结束）",
    )
    parser.add_argument(
        "--ga-time",
        type=float,
        default=60,
        help="GA 最大运行时长（秒）。若传 0，则按数据规模自适应且下限为 60 秒。",
    )
    parser.add_argument(
        "--ga-pop",
        type=int,
        default=0,
        help="GA 种群大小（0 表示按数据规模自适应）",
    )
    parser.add_argument(
        "--ga-seed",
        type=int,
        default=42,
        help="GA 随机种子（用于复现实验）",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)

    cfg = default_experiment_configs()

    rows: List[MonthMethodReportRow] = []
    violation_count_rows: List[dict] = []
    violation_detail_rows: List[dict] = []

    input_paths = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if args.max_files and args.max_files > 0:
        input_paths = input_paths[: args.max_files]

    for input_path in input_paths:
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        rounds = month.rounds
        constraint_config = dict(cfg.constraint_config)
        if "按价" in month.product_data.columns:
            price_based_cnt = int(
                month.product_data["按价"].fillna("").astype(str).str.strip().ne("").sum()
            )
            if price_based_cnt <= 1:
                constraint_config["enable_price_based_constraints"] = False
        if month.label_data is not None:
            _calibrate_c_constraints_from_label(constraint_config, month.label_data, rounds)

        print(
            json.dumps(
                {
                    "month_id": month.month_id,
                    "lp_time_limit_sec": int(args.lp_time_limit),
                    "constraint_config": constraint_config,
                    "objective_config": cfg.objective_config,
                },
                ensure_ascii=False,
                default=str,
            )
        )

        lp_out = solve_with_lp(
            data_loader=month.data_loader,
            constraint_config=constraint_config,
            objective_config=cfg.objective_config,
            time_limit_sec=int(args.lp_time_limit),
        )
        heuristic_out = solve_with_rule_based(
            data_loader=month.data_loader,
            constraint_config=constraint_config,
        )
        sa_out = solve_with_simulated_annealing(
            data_loader=month.data_loader,
            constraint_config=constraint_config,
            objective_config=cfg.objective_config,
            max_time_sec=float(args.sa_time),
            max_iters=int(args.sa_iters),
            seed=42,
        )
        auto_ga_time, auto_ga_pop = _auto_ga_params(n_products=len(month.product_data), n_rounds=len(rounds))
        ga_out = solve_with_ga(
            data_loader=month.data_loader,
            constraint_config=constraint_config,
            objective_config=cfg.objective_config,
            max_time_sec=float(args.ga_time) if float(args.ga_time) > 0 else float(auto_ga_time),
            pop_size=int(args.ga_pop) if int(args.ga_pop) > 0 else int(auto_ga_pop),
            seed=int(args.ga_seed),
        )

        for method, out in [
            ("LP", lp_out),
            ("RuleBased", heuristic_out),
            ("SA", sa_out),
            ("GA", ga_out),
        ]:
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
                    objective_config=cfg.objective_config,
                )

            gnae = None
            if month.label_data is not None and isinstance(out.allocation_matrix_full, pd.DataFrame) and not out.allocation_matrix_full.empty and has_round_cols:
                alloc_std = standardize_allocation_matrix(out.allocation_matrix_full, rounds)
                label_std = standardize_allocation_matrix(month.label_data, rounds)
                alloc_aligned, label_aligned = _align_by_code(alloc_std, label_std)
                gnae = compute_gnae(alloc_aligned, label_aligned, rounds)

            rows.append(
                MonthMethodReportRow(
                    month_id=month.month_id,
                    method=method,
                    status=out.status,
                    feasible=constraint_check.feasible,
                    total_violations=constraint_check.total_violations,
                    solve_time_sec=float(out.solve_time_sec),
                    objective_value=float(objective_value) if objective_value is not None else None,
                    gnae=float(gnae) if gnae is not None else None,
                )
            )

            for r in violations_to_count_rows(constraint_check.violations):
                violation_count_rows.append(
                    {
                        "month_id": month.month_id,
                        "method": method,
                        "constraint_name": r.get("constraint_name"),
                        "count": r.get("count"),
                    }
                )

            for r in violations_to_detail_rows(constraint_check.violations):
                violation_detail_rows.append(
                    {
                        "month_id": month.month_id,
                        "method": method,
                        "constraint_name": r.get("constraint_name"),
                        "violation_index": r.get("violation_index"),
                        "violation": r.get("violation"),
                    }
                )

            if args.export_details:
                detail_name = f"benchmark_details_{month.month_id}_{method}.xlsx"
                extra_sheets = None
                if method == "LP":
                    def _as_text(v: object) -> str:
                        if isinstance(v, (dict, list, tuple)):
                            return json.dumps(v, ensure_ascii=False, default=str)
                        return str(v)

                    extra_sheets = {
                        "lp_params": pd.DataFrame(
                            [
                                {"key": "month_id", "value": month.month_id},
                                {"key": "lp_time_limit_sec", "value": int(args.lp_time_limit)},
                            ]
                        ),
                        "constraint_config": pd.DataFrame(
                            [{"key": k, "value": _as_text(v)} for k, v in constraint_config.items()]
                        ),
                        "objective_config": pd.DataFrame(
                            [{"key": k, "value": _as_text(v)} for k, v in cfg.objective_config.items()]
                        ),
                    }
                write_allocations_xlsx(out.allocation_matrix_full, output_dir / detail_name, extra_sheets=extra_sheets)
                vio_name = f"constraint_violations_{month.month_id}_{method}.json"
                write_violations_json(constraint_check.violations, output_dir / vio_name)

    write_summary_csv(rows, output_dir / "benchmark_summary.csv")
    if violation_count_rows:
        write_violation_counts_csv(violation_count_rows, output_dir / "constraint_violation_counts.csv")
    if violation_detail_rows:
        write_violation_details_csv(violation_detail_rows, output_dir / "constraint_violation_details.csv")

    print(
        json.dumps(
            {
                "输出文件说明": {
                    "benchmark_summary.csv": "每月×方法汇总：status/feasible/total_violations/solve_time_sec/objective_value/gnae",
                    "benchmark_summary.csv 字段说明": {
                        "month_id": "月份标识（例如 202501）",
                        "method": "方法名（LP/RuleBased/SA/GA）",
                        "status": "求解状态（可能包含 auto: 自动回退信息与 fallback: 回退信息）",
                        "feasible": "后端校验器 overall_valid（不是 LP 的 Optimal/Infeasible）",
                        "total_violations": "后端校验器统计的违反条数汇总",
                        "solve_time_sec": "该方法的耗时（秒）",
                        "objective_value": "按后端目标项数值化复刻的目标值（最小化口径，越小越好）",
                        "gnae": "与 label 的 GNAE（越小越好；缺 label 则为空）",
                    },
                    "constraint_violation_counts.csv": "每月×方法×约束项的违反次数统计",
                    "constraint_violation_details.csv": "每月×方法×约束项的逐条 violation 明细（JSON 字符串）",
                    "benchmark_details_<YYYYMM>_<Method>.xlsx": "（--export-details）每月×方法分配矩阵明细；LP 还会附带 lp_params/constraint_config/objective_config",
                    "constraint_violations_<YYYYMM>_<Method>.json": "（--export-details）后端校验器输出的完整 violations 结构",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


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
    rounds: list[str],
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
    n_products = max(0, int(n_products))
    n_rounds = max(1, int(n_rounds))
    complexity = float(n_products * n_rounds)
    time_sec = max(60.0, min(240.0, 60.0 + 0.04 * complexity))
    pop = int(max(20, min(120, 20 + 0.03 * n_products + 5 * n_rounds)))
    return float(time_sec), int(pop)


if __name__ == "__main__":
    raise SystemExit(main())
