"""
LP 自动松弛路径入口：逐月复现正式 LP 的自动回退顺序，并生成步骤表、过程报告、路径图和关键解。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

# 使用无界面后端，确保在批处理环境可稳定导出路径图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data
from bench.metrics import validate_constraints
from bench.report import write_allocations_xlsx, write_rows_csv
from bench.solvers.lp_solver import RelaxationAttempt, solve_with_lp_with_trace


def main() -> int:
    """按现有求解器的真实顺序生成每个月的自动松弛路径。"""
    parser = argparse.ArgumentParser(
        description="LP 自动松弛路径：记录当前求解器实际执行的约束回退过程，而非全局最小松弛搜索。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "lp_relaxation_path"),
    )
    parser.add_argument("--max-files", type=int, default=0, help="最多处理多少个输入文件；0 表示全量。")
    parser.add_argument("--lp-time-limit", type=int, default=60, help="每个 LP 松弛步骤的求解时间上限（秒）。")
    parser.add_argument("--no-export-details", action="store_true", help="不导出初始、首次 LP 最优和回退的关键矩阵。")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if args.max_files > 0:
        input_paths = input_paths[: args.max_files]

    cfg = default_experiment_configs()
    trace_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    report_sections: List[str] = []
    for month_index, input_path in enumerate(input_paths, start=1):
        month = load_month_data(input_path=input_path, label_dir=label_dir)
        constraint_config = _build_constraint_config(cfg.constraint_config, month)
        print(
            json.dumps(
                {
                    "stage": "lp_relaxation_path",
                    "month_id": month.month_id,
                    "month_progress": f"{month_index}/{len(input_paths)}",
                    "message": "开始复现 LP 自动松弛路径",
                },
                ensure_ascii=False,
            )
        )
        trace = solve_with_lp_with_trace(
            data_loader=month.data_loader,
            constraint_config=constraint_config,
            objective_config=dict(cfg.objective_config),
            time_limit_sec=int(args.lp_time_limit),
        )
        first_optimal_step = _first_optimal_step(trace.attempts)
        month_rows = _build_trace_rows(month, trace.attempts, first_optimal_step)
        trace_rows.extend(month_rows)
        summary = _build_summary_row(month.month_id, trace, first_optimal_step)
        summary_rows.append(summary)
        report_sections.append(_build_report_section(month.month_id, month_rows, summary))
        if not args.no_export_details:
            _export_key_attempts(output_dir, month.month_id, trace.attempts, first_optimal_step)
        print(
            json.dumps(
                {
                    "stage": "lp_relaxation_path",
                    "month_id": month.month_id,
                    "attempt_count": len(trace.attempts),
                    "first_optimal_step": first_optimal_step,
                    "final_status": trace.output.status,
                },
                ensure_ascii=False,
            )
        )

    write_rows_csv(trace_rows, output_dir / "lp_relaxation_trace.csv")
    write_rows_csv(summary_rows, output_dir / "lp_relaxation_path_summary.csv")
    _write_markdown_report(output_dir / "lp_relaxation_path_report.md", report_sections)
    _write_path_figure(pd.DataFrame(trace_rows), output_dir)
    return 0


def _build_constraint_config(base: Dict[str, Any], month: object) -> Dict[str, Any]:
    """复用主 benchmark 的约束配置、按价开关与标签校准口径。"""
    config = dict(base)
    if "按价" in month.product_data.columns:
        price_based_count = int(month.product_data["按价"].fillna("").astype(str).str.strip().ne("").sum())
        if price_based_count <= 1:
            config["enable_price_based_constraints"] = False
    if month.label_data is not None:
        _calibrate_c_constraints_from_label(config, month.label_data, month.rounds)
    return config


def _first_optimal_step(attempts: List[RelaxationAttempt]) -> Optional[int]:
    """返回首次获得 LP Optimal 解的步骤号；RuleBased 回退不被视为 LP 最优解。"""
    for attempt in attempts:
        if attempt.is_lp_attempt and attempt.is_optimal:
            return int(attempt.step_index)
    return None


def _build_trace_rows(month: object, attempts: List[RelaxationAttempt], first_optimal_step: Optional[int]) -> List[Dict[str, Any]]:
    """将每个求解步骤转换为可直接审阅、筛选和作图的长表记录。"""
    rows: List[Dict[str, Any]] = []
    for attempt in attempts:
        allocation = attempt.allocation_matrix_full
        has_round_columns = isinstance(allocation, pd.DataFrame) and not allocation.empty and all(round_name in allocation.columns for round_name in month.rounds)
        feasible: Optional[bool] = None
        total_violations: Optional[int] = None
        if has_round_columns:
            check = validate_constraints(month.data_loader, allocation, attempt.applied_constraint_config)
            feasible = bool(check.feasible)
            total_violations = int(check.total_violations)
        config = attempt.applied_constraint_config
        rows.append(
            {
                "month_id": str(month.month_id),
                "step_index": int(attempt.step_index),
                "action": attempt.action,
                "action_label": _action_label(attempt.action),
                "cumulative_adjustments": json.dumps(attempt.cumulative_adjustments, ensure_ascii=False),
                "is_lp_attempt": bool(attempt.is_lp_attempt),
                "is_optimal": bool(attempt.is_optimal),
                "is_first_optimal": bool(first_optimal_step == attempt.step_index),
                "status": attempt.status,
                "solve_time_sec": float(attempt.solve_time_sec),
                "objective_value": attempt.objective_value,
                "has_allocation_matrix": bool(has_round_columns),
                "feasible": feasible,
                "total_violations": total_violations,
                "volume_tolerance": config.get("volume_tolerance"),
                "enable_demand_split_constraints": config.get("enable_demand_split_constraints"),
                "enable_price_based_constraints": config.get("enable_price_based_constraints"),
                "enable_c_type_constraints": config.get("enable_c_type_constraints"),
                "enable_price_constraints": config.get("enable_price_constraints"),
                "applied_constraint_config": json.dumps(config, ensure_ascii=False, default=str, sort_keys=True),
            }
        )
    return rows


def _build_summary_row(month_id: str, trace: object, first_optimal_step: Optional[int]) -> Dict[str, Any]:
    """提炼单个月份的首次可行位置、最终状态与最终 LP 配置。"""
    attempts = trace.attempts
    final_attempt = attempts[-1] if attempts else None
    return {
        "month_id": str(month_id),
        "attempt_count": int(len(attempts)),
        "lp_attempt_count": int(sum(1 for attempt in attempts if attempt.is_lp_attempt)),
        "first_optimal_step": first_optimal_step,
        "used_rule_based_fallback": bool(trace.used_rule_based_fallback),
        "final_status": trace.output.status,
        "final_solve_time_sec": float(trace.output.solve_time_sec),
        "final_objective_value": trace.output.objective_value,
        "final_action": final_attempt.action if final_attempt is not None else None,
        "final_constraint_config": json.dumps(trace.final_constraint_config, ensure_ascii=False, default=str, sort_keys=True),
    }


def _export_key_attempts(output_dir: Path, month_id: str, attempts: List[RelaxationAttempt], first_optimal_step: Optional[int]) -> None:
    """仅保留初始、首次 LP 最优与最终 RuleBased 回退等关键解，避免导出大量中间文件。"""
    selected_steps = {0}
    if first_optimal_step is not None:
        selected_steps.add(first_optimal_step)
    if attempts and not attempts[-1].is_lp_attempt:
        selected_steps.add(attempts[-1].step_index)
    for attempt in attempts:
        if attempt.step_index not in selected_steps or attempt.allocation_matrix_full.empty:
            continue
        label = "initial" if attempt.step_index == 0 else "first_optimal" if attempt.step_index == first_optimal_step else "fallback_rule_based"
        extra_sheets = {
            "trace": pd.DataFrame(
                [
                    {
                        "step_index": attempt.step_index,
                        "action": attempt.action,
                        "status": attempt.status,
                        "is_lp_attempt": attempt.is_lp_attempt,
                        "is_optimal": attempt.is_optimal,
                    }
                ]
            ),
            "constraint_config": pd.DataFrame([{"key": key, "value": str(value)} for key, value in attempt.applied_constraint_config.items()]),
        }
        write_allocations_xlsx(
            attempt.allocation_matrix_full,
            output_dir / f"lp_relaxation_solution_{month_id}_{label}.xlsx",
            extra_sheets=extra_sheets,
        )


def _build_report_section(month_id: str, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """生成可直接纳入实验附录的单月自动松弛过程描述。"""
    lines = [f"## {month_id}", ""]
    lines.append(f"- 共执行 {summary['lp_attempt_count']} 次 LP 尝试；最终状态：`{summary['final_status']}`。")
    if summary["first_optimal_step"] is None:
        lines.append("- 未获得 LP Optimal 解；最终结果为 RuleBased 回退或无有效 LP 解，不应表述为 LP 最优松弛解。")
    else:
        lines.append(f"- 首次获得 LP Optimal 解的步骤：{summary['first_optimal_step']}。")
    lines.append("")
    lines.append("| 步骤 | 动作 | 状态 | 是否为首次 LP Optimal | 可行性 | 违背数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        feasible_text = "" if row["feasible"] is None else str(row["feasible"])
        violations_text = "" if row["total_violations"] is None else str(row["total_violations"])
        lines.append(
            f"| {row['step_index']} | {row['action_label']} | {row['status']} | {row['is_first_optimal']} | {feasible_text} | {violations_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_markdown_report(path: Path, report_sections: List[str]) -> None:
    """写出总报告，并明确路径是既有自动策略而非全局最小松弛搜索。"""
    intro = [
        "# LP 自动松弛路径报告",
        "",
        "本报告忠实记录当前 LP 封装的实际自动回退顺序，不执行全组合搜索，因此不代表全局最小松弛代价路径。",
        "",
    ]
    path.write_text("\n".join(intro + report_sections), encoding="utf-8")


def _write_path_figure(df: pd.DataFrame, output_dir: Path) -> None:
    """以按步骤连接的节点图呈现每个月的真实自动松弛路径。"""
    if df.empty:
        return
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    months = sorted(df["month_id"].astype(str).unique().tolist())
    fig, axis = plt.subplots(figsize=(15, max(4.0, 1.25 * len(months))), constrained_layout=True)
    color_map = {"optimal": "#009E73", "fallback": "#CC79A7", "infeasible": "#D55E00", "other": "#999999"}
    for y_value, month_id in enumerate(months):
        part = df[df["month_id"].astype(str) == month_id].sort_values("step_index")
        x_values = part["step_index"].astype(float).tolist()
        axis.plot(x_values, [y_value] * len(x_values), color="#666666", linewidth=1.5, zorder=1)
        for _, row in part.iterrows():
            category = _node_category(row)
            axis.scatter(row["step_index"], y_value, s=130, color=color_map[category], edgecolors="black", linewidths=0.5, zorder=2)
            axis.annotate(_action_figure_label(str(row["action"])), (row["step_index"], y_value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, rotation=25)
    axis.set_yticks(range(len(months)), months)
    axis.set_xlabel("Relaxation step")
    axis.set_ylabel("Month")
    axis.set_title("LP automatic relaxation path (green: LP optimal; red: infeasible; purple: RuleBased fallback)")
    axis.grid(True, axis="x", alpha=0.25)
    fig.savefig(output_dir / "lp_relaxation_path.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "lp_relaxation_path.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _node_category(row: pd.Series) -> str:
    """根据步骤结果选择路径节点颜色。"""
    if not bool(row["is_lp_attempt"]):
        return "fallback"
    if bool(row["is_optimal"]):
        return "optimal"
    if str(row["status"]) == "Infeasible":
        return "infeasible"
    return "other"


def _action_label(action: str) -> str:
    """将内部配置键转换成路径图和报告中可读的中文动作名。"""
    mapping = {
        "initial": "初始约束",
        "enable_demand_split_constraints=False": "关闭需求集中",
        "enable_price_based_constraints=False": "关闭按价比例",
        "enable_c_type_constraints=False": "关闭 C 类",
        "enable_price_constraints=False": "关闭价格约束",
        "fallback:RuleBased": "RuleBased 回退",
    }
    if action in mapping:
        return mapping[action]
    if action.startswith("volume_tolerance="):
        return "总量容差=" + action.split("=", 1)[1]
    return action


def _action_figure_label(action: str) -> str:
    """将路径图注释转换为英文，并以通用结构约束表述避免主图出现内部业务缩写。"""
    mapping = {
        "initial": "Initial constraints",
        "enable_demand_split_constraints=False": "Disable demand concentration",
        "enable_price_based_constraints=False": "Disable price proportion",
        "enable_c_type_constraints=False": "Relax structural constraint",
        "enable_price_constraints=False": "Disable price constraint",
        "fallback:RuleBased": "RuleBased fallback",
    }
    if action in mapping:
        return mapping[action]
    if action.startswith("volume_tolerance="):
        return "Volume tolerance=" + action.split("=", 1)[1]
    return action


def _calibrate_c_constraints_from_label(constraint_config: Dict[str, Any], label_df: pd.DataFrame, rounds: List[str]) -> None:
    """沿用 benchmark 标签校准，避免 C 类阈值与真实历史结构冲突。"""
    if "C" not in label_df.columns:
        return
    is_c = label_df["C"].fillna("").astype(str).str.strip().ne("")
    c_by_round = label_df.loc[is_c, rounds].fillna(0.0).astype(float).sum(axis=0)
    constraint_config["c_type_volume_limit"] = max(float(constraint_config.get("c_type_volume_limit", 0.0) or 0.0), float(c_by_round.max()) + 1.0)
    if "C类" not in label_df.columns:
        return
    c_total = c_by_round.replace(0.0, pd.NA)
    chang_by_round = label_df.loc[label_df["C类"].fillna("").astype(str).eq("长"), rounds].fillna(0.0).astype(float).sum(axis=0)
    xi_by_round = label_df.loc[label_df["C类"].fillna("").astype(str).eq("细"), rounds].fillna(0.0).astype(float).sum(axis=0)
    constraint_config["chang_type_volume_limit"] = max(float(constraint_config.get("chang_type_volume_limit", 0.0) or 0.0), float(chang_by_round.max()) + 1.0)
    constraint_config["xi_type_volume_limit"] = max(float(constraint_config.get("xi_type_volume_limit", 0.0) or 0.0), float(xi_by_round.max()) + 1.0)
    chang_ratio = (chang_by_round / c_total).dropna().max()
    xi_ratio = (xi_by_round / c_total).dropna().max()
    if pd.notna(chang_ratio):
        constraint_config["chang_type_ratio"] = max(float(constraint_config.get("chang_type_ratio", 0.0) or 0.0), float(chang_ratio) + 0.01)
    if pd.notna(xi_ratio):
        constraint_config["xi_type_ratio"] = max(float(constraint_config.get("xi_type_ratio", 0.0) or 0.0), float(xi_ratio) + 0.01)


if __name__ == "__main__":
    raise SystemExit(main())
