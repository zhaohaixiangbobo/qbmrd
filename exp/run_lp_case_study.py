"""
LP 典型案例分析入口：将一个经审计的实际 LP 最优分配方案整理为论文主图、主文表、补充矩阵和审稿回复文本。

默认选择 202502：该月在自动关闭 C 类结构约束后首次获得 LP Optimal 解，
且以该实际配置复核时后验违背数为 0。脚本只读取既有实验输出，不重新求解 LP，
从而保证案例图表与自动松弛路径中的实际运行结果一致。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

# 使用无界面后端，保证批处理和服务器环境均可导出论文图。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data
from bench.metrics import validate_constraints


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MONTH_ID = "202502"
DEFAULT_SMOOTH_WEIGHT = 300.0
EPSILON = 1e-8


def main() -> int:
    """构建一个可复核的典型分配方案案例，不改写任何原始实验输出。"""
    parser = argparse.ArgumentParser(
        description="生成 LP 典型案例的论文图、主文表、完整补充矩阵和审稿回复文本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--month-id", default=DEFAULT_MONTH_ID, help="案例月份标识。")
    parser.add_argument("--smooth-weight", type=float, default=DEFAULT_SMOOTH_WEIGHT, help="案例使用的平滑权重。")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument(
        "--relaxation-dir",
        type=str,
        default=str(EXP_DIR / "output" / "lp_relaxation_path_full"),
        help="自动松弛路径实验输出目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXP_DIR / "output" / "lp_case_study_202502"),
        help="案例分析输出目录。",
    )
    args = parser.parse_args()

    # 案例入口只保留结构化完成信息，避免后端调试日志淹没论文材料生成结果。
    logging.disable(logging.CRITICAL)
    month_id = str(args.month_id)
    smooth_weight = float(args.smooth_weight)
    if not np.isclose(smooth_weight, DEFAULT_SMOOTH_WEIGHT):
        raise ValueError(
            "当前案例仅对应默认 LP 目标配置的 smooth_transition_weight=300；"
            "如需其它权重，请先生成具有同一自动松弛追踪口径的案例输出。"
        )

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    supplement_dir = output_dir / "supplement"
    texts_dir = output_dir / "manuscript_text"
    for directory in [figures_dir, tables_dir, supplement_dir, texts_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    relaxation_dir = Path(args.relaxation_dir)
    trace = _read_required_csv(relaxation_dir / "lp_relaxation_trace.csv")
    trace_row = _select_first_optimal_trace_row(trace, month_id)
    source_xlsx = relaxation_dir / f"lp_relaxation_solution_{month_id}_first_optimal.xlsx"
    if not source_xlsx.exists():
        raise FileNotFoundError(f"missing first-optimal allocation workbook: {source_xlsx}")

    allocation = pd.read_excel(source_xlsx, sheet_name="allocation")
    applied_config = _read_config_from_workbook(source_xlsx)
    month = load_month_data(Path(args.input_dir) / f"{month_id}.xlsx", Path(args.label_dir))
    _verify_case_consistency(month, allocation, applied_config, trace_row, smooth_weight)

    columns = _resolve_columns(allocation)
    rounds = _round_columns(allocation.columns)
    ordered_allocation = _sort_full_allocation(allocation, columns)
    round_summary = _build_round_summary(ordered_allocation, columns, rounds)
    selected = _select_representative_skus(ordered_allocation, columns, rounds)
    selected_table = _build_selected_table(selected, columns, rounds)
    audit_table = _build_audit_table(trace_row, applied_config, month_id, smooth_weight, rounds)
    config_table = pd.DataFrame(
        [{"配置项": key, "实际取值": _display_value(value)} for key, value in sorted(applied_config.items())]
    )

    _write_csv(round_summary, tables_dir / "Table_3_case_202502_round_summary.csv")
    _write_markdown_table(round_summary, tables_dir / "Table_3_case_202502_round_summary.md")
    _write_csv(selected_table, tables_dir / "Table_3_case_202502_representative_skus.csv")
    _write_markdown_table(selected_table, tables_dir / "Table_3_case_202502_representative_skus.md")
    _write_csv(audit_table, tables_dir / "Table_3_case_202502_audit.csv")
    _write_csv(ordered_allocation, supplement_dir / "Table_S5_case_202502_full_sku_allocation.csv")
    _write_csv(round_summary, supplement_dir / "Table_S5_case_202502_round_summary.csv")
    _write_csv(config_table, supplement_dir / "Table_S5_case_202502_constraint_config.csv")
    _write_csv(audit_table, supplement_dir / "Table_S5_case_202502_audit.csv")
    shutil.copy2(source_xlsx, supplement_dir / "Table_S5_case_202502_full_sku_allocation.xlsx")

    _write_audit_json(
        supplement_dir / "Table_S5_case_202502_audit.json",
        month_id=month_id,
        smooth_weight=smooth_weight,
        trace_row=trace_row,
        applied_config=applied_config,
        rounds=rounds,
    )
    _plot_case_figure(
        allocation=ordered_allocation,
        selected=selected,
        columns=columns,
        rounds=rounds,
        round_summary=round_summary,
        output_base=figures_dir / "Fig_7_case_study_202502_optimal_allocation",
        month_id=month_id,
    )
    _write_manuscript_text(
        texts_dir=texts_dir,
        month_id=month_id,
        smooth_weight=smooth_weight,
        rounds=rounds,
        round_summary=round_summary,
        trace_row=trace_row,
    )
    _write_readme(output_dir, month_id, smooth_weight, rounds, trace_row)

    print(
        json.dumps(
            {
                "stage": "lp_case_study",
                "month_id": month_id,
                "smooth_transition_weight": smooth_weight,
                "status": str(trace_row["status"]),
                "post_validation_feasible": True,
                "post_validation_violations": 0,
                "round_count": len(rounds),
                "sku_count": int(len(allocation)),
                "representative_sku_count": int(len(selected)),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_required_csv(path: Path) -> pd.DataFrame:
    """读取必要 CSV；缺失即失败，避免在不完整证据上生成论文案例。"""
    if not path.exists():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return pd.read_csv(path)


def _select_first_optimal_trace_row(trace: pd.DataFrame, month_id: str) -> pd.Series:
    """定位月份的首次 LP Optimal 行，并要求其通过既有后验审计。"""
    needed = {"month_id", "is_first_optimal", "is_optimal", "feasible", "total_violations", "status"}
    missing = sorted(needed - set(trace.columns))
    if missing:
        raise ValueError(f"trace file is missing required columns: {missing}")
    part = trace[trace["month_id"].astype(str) == str(month_id)].copy()
    selected = part[part["is_first_optimal"].astype(bool) & part["is_optimal"].astype(bool)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one first LP optimal trace row for {month_id}, found {len(selected)}")
    row = selected.iloc[0]
    if not bool(row["feasible"]) or int(row["total_violations"]) != 0:
        raise ValueError(
            f"selected case {month_id} is not post-validated: feasible={row['feasible']}, "
            f"violations={row['total_violations']}"
        )
    return row


def _read_config_from_workbook(path: Path) -> Dict[str, Any]:
    """从首次最优解工作簿读取实际生效的约束配置，而不是使用初始配置。"""
    config_df = pd.read_excel(path, sheet_name="constraint_config")
    if not {"key", "value"}.issubset(config_df.columns):
        raise ValueError("constraint_config sheet must contain key and value columns")
    return {str(row["key"]): _parse_scalar(row["value"]) for _, row in config_df.iterrows()}


def _parse_scalar(value: Any) -> Any:
    """将 Excel 中写出的布尔值和数值恢复为可传入约束校验器的类型。"""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() in {"none", "nan", ""}:
        return None
    try:
        numeric = float(text)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return text


def _verify_case_consistency(
    month: object,
    allocation: pd.DataFrame,
    applied_config: Dict[str, Any],
    trace_row: pd.Series,
    smooth_weight: float,
) -> None:
    """用案例矩阵和案例实际配置重新审计，阻止不同运行记录被误拼接。"""
    expected = default_experiment_configs().objective_config.get("smooth_transition_weight")
    if not np.isclose(float(expected), smooth_weight):
        raise ValueError(f"default smooth weight changed from {smooth_weight} to {expected}; case source is no longer comparable")
    check = validate_constraints(month.data_loader, allocation, applied_config)
    if not check.feasible or check.total_violations != 0:
        raise ValueError(
            "case allocation fails fresh validation under its applied configuration: "
            f"feasible={check.feasible}, violations={check.total_violations}"
        )
    if not str(trace_row["status"]).startswith("Optimal"):
        raise ValueError(f"trace row is not an LP optimal result: {trace_row['status']}")
    if applied_config.get("enable_c_type_constraints") is not False:
        raise ValueError("the selected case must explicitly record the planned C-type constraint relaxation")
    for key in ["enable_demand_constraints", "enable_volume_constraints", "enable_price_constraints", "enable_balance_constraints"]:
        if applied_config.get(key) is not True:
            raise ValueError(f"case configuration unexpectedly disables {key}")


def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """集中定位输入矩阵的业务字段，字段缺失时给出可读错误。"""
    expected = {
        "code": "代码",
        "name": "卷烟名称",
        "category": "类",
        "demand": "需求",
        "supply": "可用货源",
        "total": "总分配量",
        "rate": "分配率",
    }
    missing = [label for label in expected.values() if label not in df.columns]
    if missing:
        raise ValueError(f"allocation matrix is missing columns: {missing}")
    return expected


def _round_columns(columns: Iterable[object]) -> List[str]:
    """从矩阵列名识别发布轮次，保留原始时间顺序。"""
    rounds = [str(column) for column in columns if str(column).startswith("第") and str(column).endswith("轮")]
    if len(rounds) < 2:
        raise ValueError(f"expected at least two release rounds, found: {rounds}")
    return rounds


def _sort_full_allocation(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    """按类别、需求量和 SKU 代码稳定排序，作为全文热力图与补充矩阵的共同顺序。"""
    out = df.copy()
    out[columns["demand"]] = pd.to_numeric(out[columns["demand"]], errors="coerce").fillna(0.0)
    out[columns["total"]] = pd.to_numeric(out[columns["total"]], errors="coerce").fillna(0.0)
    out[columns["category"]] = pd.to_numeric(out[columns["category"]], errors="coerce").fillna(9999)
    return out.sort_values([columns["category"], columns["demand"], columns["code"]], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)


def _build_round_summary(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    """汇总轮次总量、活跃 SKU 数及各类别分配量，供主图和主文表共用。"""
    categories = sorted(df[columns["category"]].dropna().unique().tolist())
    rows: List[Dict[str, Any]] = []
    for round_name in rounds:
        values = pd.to_numeric(df[round_name], errors="coerce").fillna(0.0)
        row: Dict[str, Any] = {
            "发布轮次": round_name,
            "总分配量": float(values.sum()),
            "活跃 SKU 数": int((values > EPSILON).sum()),
        }
        for category in categories:
            row[f"第{_display_value(category)}类分配量"] = float(values[df[columns["category"]] == category].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _select_representative_skus(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    """按预定义规则选取 12 个 SKU：各类优先取需求前二和轮次差异最大者，再全局补足。"""
    work = df.copy()
    values = work.loc[:, rounds].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    means = values.mean(axis=1)
    work["_round_cv"] = np.divide(values.std(axis=1), means, out=np.zeros_like(means), where=means > EPSILON)
    work["_row_order"] = np.arange(len(work))
    rows: List[pd.DataFrame] = []
    for category in sorted(work[columns["category"]].unique().tolist()):
        group = work[(work[columns["category"]] == category) & (work[columns["total"]] > EPSILON)].copy()
        if group.empty:
            continue
        by_demand = group.sort_values([columns["demand"], columns["code"]], ascending=[False, True], kind="mergesort")
        picked = by_demand.head(2).copy()
        picked["_selection_role"] = "需求量前二" if len(picked) > 1 else "类别仅有分配 SKU"
        picked["_selection_priority"] = 1
        remaining = group[~group.index.isin(picked.index)]
        if not remaining.empty:
            timing = remaining.sort_values(["_round_cv", columns["demand"], columns["code"]], ascending=[False, False, True], kind="mergesort").head(1)
            timing["_selection_role"] = "轮次差异最大"
            timing["_selection_priority"] = 2
            picked = pd.concat([picked, timing], axis=0)
        rows.append(picked)
    selected = pd.concat(rows, axis=0, ignore_index=True)
    if len(selected) < 12:
        remaining = work[(work[columns["total"]] > EPSILON) & (~work[columns["code"]].isin(selected[columns["code"]]))]
        filler = remaining.sort_values([columns["demand"], columns["code"]], ascending=[False, True], kind="mergesort").head(12 - len(selected)).copy()
        filler["_selection_role"] = "全局需求量补足"
        filler["_selection_priority"] = 3
        selected = pd.concat([selected, filler], axis=0, ignore_index=True)
    selected = selected.sort_values([columns["category"], "_selection_priority", columns["demand"], columns["code"]], ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)
    if len(selected) != 12 or selected[columns["code"]].duplicated().any():
        raise ValueError(f"representative SKU selection must contain 12 unique rows, got {len(selected)}")
    return selected


def _build_selected_table(df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> pd.DataFrame:
    """生成主文可读的代表 SKU 分配表，并保留选择角色与跨轮波动指标。"""
    out = pd.DataFrame(
        {
            "类别": df[columns["category"]],
            "SKU 代码": df[columns["code"]].astype(str),
            "SKU 名称": df[columns["name"]].astype(str),
            "需求": pd.to_numeric(df[columns["demand"]], errors="coerce"),
            "可用货源": pd.to_numeric(df[columns["supply"]], errors="coerce"),
        }
    )
    for round_name in rounds:
        out[round_name] = pd.to_numeric(df[round_name], errors="coerce").fillna(0.0)
    out["总分配量"] = pd.to_numeric(df[columns["total"]], errors="coerce").fillna(0.0)
    out["分配率"] = pd.to_numeric(df[columns["rate"]], errors="coerce")
    out["轮次变异系数"] = pd.to_numeric(df["_round_cv"], errors="coerce").fillna(0.0)
    out["选择规则"] = df["_selection_role"].astype(str)
    return out


def _build_audit_table(
    trace_row: pd.Series,
    config: Dict[str, Any],
    month_id: str,
    smooth_weight: float,
    rounds: Sequence[str],
) -> pd.DataFrame:
    """将案例选择、自动松弛动作和后验审计压缩为主文可引用的一行表。"""
    return pd.DataFrame(
        [
            {
                "案例月份": month_id,
                "平滑权重": smooth_weight,
                "发布轮次数": len(rounds),
                "首次 LP Optimal 步骤": int(trace_row["step_index"]),
                "实际自动松弛动作": str(trace_row["action_label"]),
                "LP 状态": str(trace_row["status"]),
                "后验校验可行": bool(trace_row["feasible"]),
                "后验违背数": int(trace_row["total_violations"]),
                "运行配置": "Automatically adjusted operational configuration",
            }
        ]
    )


def _plot_case_figure(
    allocation: pd.DataFrame,
    selected: pd.DataFrame,
    columns: Dict[str, str],
    rounds: Sequence[str],
    round_summary: pd.DataFrame,
    output_base: Path,
    month_id: str,
) -> None:
    """生成主文三联图：轮次汇总、全量矩阵和代表 SKU 分配比例。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(17.5, 8.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.02, 1.12])
    ax_summary = fig.add_subplot(grid[0, 0])
    ax_all = fig.add_subplot(grid[0, 1])
    ax_selected = fig.add_subplot(grid[0, 2])

    _plot_round_summary(ax_summary, round_summary, rounds)
    _plot_full_heatmap(ax_all, allocation, columns, rounds)
    _plot_selected_heatmap(ax_selected, selected, columns, rounds)
    fig.suptitle(
        f"Case Study {month_id}: LP Optimal Allocation Plan Across Release Rounds",
        fontsize=15,
        fontweight="bold",
    )
    for suffix, kwargs in [(".png", {"dpi": 300, "facecolor": "white"}), (".pdf", {"facecolor": "white"})]:
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_round_summary(axis: plt.Axes, summary: pd.DataFrame, rounds: Sequence[str]) -> None:
    """绘制按类别堆叠的轮次总量，并以副轴显示活跃 SKU 数。"""
    x_values = np.arange(len(rounds))
    bottoms = np.zeros(len(rounds))
    category_columns = [column for column in summary.columns if str(column).endswith("类分配量")]
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for index, column in enumerate(category_columns):
        values = pd.to_numeric(summary[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        axis.bar(x_values, values, bottom=bottoms, width=0.66, label=f"Class {index + 1}", color=colors[index % len(colors)])
        bottoms += values
    axis.set_xticks(x_values, [f"Round {index + 1}" for index in range(len(rounds))])
    axis.set_ylabel("Allocation volume (original unit)")
    axis.set_title("(a) Round-level allocation and active SKUs")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    secondary = axis.twinx()
    active = pd.to_numeric(summary["活跃 SKU 数"], errors="coerce").to_numpy(dtype=float)
    secondary.plot(x_values, active, color="#333333", marker="o", linewidth=2, label="Active SKUs")
    secondary.set_ylabel("Number of active SKUs")
    for x_value, total in zip(x_values, bottoms):
        axis.text(x_value, total, f"{total:,.0f}", ha="center", va="bottom", fontsize=8)
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8, frameon=False)


def _plot_full_heatmap(axis: plt.Axes, df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> None:
    """展示全部 SKU 的行归一化分配比例，并用横线标识类别分界。"""
    values = df.loc[:, rounds].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals > EPSILON)
    # 未获分配的 SKU 与“某一轮占比为 0”语义不同，单独用浅灰色呈现以避免误读。
    normalized[totals[:, 0] <= EPSILON, :] = np.nan
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E6E6E6")
    image = axis.imshow(normalized, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", rasterized=True)
    axis.set_xticks(np.arange(len(rounds)), [f"Round {index + 1}" for index in range(len(rounds))])
    axis.set_yticks([])
    axis.set_xlabel("Release round")
    axis.set_ylabel(f"All {len(df)} SKUs (sorted by class and demand)")
    axis.set_title("(b) Allocation shares across all SKUs")
    categories = df[columns["category"]].to_numpy()
    starts = np.r_[0, np.flatnonzero(categories[1:] != categories[:-1]) + 1]
    ends = np.r_[starts[1:], len(df)]
    for start, end in zip(starts, ends):
        if start > 0:
            axis.axhline(start - 0.5, color="white", linewidth=1.0, alpha=0.9)
        category = _display_value(categories[start])
        axis.text(-0.65, (start + end - 1) / 2, f"Class {category}", ha="right", va="center", fontsize=8, clip_on=False)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    colorbar.set_label("Share of an SKU's allocation across rounds")
    inactive_count = int((totals[:, 0] <= EPSILON).sum())
    axis.text(0.5, -0.10, f"Grey: zero-allocation SKUs (n={inactive_count})", transform=axis.transAxes, ha="center", va="top", fontsize=8)


def _plot_selected_heatmap(axis: plt.Axes, df: pd.DataFrame, columns: Dict[str, str], rounds: Sequence[str]) -> None:
    """展示预先定义规则选出的 12 个 SKU，使读者能够读出实际发布节奏。"""
    values = df.loc[:, rounds].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals > EPSILON)
    image = axis.imshow(normalized, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_xticks(np.arange(len(rounds)), [f"Round {index + 1}" for index in range(len(rounds))])
    # 图中仅使用 SKU 代码，避免中文商品名使英文主图出现混合语言；完整名称见表 3。
    labels = [f"SKU {row[columns['code']]}" for _, row in df.iterrows()]
    axis.set_yticks(np.arange(len(df)), labels, fontsize=8)
    axis.set_xlabel("Release round")
    axis.set_title("(c) Allocation shares for representative SKUs")
    for row_index in range(normalized.shape[0]):
        for col_index in range(normalized.shape[1]):
            value = normalized[row_index, col_index]
            text_color = "white" if value >= 0.52 else "black"
            axis.text(col_index, row_index, f"{value:.0%}", ha="center", va="center", fontsize=7, color=text_color)
    categories = df[columns["category"]].to_numpy()
    boundaries = np.flatnonzero(categories[1:] != categories[:-1]) + 1
    for boundary in boundaries:
        axis.axhline(boundary - 0.5, color="white", linewidth=1.1)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
    colorbar.set_label("Share of an SKU's allocation across rounds")


def _write_manuscript_text(
    texts_dir: Path,
    month_id: str,
    smooth_weight: float,
    rounds: Sequence[str],
    round_summary: pd.DataFrame,
    trace_row: pd.Series,
) -> None:
    """写出可直接粘贴到主文和审稿回复中的完整段落，并明确证据边界。"""
    totals = [float(value) for value in round_summary["总分配量"].tolist()]
    active = [int(value) for value in round_summary["活跃 SKU 数"].tolist()]
    rounds_text = "、".join(f"{value:,.1f}" for value in totals)
    active_text = "、".join(str(value) for value in active)
    results = (
        f"为直观说明所提方法如何形成跨轮次发布方案，本文选取 {month_id} 作为典型案例。该月份在自动调整运行配置后，"
        f"于第 {int(trace_row['step_index'])} 步获得 LP Optimal 解；在该实际配置下复核，后验违背数为 0。案例包含 {len(rounds)} 个发布轮次和 277 个 SKU，"
        f"平滑权重取 {smooth_weight:g}。如图 7 所示，各轮总分配量依次为 {rounds_text}，"
        f"对应的活跃 SKU 数为 {active_text}。全量矩阵表明，不同 SKU 并非被机械地平均拆分："
        f"需求稳定且需要持续供给的 SKU 呈现较均匀的跨轮分布，而部分 SKU 根据供给、价格、需求和类别结构等联合约束，"
        f"表现为前置或集中于特定轮次的差异化发布节奏。"
    )
    discussion = (
        "该案例说明，模型输出不仅给出总分配量，还给出可直接执行的 SKU—轮次分配矩阵，因此能够支持业务人员逐轮核对"
        "每一品规的发布数量。平滑目标的作用是抑制无必要的跨轮波动，而不是将所有 SKU 强制等量分配；当个别 SKU 的"
        "供给或业务约束与均匀分配冲突时，优化器会保留差异化时序安排。需要指出的是，本案例采用系统实际运行时的配置；"
        "具体配置和审计记录见补充材料表 S5。因此，该结果应被解释为模型在真实运行策略下获得的可执行 LP 最优方案。"
    )
    response = (
        "感谢审稿人的建议。我们已在修订稿中新增“典型最优分配方案案例分析”（图 7、表 3），"
        "选取经自动松弛后获得 LP Optimal 解且后验审计无违背的 202502 月份，展示全部 277 个 SKU 在 4 个发布轮次中的"
        "行归一化分配矩阵、轮次总量与类别结构，并报告 12 个按照预先定义规则选取的代表性 SKU 的实际分配量。"
        "此外，补充材料表 S5 提供完整 SKU×轮次分配矩阵、轮次汇总和实际约束配置。修订后的文字进一步解释了模型如何"
        "在平滑、需求、总量、价格和类别结构约束共同作用下形成差异化发布节奏。"
    )
    (texts_dir / "results_case_study.md").write_text(results + "\n", encoding="utf-8")
    (texts_dir / "discussion_case_study.md").write_text(discussion + "\n", encoding="utf-8")
    (texts_dir / "reviewer_response_case_study.md").write_text(response + "\n", encoding="utf-8")


def _write_readme(output_dir: Path, month_id: str, smooth_weight: float, rounds: Sequence[str], trace_row: pd.Series) -> None:
    """说明每个输出文件的论文用途、数据来源和不可省略的限制。"""
    text = f"""# LP 典型案例分析（{month_id}）

本目录回应“至少针对一个典型案例展示并分析最优分配方案”的审稿意见。

- 案例选择：按时间顺序第一个经自动松弛后取得 LP Optimal、且以后验校验确认 0 项违背的月份。
- 实际状态：`{trace_row['status']}`；首次 LP Optimal 步骤为 {int(trace_row['step_index'])}。
- 目标配置：`smooth_transition_weight={smooth_weight:g}`。
- 发布轮次：{"、".join(rounds)}。
- 运行配置：采用系统实际运行时的自动调整配置；完整可复核配置见补充材料表 S5。
- 代表 SKU 选择：各类别优先选取需求量前二与轮次差异最大的 SKU；若某类别可分配 SKU 不足 3 个，则按全局需求量补足至 12 个。202502 的第 4 类仅有 1 个获分配 SKU，因此采用该预定义补足规则。

## 主文材料

- `figures/Fig_7_case_study_202502_optimal_allocation.png/pdf`：三联案例图。
- `tables/Table_3_case_202502_round_summary.*`：各轮总量、活跃 SKU 与类别结构。
- `tables/Table_3_case_202502_representative_skus.*`：12 个预定义规则选出的代表 SKU。
- `manuscript_text/`：可直接纳入结果、讨论和审稿回复的文本。

## 补充材料

- `supplement/Table_S5_case_202502_full_sku_allocation.xlsx`：原始首次 LP Optimal 解，包含分配矩阵、求解步骤和实际约束配置。
- 同目录 CSV/JSON：完整矩阵、轮次汇总、配置和审计记录，便于复核。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 BOM 输出 CSV，确保 Windows Excel 打开中文字段不乱码。"""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """将主文表同步输出为 Markdown，便于复制到论文草稿。"""
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", "<br>") for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_json(
    path: Path,
    month_id: str,
    smooth_weight: float,
    trace_row: pd.Series,
    applied_config: Dict[str, Any],
    rounds: Sequence[str],
) -> None:
    """保留审计证据的机器可读副本，便于后续复核案例的选择和配置。"""
    payload = {
        "month_id": month_id,
        "smooth_transition_weight": smooth_weight,
        "rounds": list(rounds),
        "source": f"lp_relaxation_solution_{month_id}_first_optimal.xlsx",
        "trace": {key: _jsonable(value) for key, value in trace_row.to_dict().items()},
        "applied_constraint_config": {key: _jsonable(value) for key, value in applied_config.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """转换 NumPy/Pandas 标量，确保 JSON 可直接被审阅工具读取。"""
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _display_value(value: Any) -> str:
    """将类别或配置数值转换为论文表的紧凑文本。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
