"""
论文结果交付包构建脚本：核验已完成实验，整理可直接引用的表格、数据副本、图件与证据边界说明。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

# 采用无界面后端，确保批处理环境可以生成可复现的论文图件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_DEFAULT = EXP_DIR / "output" / "paper_final_20260808"


def main() -> int:
    """构建论文交付目录，不改写任何原始实验结果。"""
    parser = argparse.ArgumentParser(description="整理已完成 LP 实验的论文图表、汇总表和原始数据副本。")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    for directory in [figures_dir, tables_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    marginal_dir = EXP_DIR / "output" / "lp_marginal_analysis_full"
    relaxation_dir = EXP_DIR / "output" / "lp_relaxation_path_full"
    scalability_dir = EXP_DIR / "output" / "lp_scalability_full"
    fairness_dir = EXP_DIR / "output" / "lp_fairness_example"
    ablation_dir = EXP_DIR / "output" / "lp_ablation_full"
    snapshot_dir = EXP_DIR.parent / "submission_revision_202608" / "results" / "benchmark_20260807T102254"

    marginal = _read_required(marginal_dir / "lp_marginal_aggregate.csv")
    relaxation_trace = _read_required(relaxation_dir / "lp_relaxation_trace.csv")
    relaxation_summary = _read_required(relaxation_dir / "lp_relaxation_path_summary.csv")
    scalability = _read_required(scalability_dir / "lp_scalability_agg.csv")
    fairness = _read_required(fairness_dir / "metrics_table.csv")
    ablation = _read_required(ablation_dir / "lp_ablation_agg.csv")

    table_sensitivity = _make_sensitivity_table(marginal)
    table_relaxation = _make_relaxation_table(relaxation_trace, relaxation_summary)
    table_scalability = _make_scalability_table(scalability)
    table_fairness = _make_fairness_table(fairness)
    table_evidence = _make_evidence_status_table(snapshot_dir)

    _write_table(table_sensitivity, tables_dir / "Table_1_smooth_weight_sensitivity")
    _write_table(table_relaxation, tables_dir / "Table_2_automatic_relaxation_path")
    _write_table(table_scalability, tables_dir / "Table_S1_scalability")
    _write_table(table_fairness, tables_dir / "Table_S2_fairness_numerical_example")
    _write_table(table_evidence, tables_dir / "Table_S3_evidence_status")
    _write_csv(ablation, tables_dir / "Table_S4_stability_ablation_full.csv")

    _copy_existing_figures(marginal_dir, relaxation_dir, figures_dir)
    _plot_scalability(scalability, figures_dir / "Fig_5_scalability")
    _plot_fairness(fairness_dir, fairness, figures_dir / "Fig_6_fairness_numerical_example")

    _copy_data(
        data_dir,
        [
            marginal_dir / "lp_marginal_runs.csv",
            marginal_dir / "lp_marginal_aggregate.csv",
            marginal_dir / "lp_marginal_effects.csv",
            relaxation_dir / "lp_relaxation_trace.csv",
            relaxation_dir / "lp_relaxation_path_summary.csv",
            relaxation_dir / "lp_relaxation_path_report.md",
            scalability_dir / "lp_scalability_summary.csv",
            scalability_dir / "lp_scalability_agg.csv",
            fairness_dir / "metrics_table.csv",
            ablation_dir / "lp_ablation_summary.csv",
            ablation_dir / "lp_ablation_agg.csv",
            ablation_dir / "lp_ablation_sensitivity.csv",
        ],
    )
    _copy_snapshot_data(snapshot_dir, data_dir)
    _write_readme(output_dir, table_sensitivity, table_relaxation, table_evidence)
    print(f"paper_artifacts={output_dir}")
    return 0


def _read_required(path: Path) -> pd.DataFrame:
    """读取必要 CSV，缺失时显式失败，避免产出基于不完整结果的论文材料。"""
    if not path.exists():
        raise FileNotFoundError(f"missing required experiment artifact: {path}")
    return pd.read_csv(path)


def _make_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """将完整边际分析压缩成主文可直接放置的平滑权重敏感性表。"""
    out = pd.DataFrame(
        {
            "平滑权重": df["smooth_transition_weight"].map(_format_number),
            "月份数": df["n_months"].astype(int),
            "GNAE（均值±SD）": _mean_sd(df, "gnae"),
            "解变化 L1（均值±SD）": _mean_sd(df, "solution_l1_to_baseline"),
            "SKU 波动（均值±SD）": _mean_sd(df, "stability_sku"),
            "总量波动（均值±SD）": _mean_sd(df, "stability_total"),
            "后验可行率": df["feasible_rate"].map(lambda value: f"{100.0 * float(value):.1f}%"),
            "违背数（均值±SD）": _mean_sd(df, "total_violations"),
            "求解时间/s（均值±SD）": _mean_sd(df, "solve_time_sec"),
        }
    )
    return out


def _make_relaxation_table(trace: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """汇总每月实际走过的松弛动作，并保留最终后验校验状态。"""
    rows: List[Dict[str, object]] = []
    for _, item in summary.sort_values("month_id").iterrows():
        month_id = str(item["month_id"])
        month_trace = trace[trace["month_id"].astype(str) == month_id].sort_values("step_index")
        actions = [str(value) for value in month_trace["action_label"].tolist() if str(value) != "初始约束"]
        final_row = month_trace.iloc[-1] if not month_trace.empty else None
        rows.append(
            {
                "月份": month_id,
                "LP 尝试次数": int(item["lp_attempt_count"]),
                "首次 Optimal 步骤": int(item["first_optimal_step"]) if pd.notna(item["first_optimal_step"]) else "无",
                "实际自动松弛动作": " → ".join(actions) if actions else "无（初始约束即 Optimal）",
                "最终 LP 状态": item["final_status"],
                "后验校验可行": "" if final_row is None or pd.isna(final_row["feasible"]) else bool(final_row["feasible"]),
                "后验违背数": "" if final_row is None or pd.isna(final_row["total_violations"]) else int(final_row["total_violations"]),
                "最终耗时/s": f"{float(item['final_solve_time_sec']):.3f}",
            }
        )
    return pd.DataFrame(rows)


def _make_scalability_table(df: pd.DataFrame) -> pd.DataFrame:
    """整理规模扩展实验，为补充材料提供规模—时间—误差证据。"""
    return pd.DataFrame(
        {
            "场景": df["scale"],
            "SKU 数（均值±SD）": _mean_sd(df, "number_of_skus", digits=1),
            "轮次数": df["rounds_mean"].map(_format_number),
            "需求规模倍数": df["demand_scale_mean"].map(_format_number),
            "GNAE（均值±SD）": _mean_sd(df, "gnae"),
            "规则违背数（均值±SD）": _mean_sd(df, "rule_deviations"),
            "求解时间/s（均值±SD）": _mean_sd(df, "solve_time_sec"),
        }
    )


def _make_fairness_table(df: pd.DataFrame) -> pd.DataFrame:
    """保留公平性数值例子的原始指标及其中文解释。"""
    out = df.copy()
    out.columns = ["指标", "无公平项", "有公平项", "解释"]
    return out


def _make_evidence_status_table(snapshot_dir: Path) -> pd.DataFrame:
    """说明哪些实验可用于主文，避免把空目录或后验不可行结果误作完整比较证据。"""
    rows: List[Dict[str, str]] = [
        {
            "实验": "平滑权重边际分析",
            "数据状态": "完整：6 月 × 6 权重 = 36 条",
            "建议用途": "主文敏感性与解变化分析",
            "关键限制": "六个权重点后验可行率均为 0%，不可作为约束满足证据。",
        },
        {
            "实验": "自动松弛路径",
            "数据状态": "完整：6 月、13 个步骤",
            "建议用途": "方法机制图与附录过程说明",
            "关键限制": "6 月仅 1 月后验可行；应同时报告 solver status 与后验校验。",
        },
        {
            "实验": "LP 规模扩展",
            "数据状态": "完整：24 条 synthetic runs",
            "建议用途": "补充材料的可扩展性证据",
            "关键限制": "使用合成规模场景，不替代真实数据性能比较。",
        },
        {
            "实验": "公平性数值例子",
            "数据状态": "完整：8 SKU toy example",
            "建议用途": "方法直观示例或附录",
            "关键限制": "玩具实例，不应作为真实业务性能主证据。",
        },
    ]
    current_multiseed_dirs = [
        EXP_DIR / "output" / "benchmark_multiseed_10seed_sa300_ga600",
        EXP_DIR / "output" / "benchmark_multiseed_30seed",
    ]
    has_current_multiseed = any(any(directory.glob("benchmark_multiseed_*.csv")) for directory in current_multiseed_dirs)
    rows.append(
        {
            "实验": "当前多种子 benchmark",
            "数据状态": "已找到完整 CSV" if has_current_multiseed else "两个输出目录为空",
            "建议用途": "主文算法对比与显著性分析" if has_current_multiseed else "暂不纳入主文汇总表",
            "关键限制": "缺少当前脚本生成的 runs/summary/significance CSV，不能声称完成当前版本的多种子显著性分析。",
        }
    )
    snapshot_summary = snapshot_dir / "benchmark_month_method_summary.csv"
    rows.append(
        {
            "实验": "20260807 benchmark 快照",
            "数据状态": "存在 24 条月×方法汇总和 372 条原始记录" if snapshot_summary.exists() else "未找到",
            "建议用途": "作为历史复现数据副本",
            "关键限制": "与当前实验脚本状态不同，且快照中 LP 存在 Infeasible 记录；不与当前边际实验混合计算。",
        }
    )
    return pd.DataFrame(rows)


def _mean_sd(df: pd.DataFrame, prefix: str, digits: int = 4) -> pd.Series:
    """把数值均值和标准差格式化为论文表通用的 mean ± SD。"""
    means = pd.to_numeric(df[f"{prefix}_mean"], errors="coerce")
    stds = pd.to_numeric(df[f"{prefix}_std"], errors="coerce")
    return pd.Series([f"{mean:.{digits}f} ± {std:.{digits}f}" if pd.notna(mean) and pd.notna(std) else "" for mean, std in zip(means, stds)])


def _format_number(value: float) -> str:
    """格式化整数与小数，避免论文表中出现冗余 .0。"""
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.3f}"


def _write_table(df: pd.DataFrame, path_without_suffix: Path) -> None:
    """同步写出 UTF-8 CSV 与无需额外依赖的 Markdown 表格。"""
    _write_csv(df, path_without_suffix.with_suffix(".csv"))
    columns = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", "<br>") for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    path_without_suffix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    """统一以 UTF-8 BOM 写 CSV，保证 Windows Excel 直接打开中文列名正常。"""
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _copy_existing_figures(marginal_dir: Path, relaxation_dir: Path, figures_dir: Path) -> None:
    """复制已验证的全量图件，保留 PNG 与 PDF 两种论文交付格式。"""
    mapping = {
        marginal_dir / "lp_marginal_overall_trends": "Fig_1_smooth_weight_overall_trends",
        marginal_dir / "lp_marginal_solution_change_by_month": "Fig_2_solution_change_by_month",
        marginal_dir / "lp_marginal_stability_fit_tradeoff": "Fig_3_stability_fit_tradeoff",
        relaxation_dir / "lp_relaxation_path": "Fig_4_automatic_relaxation_path",
    }
    for source_base, target_stem in mapping.items():
        for suffix in [".png", ".pdf"]:
            source = source_base.with_suffix(suffix)
            if source.exists():
                shutil.copy2(source, figures_dir / f"{target_stem}{suffix}")


def _plot_scalability(df: pd.DataFrame, path_without_suffix: Path) -> None:
    """生成规模扩展的双面板图：实例规模—耗时与实例规模—GNAE。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = pd.to_numeric(df["number_of_skus_mean"], errors="coerce")
    axes[0].errorbar(x, df["solve_time_sec_mean"], yerr=df["solve_time_sec_std"], marker="o", linewidth=2, capsize=4, color="#0072B2")
    axes[0].set_xlabel("Number of SKUs")
    axes[0].set_ylabel("Solve time (s)")
    axes[0].set_title("Solve time under scalability scenarios")
    axes[0].grid(True, alpha=0.3)
    axes[1].errorbar(x, df["gnae_mean"], yerr=df["gnae_std"], marker="o", linewidth=2, capsize=4, color="#D55E00")
    axes[1].set_xlabel("Number of SKUs")
    axes[1].set_ylabel("GNAE")
    axes[1].set_title("Historical-plan deviation under scalability scenarios")
    axes[1].grid(True, alpha=0.3)
    for axis, labels in zip(axes, [df["scale"], df["scale"]]):
        for x_value, y_value, label in zip(x, axis.lines[0].get_ydata(), labels):
            axis.annotate(str(label).split(" ")[0], (x_value, y_value), xytext=(4, 5), textcoords="offset points", fontsize=8)
    _save_figure(fig, path_without_suffix)


def _plot_fairness(fairness_dir: Path, metrics: pd.DataFrame, path_without_suffix: Path) -> None:
    """将公平性数值例子整理为轮次总量对比与三项平滑指标对比图。"""
    without = _read_required(fairness_dir / "allocation_model1.csv")
    with_fairness = _read_required(fairness_dir / "allocation_model2.csv")
    rounds = [column for column in ["第一轮", "第二轮", "第三轮"] if column in without.columns and column in with_fairness.columns]
    round_labels = [f"Round {index + 1}" for index in range(len(rounds))]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = list(range(len(rounds)))
    width = 0.36
    axes[0].bar([value - width / 2 for value in x], [float(without[round_name].sum()) for round_name in rounds], width, label="Without fairness", color="#999999")
    axes[0].bar([value + width / 2 for value in x], [float(with_fairness[round_name].sum()) for round_name in rounds], width, label="With fairness", color="#009E73")
    axes[0].set_xticks(x, round_labels)
    axes[0].set_ylabel("Total allocation volume")
    axes[0].set_title("Fairness term balances total allocation across rounds")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)
    selected = metrics.iloc[:3].copy()
    x2 = list(range(len(selected)))
    axes[1].bar([value - width / 2 for value in x2], selected["Without fairness"], width, label="Without fairness", color="#999999")
    axes[1].bar([value + width / 2 for value in x2], selected["With fairness"], width, label="With fairness", color="#009E73")
    axes[1].set_xticks(x2, ["Max round deviation", "Adjacent-round change", "SKU deviation"], rotation=15)
    axes[1].set_ylabel("Metric value (lower is better)")
    axes[1].set_title("Smoothness metrics in the fairness example")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    _save_figure(fig, path_without_suffix)


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    """统一导出 300 DPI PNG 与可编辑 PDF，并释放绘图资源。"""
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _copy_data(data_dir: Path, paths: Iterable[Path]) -> None:
    """复制当前实验的原始表，保留论文汇总表可追溯的数据来源。"""
    for path in paths:
        if path.exists():
            shutil.copy2(path, data_dir / path.name)


def _copy_snapshot_data(snapshot_dir: Path, data_dir: Path) -> None:
    """复制历史 benchmark 快照，但不把它与当前实验结果混合计算。"""
    for name in ["benchmark_month_method_summary.csv", "benchmark_raw_runs.csv", "rule_audit_details.csv"]:
        source = snapshot_dir / name
        if source.exists():
            shutil.copy2(source, data_dir / f"snapshot_20260807_{name}")


def _write_readme(output_dir: Path, sensitivity: pd.DataFrame, relaxation: pd.DataFrame, evidence: pd.DataFrame) -> None:
    """写出交付索引与必要的证据边界，便于论文作者避免过度解释。"""
    direct_optimal = int((relaxation["实际自动松弛动作"] == "无（初始约束即 Optimal）").sum())
    postcheck_valid = int((relaxation["后验校验可行"] == True).sum())
    zero_row = sensitivity[sensitivity["平滑权重"] == "0"].iloc[0]
    baseline_row = sensitivity[sensitivity["平滑权重"] == "300"].iloc[0]
    lines = [
        "# 论文实验结果交付包",
        "",
        "## 主文建议使用",
        "",
        "- 图 1–3：平滑权重敏感性、解变化和稳定性—拟合权衡。",
        "- 图 4 与表 2：自动松弛路径；用于说明实际回退机制，而非全局最小松弛代价。",
        "- 表 1：六个月边际分析的权重敏感性汇总。",
        "",
        "## 关键观察",
        "",
        f"- 权重 0 的 GNAE 为 {zero_row['GNAE（均值±SD）']}，而基线 300 为 {baseline_row['GNAE（均值±SD）']}；无平滑项会显著改变解并恶化历史计划拟合。",
        f"- {direct_optimal}/6 个月在有效初始约束下直接获得 Optimal；{postcheck_valid}/6 个月通过独立后验约束校验。",
        "- 因边际分析各权重点的后验可行率均为 0%，这些结果只应用于敏感性和机制解释，不能作为“全部约束满足”的证据。",
        "",
        "## 目录说明",
        "",
        "- `figures/`：可直接插入论文的 PNG（300 DPI）和 PDF（矢量）图件。",
        "- `tables/`：最终汇总表，均提供 CSV 与 Markdown；S4 为原稳定性消融全表。",
        "- `data/`：构建汇总表所依据的原始实验数据副本。",
        "",
        "## 多种子 benchmark 状态",
        "",
        "当前 `benchmark_multiseed_10seed_sa300_ga600` 与 `benchmark_multiseed_30seed` 目录没有 CSV 结果，因此本交付包不生成当前版本的显著性主表。历史 20260807 benchmark 快照仅作为 `data/` 副本保留，不与当前实验混合。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
