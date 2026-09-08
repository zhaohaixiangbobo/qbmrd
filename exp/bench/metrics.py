from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstraintCheckResult:
    """
    约束可行性验证摘要。

    Attributes:
        feasible: 是否整体可行
        total_violations: 违反条数汇总
        violations: 后端 ConstraintManager 返回的完整结果（可用于导出 json）
    """

    feasible: bool
    total_violations: int
    violations: Dict[str, Any]


def compute_gnae(allocation_matrix: pd.DataFrame, label_matrix: pd.DataFrame, rounds: List[str]) -> float:
    """
    计算论文定义的 GNAE（Global Normalized Allocation Error）。

    Args:
        allocation_matrix: 算法输出分配矩阵（包含轮次列）
        label_matrix: 历史/真实分配矩阵（包含轮次列）
        rounds: 轮次列名列表

    Returns:
        float: GNAE，lower is better
    """
    a = allocation_matrix.loc[:, rounds].fillna(0.0).astype(float)
    y = label_matrix.loc[:, rounds].fillna(0.0).astype(float)

    d = (a - y).abs().to_numpy().sum()
    t = y.to_numpy().sum()
    if t <= 0:
        return 0.0
    return float(d / (2.0 * t))


def compute_normalized_allocation_l1_distance(
    allocation_matrix: pd.DataFrame,
    baseline_matrix: pd.DataFrame,
    rounds: List[str],
) -> float:
    """
    计算候选分配解相对基线解的归一化 L1 距离。

    该指标将相同 SKU、相同轮次上的绝对分配差异求和，再除以基线总投放量。
    0 表示分配解与基线完全一致，数值越大表示参数变化引起的最优分配变化越大。
    """
    def _round_values(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for round_name in rounds:
            if round_name not in out.columns:
                out[round_name] = 0.0
        if "代码" in out.columns:
            # 用代码对齐，避免输入行顺序变化被误判为解变化。
            out["代码"] = out["代码"].astype(str)
            return out.groupby("代码", dropna=False)[rounds].sum().astype(float)
        return out.loc[:, rounds].reset_index(drop=True).astype(float)

    candidate = _round_values(allocation_matrix)
    baseline = _round_values(baseline_matrix)
    if "代码" in allocation_matrix.columns and "代码" in baseline_matrix.columns:
        all_index = candidate.index.union(baseline.index)
        candidate = candidate.reindex(all_index, fill_value=0.0)
        baseline = baseline.reindex(all_index, fill_value=0.0)
    else:
        n_rows = max(len(candidate), len(baseline))
        candidate = candidate.reindex(range(n_rows), fill_value=0.0)
        baseline = baseline.reindex(range(n_rows), fill_value=0.0)

    distance = float((candidate - baseline).abs().to_numpy().sum())
    baseline_total = float(baseline.abs().to_numpy().sum())
    if baseline_total <= 1e-12:
        return 0.0 if distance <= 1e-12 else float(distance)
    return float(distance / baseline_total)


def compute_objective_value(
    allocation_matrix_full: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
) -> float:
    """
    对给定分配矩阵做“数值版目标函数”评估，口径与后端 LP 目标构造保持一致。

    Args:
        allocation_matrix_full: 需包含产品行数据（至少 需求/可用货源/按需/按价/C/条支比/批发价）与轮次列
        rounds: 轮次列名列表
        existing_allocations: DataLoader.get_existing_allocations() 的返回值
        constraint_config: 约束配置（用于软约束惩罚项，例如按需惩罚/均衡惩罚）
        objective_config: 目标权重配置

    Returns:
        float: 目标函数数值（与 LP 一致为最小化，越小越好）
    """
    df = allocation_matrix_full.copy()
    for r in rounds:
        if r not in df.columns:
            df[r] = 0.0
    df[rounds] = df[rounds].fillna(0.0).astype(float)

    w_max = float(objective_config.get("maximize_allocation_weight", 0.0))
    w_round_balance = float(objective_config.get("round_balance_weight", 0.0))
    w_round_var = float(objective_config.get("round_variance_weight", 0.0))
    w_prod_balance = float(objective_config.get("product_balance_weight", 0.0))
    w_smooth = float(objective_config.get("smooth_transition_weight", 0.0))

    round_totals = df[rounds].sum(axis=0).to_dict()
    round_total_values = [float(round_totals[r]) for r in rounds]

    obj = 0.0

    if w_max > 0:
        total_allocation = float(df[rounds].to_numpy().sum())
        obj -= w_max * total_allocation

    if len(round_total_values) >= 2:
        if w_round_balance > 0:
            obj += w_round_balance * (max(round_total_values) - min(round_total_values))

        if w_round_var > 0:
            avg_round = sum(round_total_values) / len(round_total_values)
            obj += w_round_var * sum(abs(v - avg_round) for v in round_total_values)

        if w_prod_balance > 0:
            obj += _compute_product_balance_term(df, rounds, existing_allocations, w_prod_balance)

        if w_smooth > 0:
            obj += _compute_smooth_transition_term(df, rounds, existing_allocations, w_smooth)

    if constraint_config.get("enable_demand_based_constraints", True):
        obj += _compute_demand_based_penalty(df, rounds)

    if constraint_config.get("enable_balance_constraints", True):
        obj += _compute_balance_soft_penalty(round_total_values)

    return float(obj)


def compute_round_volume_deviation(allocation_matrix_full: pd.DataFrame, rounds: List[str]) -> float:
    df = allocation_matrix_full
    if df is None or df.empty or not rounds:
        return 0.0
    missing = [r for r in rounds if r not in df.columns]
    if missing:
        return 0.0
    totals = df.loc[:, rounds].fillna(0.0).astype(float).sum(axis=0).to_numpy(dtype=float)
    mean = float(np.mean(totals)) if len(totals) > 0 else 0.0
    if mean <= 0:
        return 0.0
    return float((float(np.max(totals)) - float(np.min(totals))) / mean)


def compute_total_stability_metric(allocation_matrix_full: pd.DataFrame, rounds: List[str]) -> float:
    df = allocation_matrix_full
    if df is None or df.empty or len(rounds) < 2:
        return 0.0
    missing = [r for r in rounds if r not in df.columns]
    if missing:
        return 0.0
    totals = df.loc[:, rounds].fillna(0.0).astype(float).sum(axis=0).to_numpy(dtype=float)
    mean = float(np.mean(totals)) if len(totals) > 0 else 0.0
    if mean <= 0:
        return 0.0
    diffs = np.abs(np.diff(totals)).sum()
    denom = mean * float(max(1, len(totals) - 1))
    if denom <= 0:
        return 0.0
    return float(float(diffs) / denom)


def compute_sku_stability_metric(allocation_matrix_full: pd.DataFrame, rounds: List[str]) -> float:
    df = allocation_matrix_full
    if df is None or df.empty or len(rounds) < 2:
        return 0.0
    missing = [r for r in rounds if r not in df.columns]
    if missing:
        return 0.0
    x = df.loc[:, rounds].fillna(0.0).astype(float).to_numpy(dtype=float)
    total_allocated = float(x.sum())
    if total_allocated <= 0:
        return 0.0
    diffs = float(np.abs(np.diff(x, axis=1)).sum())
    return float(diffs / total_allocated)


def compute_sku_concentration_metric(allocation_matrix_full: pd.DataFrame, rounds: List[str]) -> float:
    df = allocation_matrix_full
    if df is None or df.empty or not rounds:
        return 0.0
    missing = [r for r in rounds if r not in df.columns]
    if missing:
        return 0.0
    if "需求" not in df.columns:
        return 0.0
    demand = df["需求"].fillna(0.0).astype(float).to_numpy(dtype=float)
    x = df.loc[:, rounds].fillna(0.0).astype(float).to_numpy(dtype=float)
    out: List[float] = []
    for i in range(len(demand)):
        d = float(demand[i])
        if d <= 0:
            continue
        out.append(float(x[i, :].max()) / d)
    if not out:
        return 0.0
    return float(np.mean(np.array(out, dtype=float)))


def _compute_product_balance_term(
    df: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    weight: float,
) -> float:
    """
    复刻后端 _add_product_level_balance 的数值化版本。
    """
    existing_indices = _existing_indices(existing_allocations)
    demand = df["需求"].fillna(0.0).astype(float).to_numpy()
    alloc = df[rounds].to_numpy(dtype=float)

    total = 0.0
    for idx in range(len(df)):
        if demand[idx] <= 0:
            continue
        if idx in existing_indices:
            continue

        row = alloc[idx, :]
        if len(row) >= 2:
            total += weight * (row.max() - row.min())

            for j, x in enumerate(row):
                total += weight * 3.0 * max(0.0, x - 0.6 * demand[idx])
                total += weight * 5.0 * max(0.0, x - 0.8 * demand[idx])
                total += weight * 10.0 * max(0.0, x - 0.9 * demand[idx])

            if demand[idx] >= 50:
                rounds_used = int((row >= 0.1).sum())
                if demand[idx] >= 100:
                    shortage = max(0.0, 2.0 - float(rounds_used))
                    total += weight * 1.5 * shortage

    return float(total)


def _compute_smooth_transition_term(
    df: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    weight: float,
) -> float:
    """
    复刻后端 _add_smooth_transition 的数值化版本。
    """
    round_totals = df[rounds].sum(axis=0).to_numpy(dtype=float)
    total = 0.0

    for i in range(len(round_totals) - 1):
        total += weight * abs(float(round_totals[i + 1] - round_totals[i]))

    existing_indices = _existing_indices(existing_allocations)
    demand = df["需求"].fillna(0.0).astype(float).to_numpy()
    alloc = df[rounds].to_numpy(dtype=float)

    for idx in range(len(df)):
        if demand[idx] <= 0:
            continue
        if idx in existing_indices:
            continue
        if demand[idx] < 30:
            continue
        row = alloc[idx, :]
        for i in range(len(row) - 1):
            total += weight * 0.17 * abs(float(row[i + 1] - row[i]))

    return float(total)


def _compute_demand_based_penalty(df: pd.DataFrame, rounds: List[str]) -> float:
    """
    复刻后端 create_objective_function 中对 demand_penalty_vars 的惩罚项。
    """
    if "按需" not in df.columns:
        return 0.0

    is_demand_based = df["按需"].fillna("").astype(str).str.contains("需", na=False).to_numpy()
    alloc = df[rounds].to_numpy(dtype=float)

    total = 0.0
    for r_idx in range(2, len(rounds)):
        penalty_weight = 50.0 * (2.0 ** float(r_idx - 2))
        total += float((alloc[is_demand_based, r_idx].sum()) * penalty_weight)
    return float(total)


def _compute_balance_soft_penalty(round_total_values: List[float]) -> float:
    """
    复刻后端 create_objective_function 中对 balance_penalty_vars 的惩罚项。
    """
    balance_penalty_weight = 500.0
    total = 0.0
    for i in range(len(round_total_values) - 1):
        a = float(round_total_values[i])
        b = float(round_total_values[i + 1])
        pos = max(0.0, a - 1.2 * b)
        neg = max(0.0, 0.8 * b - a)
        total += balance_penalty_weight * (pos + neg)
    return float(total)


def _existing_indices(existing_allocations: Dict[str, pd.Series]) -> set[int]:
    """
    识别“已确定分配”的行索引集合，用于与后端一致地跳过部分目标项。
    """
    idx_set: set[int] = set()
    for _, series in existing_allocations.items():
        try:
            fixed = series[series > 0]
            idx_set.update(int(i) for i in fixed.index.tolist())
        except Exception:
            continue
    return idx_set


def validate_constraints(
    data_loader: object,
    allocation_matrix_full: pd.DataFrame,
    constraint_config: Dict[str, Any],
) -> ConstraintCheckResult:
    """
    复用后端 ConstraintManager 对分配矩阵进行可行性验证。

    Args:
        data_loader: 后端 DataLoader 实例
        allocation_matrix_full: 分配矩阵（含轮次列与必要产品字段）
        constraint_config: 约束配置

    Returns:
        ConstraintCheckResult: 约束验证摘要
    """
    from constraint_manager import ConstraintManager

    manager = ConstraintManager(data_loader, constraint_config=constraint_config)
    rounds = data_loader.get_rounds()
    missing = [r for r in rounds if r not in allocation_matrix_full.columns]
    if missing:
        return ConstraintCheckResult(
            feasible=False,
            total_violations=-1,
            violations={"error": "missing_round_columns", "missing": missing},
        )
    allocation_only = allocation_matrix_full.loc[:, rounds].copy()
    for r in rounds:
        allocation_only[r] = pd.to_numeric(allocation_only[r], errors="coerce").fillna(0.0)
    results = manager.validate_all_constraints(allocation_only)
    feasible = bool(results.get("overall_valid", False))
    total_violations = int(results.get("summary", {}).get("total_violations", 0))
    return ConstraintCheckResult(feasible=feasible, total_violations=total_violations, violations=results)
