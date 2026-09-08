from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..bootstrap import ensure_backend_on_path


@dataclass(frozen=True)
class SolverOutput:
    """
    求解器统一输出结构。
    """

    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame


def solve_with_rule_based(
    data_loader: object,
    constraint_config: Dict[str, Any],
    repair_strength: float = 0.1,
) -> SolverOutput:
    """
    Rule-based heuristic：按比例拆分 + sequential repair，作为对照基线。
    """
    start = time.perf_counter()

    s = float(repair_strength)
    if not np.isfinite(s):
        s = 0.1
    s = float(min(1.0, max(0.0, s)))

    product_data = data_loader.get_product_data().copy()
    rounds = data_loader.get_rounds()
    existing_allocations = data_loader.get_existing_allocations()

    allocation = _init_allocation_matrix(product_data, rounds, existing_allocations)
    _initial_proportional_split(data_loader, allocation, rounds, existing_allocations, repair_strength=s)
    _sequential_repair(
        data_loader,
        allocation,
        rounds,
        existing_allocations,
        constraint_config,
        repair_strength=s,
    )
    allocation = recompute_totals_and_rates(allocation, rounds)

    end = time.perf_counter()
    return SolverOutput(
        status="HEURISTIC",
        solve_time_sec=float(end - start),
        objective_value=None,
        allocation_matrix_full=allocation,
    )


def recompute_totals_and_rates(allocation: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    df = allocation.copy()
    for r in rounds:
        if r not in df.columns:
            df[r] = 0.0
    df.loc[:, rounds] = df.loc[:, rounds].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df["总分配量"] = df.loc[:, rounds].sum(axis=1)
    if "需求" in df.columns:
        demand = pd.to_numeric(df["需求"], errors="coerce").fillna(0.0)
        df["分配率"] = np.where(demand > 0, df["总分配量"] / demand, 1.0)
    return df


def _init_allocation_matrix(
    product_data: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    构造分配矩阵，并写入既有固定投放。
    """
    df = product_data.copy()
    for r in rounds:
        if r not in df.columns:
            df[r] = 0.0

    for round_name, series in existing_allocations.items():
        if round_name not in df.columns:
            continue
        for idx, v in series.items():
            try:
                df.at[int(idx), round_name] = float(v)
            except Exception:
                continue
    return df


def _initial_proportional_split(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    repair_strength: float,
) -> None:
    """
    初始解：先按 SKU 剩余需求，再按轮次目标量比例拆分。
    """
    demand = allocation["需求"].fillna(0.0).astype(float).to_numpy()
    supply = allocation["可用货源"].fillna(0.0).astype(float).to_numpy()

    fixed_mask = _fixed_mask(existing_allocations, len(allocation), rounds)
    fixed_sum_by_row = (allocation.loc[:, rounds].to_numpy(dtype=float) * fixed_mask).sum(axis=1)
    remaining_demand = np.maximum(0.0, demand - fixed_sum_by_row)

    targets = []
    for r in rounds:
        try:
            c = data_loader.get_round_constraints(r)
            targets.append(float(c["total_quantity"]))
        except Exception:
            targets.append(0.0)
    targets = np.array(targets, dtype=float)

    fixed_sum_by_col = (allocation.loc[:, rounds].to_numpy(dtype=float) * fixed_mask).sum(axis=0)
    remaining_targets = np.maximum(0.0, targets - fixed_sum_by_col)
    target_total = float(remaining_targets.sum())
    if target_total <= 0:
        shares = np.zeros(len(rounds), dtype=float)
    else:
        shares = remaining_targets / target_total

    decay = np.array([0.5**j for j in range(len(rounds))], dtype=float)
    decay_sum = float(decay.sum())
    if decay_sum <= 0:
        decay = np.ones(len(rounds), dtype=float) / max(1.0, float(len(rounds)))
    else:
        decay = decay / decay_sum

    s = float(repair_strength)
    if not np.isfinite(s):
        s = 0.25
    s = float(min(1.0, max(0.0, s)))
    weights = s * shares + (1.0 - s) * decay
    w_sum = float(weights.sum())
    if w_sum <= 0:
        weights = np.ones(len(rounds), dtype=float) / max(1.0, float(len(rounds)))
    else:
        weights = weights / w_sum
    alloc_matrix = allocation.loc[:, rounds].to_numpy(dtype=float)

    for i in range(len(allocation)):
        if remaining_demand[i] <= 0:
            continue
        for j, r in enumerate(rounds):
            if fixed_mask[i, j] > 0:
                continue
            alloc_matrix[i, j] = float(remaining_demand[i] * weights[j])

    allocation.loc[:, rounds] = alloc_matrix

    if "第一轮" in rounds and float(repair_strength) >= 0.7:
        _cap_first_round_supply(allocation, rounds, supply)


def _sequential_repair(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    constraint_config: Dict[str, Any],
    repair_strength: float,
) -> None:
    """
    依优先级做迭代修复：需求/固定/货源/总量 → 均价 → C类/按价/按需。
    """
    fixed_mask = _fixed_mask(existing_allocations, len(allocation), rounds)

    s = float(repair_strength)
    if not np.isfinite(s):
        s = 0.25
    s = float(min(1.0, max(0.0, s)))

    ipf_iters = int(round(30.0 * s))
    if ipf_iters <= 0:
        ipf_iters = 1

    for _ in range(ipf_iters):
        _ipf_enforce_demands(allocation, rounds, fixed_mask)
        if s >= 0.7:
            _ipf_enforce_round_totals(data_loader, allocation, rounds, fixed_mask, constraint_config)
        if s >= 0.7:
            _cap_first_round_supply(
                allocation,
                rounds,
                allocation["可用货源"].fillna(0.0).astype(float).to_numpy(),
            )

    if s >= 0.95 and constraint_config.get("enable_price_constraints", True):
        _repair_average_price(data_loader, allocation, rounds, fixed_mask, max_steps=200)

    if s >= 0.95 and constraint_config.get("enable_c_type_constraints", True):
        _repair_c_type(data_loader, allocation, rounds, fixed_mask, constraint_config, max_steps=200)

    if s >= 0.95 and constraint_config.get("enable_price_based_constraints", True):
        _repair_price_based_ratio(data_loader, allocation, rounds, fixed_mask, constraint_config, max_steps=200)


def _fixed_mask(existing_allocations: Dict[str, pd.Series], n_rows: int, rounds: List[str]) -> np.ndarray:
    """
    构造 fixed_mask[i,j]∈{0,1}，仅对 fixed_value>0 视为固定投放。
    """
    m = np.zeros((n_rows, len(rounds)), dtype=float)
    round_to_j = {r: j for j, r in enumerate(rounds)}
    for round_name, series in existing_allocations.items():
        if round_name not in round_to_j:
            continue
        j = round_to_j[round_name]
        for idx, v in series.items():
            try:
                if float(v) > 0:
                    m[int(idx), j] = 1.0
            except Exception:
                continue
    return m


def _ipf_enforce_demands(allocation: pd.DataFrame, rounds: List[str], fixed_mask: np.ndarray) -> None:
    """
    调整每个 SKU 的行和，使其满足需求（固定投放不动）。
    """
    demand = allocation["需求"].fillna(0.0).astype(float).to_numpy()
    x = allocation.loc[:, rounds].to_numpy(dtype=float)

    for i in range(len(allocation)):
        fixed_sum = float((x[i, :] * fixed_mask[i, :]).sum())
        target = max(0.0, float(demand[i]) - fixed_sum)
        adjustable = 1.0 - fixed_mask[i, :]
        adjustable_sum = float((x[i, :] * adjustable).sum())
        if adjustable_sum <= 0:
            continue
        factor = target / adjustable_sum if adjustable_sum > 0 else 1.0
        x[i, :] = x[i, :] * fixed_mask[i, :] + (x[i, :] * adjustable) * factor

    allocation.loc[:, rounds] = np.maximum(0.0, x)


def _ipf_enforce_round_totals(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    fixed_mask: np.ndarray,
    constraint_config: Dict[str, Any],
) -> None:
    """
    调整每轮的列和，使其贴近轮次目标量（固定投放不动）。
    """
    tolerance = float(constraint_config.get("volume_tolerance", 0.005))
    x = allocation.loc[:, rounds].to_numpy(dtype=float)
    for j, r in enumerate(rounds):
        try:
            target = float(data_loader.get_round_constraints(r)["total_quantity"])
        except Exception:
            continue
        fixed_sum = float((x[:, j] * fixed_mask[:, j]).sum())
        min_target = target * (1.0 - tolerance)
        max_target = target * (1.0 + tolerance)
        current = float(x[:, j].sum())
        clipped_target = min(max(current, min_target), max_target)
        adjustable = 1.0 - fixed_mask[:, j]
        adjustable_sum = float((x[:, j] * adjustable).sum())
        if adjustable_sum <= 0:
            continue
        factor = max(0.0, clipped_target - fixed_sum) / adjustable_sum
        x[:, j] = x[:, j] * fixed_mask[:, j] + (x[:, j] * adjustable) * factor

    allocation.loc[:, rounds] = np.maximum(0.0, x)


def _cap_first_round_supply(allocation: pd.DataFrame, rounds: List[str], supply: np.ndarray) -> None:
    """
    第一轮货源上限：第一轮分配量不超过 min(需求, 可用货源)。
    """
    if "第一轮" not in rounds:
        return
    j0 = rounds.index("第一轮")
    demand = allocation["需求"].fillna(0.0).astype(float).to_numpy()
    x = allocation.loc[:, rounds].to_numpy(dtype=float)

    for i in range(len(allocation)):
        cap = min(float(demand[i]), float(supply[i]))
        if x[i, j0] <= cap:
            continue
        overflow = float(x[i, j0] - cap)
        x[i, j0] = cap
        if overflow <= 0:
            continue
        remaining_rounds = [j for j in range(len(rounds)) if j != j0]
        denom = float(x[i, remaining_rounds].sum())
        if denom <= 0:
            per = overflow / float(len(remaining_rounds))
            for j in remaining_rounds:
                x[i, j] += per
        else:
            for j in remaining_rounds:
                x[i, j] += overflow * (x[i, j] / denom)

    allocation.loc[:, rounds] = np.maximum(0.0, x)


def _unit_price_per_case(allocation: pd.DataFrame) -> np.ndarray:
    """
    与后端一致：单箱价格 = 批发价 * 50000 / 条支比。
    """
    wholesale = allocation["批发价"].fillna(0.0).astype(float).to_numpy()
    ratio = allocation["条支比"].replace(0, np.nan).fillna(200.0).astype(float).to_numpy()
    return (wholesale * 50000.0) / ratio


def _repair_average_price(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    fixed_mask: np.ndarray,
    max_steps: int,
) -> None:
    """
    通过跨轮 swap 的方式粗略修复均价区间约束。
    """
    prices = _unit_price_per_case(allocation)
    x = allocation.loc[:, rounds].to_numpy(dtype=float)

    for _ in range(max_steps):
        changed = False
        for j, r in enumerate(rounds):
            try:
                c = data_loader.get_round_constraints(r)
                upper = float(c["upper_price_limit"])
                lower = float(c["lower_price_limit"])
            except Exception:
                continue

            total = float(x[:, j].sum())
            if total <= 0:
                continue
            avg = float((x[:, j] * prices).sum() / total)

            if lower <= avg <= upper:
                continue

            if avg > upper:
                i_high = int(np.argmax(np.where(x[:, j] > 0, prices, -np.inf)))
                i_low = int(np.argmin(np.where(x[:, j] > 0, prices, np.inf)))
            else:
                i_high = int(np.argmax(np.where(x[:, j] > 0, prices, -np.inf)))
                i_low = int(np.argmin(np.where(x[:, j] > 0, prices, np.inf)))

            if not np.isfinite(prices[i_high]) or not np.isfinite(prices[i_low]):
                continue
            if i_high == i_low:
                continue

            target_avg = (upper + lower) / 2.0
            k = None
            best_diff = float("inf")
            for jj in range(len(rounds)):
                if jj == j:
                    continue
                t_jj = float(x[:, jj].sum())
                if t_jj <= 0:
                    continue
                avg_jj = float((x[:, jj] * prices).sum() / t_jj)
                diff = abs(avg_jj - target_avg)
                if diff < best_diff:
                    best_diff = diff
                    k = jj
            if k is None:
                continue

            delta = min(float(x[i_high, j]), float(x[i_low, k]))
            if delta <= 0:
                continue

            delta = max(1.0, min(delta, 20.0))
            if fixed_mask[i_high, j] > 0 or fixed_mask[i_low, k] > 0:
                continue
            x[i_high, j] -= delta
            x[i_high, k] += delta
            x[i_low, k] -= delta
            x[i_low, j] += delta
            changed = True

        if not changed:
            break

    allocation.loc[:, rounds] = np.maximum(0.0, x)


def _repair_c_type(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    fixed_mask: np.ndarray,
    constraint_config: Dict[str, Any],
    max_steps: int,
) -> None:
    """
    粗略修复 C 类比例/上限约束：将 C 类量从超标轮向其他轮转移。
    """
    if "C" not in allocation.columns:
        return
    is_c = allocation["C"].fillna("").astype(str).str.strip().ne("").to_numpy()
    ratio_limit = float(constraint_config.get("c_type_ratio", 0.4))
    volume_limit = float(constraint_config.get("c_type_volume_limit", 4900))

    x = allocation.loc[:, rounds].to_numpy(dtype=float)
    for _ in range(max_steps):
        changed = False
        for j, r in enumerate(rounds):
            total = float(x[:, j].sum())
            if total <= 0:
                continue
            c_total = float(x[is_c, j].sum())
            if c_total <= volume_limit and c_total <= ratio_limit * total:
                continue

            over = max(0.0, c_total - min(volume_limit, ratio_limit * total))
            if over <= 0:
                continue

            i_c = int(np.argmax(np.where(is_c & (x[:, j] > 0) & (fixed_mask[:, j] == 0), x[:, j], -np.inf)))
            if not np.isfinite(x[i_c, j]) or x[i_c, j] <= 0:
                continue

            for k in range(len(rounds)):
                if k == j:
                    continue
                delta = min(float(x[i_c, j]), float(over))
                delta = max(1.0, min(delta, 50.0))
                if fixed_mask[i_c, k] > 0:
                    continue
                x[i_c, j] -= delta
                x[i_c, k] += delta
                changed = True
                break

        if not changed:
            break

    allocation.loc[:, rounds] = np.maximum(0.0, x)


def _repair_price_based_ratio(
    data_loader: object,
    allocation: pd.DataFrame,
    rounds: List[str],
    fixed_mask: np.ndarray,
    constraint_config: Dict[str, Any],
    max_steps: int,
) -> None:
    """
    粗略修复按价品规占比：若某轮按价投放品规数不足，尝试给按价品规补少量投放。
    """
    if "按价" not in allocation.columns:
        return
    ratio_required = float(constraint_config.get("price_based_ratio", 0.3))
    is_price_based = allocation["按价"].fillna("").astype(str).str.contains("价", na=False).to_numpy()

    x = allocation.loc[:, rounds].to_numpy(dtype=float)
    for _ in range(max_steps):
        changed = False
        for j, _ in enumerate(rounds):
            active = x[:, j] >= 1.0
            total_cnt = int(active.sum())
            if total_cnt <= 0:
                continue
            price_cnt = int((active & is_price_based).sum())
            if price_cnt >= int(np.ceil(ratio_required * total_cnt)):
                continue

            candidates = np.where(is_price_based & (x[:, j] < 1.0) & (fixed_mask[:, j] == 0) & (allocation["需求"].fillna(0.0).astype(float).to_numpy() > 0))[0]
            if len(candidates) == 0:
                continue
            i = int(candidates[0])
            x[i, j] = max(x[i, j], 1.0)
            changed = True

        if not changed:
            break

    allocation.loc[:, rounds] = np.maximum(0.0, x)
