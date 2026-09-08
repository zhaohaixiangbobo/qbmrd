from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..metrics import compute_objective_value
from .rule_based import (
    _fixed_mask,
    _ipf_enforce_demands,
    _ipf_enforce_round_totals,
    recompute_totals_and_rates,
    solve_with_rule_based,
)


@dataclass(frozen=True)
class SolverOutput:
    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame


def solve_with_ga(
    data_loader: object,
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    max_time_sec: float,
    pop_size: int,
    seed: int = 42,
) -> SolverOutput:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    base = solve_with_rule_based(data_loader, constraint_config, repair_strength=1.0).allocation_matrix_full
    rounds = data_loader.get_rounds()
    existing_allocations = data_loader.get_existing_allocations()
    fixed = _fixed_mask(existing_allocations, len(base), rounds)

    population: List[pd.DataFrame] = []
    population.append(_repair(base.copy(), data_loader, rounds, fixed, constraint_config))

    for _ in range(max(0, int(pop_size) - 1)):
        cand = _mutate_transfer(population[0], rounds, fixed, rng, tries=20)
        cand = _repair(cand, data_loader, rounds, fixed, constraint_config)
        population.append(cand)

    best = population[0]
    best_energy = _energy(best, data_loader, rounds, existing_allocations, constraint_config, objective_config)

    elite = max(1, min(5, int(pop_size * 0.1)))

    while (time.perf_counter() - start) < max_time_sec:
        scored = [
            (df, _energy(df, data_loader, rounds, existing_allocations, constraint_config, objective_config))
            for df in population
        ]
        scored.sort(key=lambda t: t[1])

        if scored[0][1] < best_energy:
            best = scored[0][0]
            best_energy = float(scored[0][1])

        next_pop: List[pd.DataFrame] = [scored[i][0].copy() for i in range(min(elite, len(scored)))]

        while len(next_pop) < pop_size and (time.perf_counter() - start) < max_time_sec:
            p1 = _tournament_select(scored, rng, k=3)
            p2 = _tournament_select(scored, rng, k=3)
            child = _crossover_rows(p1, p2, rounds, rng)
            if float(rng.random()) < 0.7:
                child = _mutate_transfer(child, rounds, fixed, rng, tries=10)
            child = _repair(child, data_loader, rounds, fixed, constraint_config)
            next_pop.append(child)

        population = next_pop

    end = time.perf_counter()
    best = recompute_totals_and_rates(best, rounds)
    obj = compute_objective_value(best, rounds, existing_allocations, constraint_config, objective_config)
    return SolverOutput(
        status="GA",
        solve_time_sec=float(end - start),
        objective_value=float(obj),
        allocation_matrix_full=best,
    )


def _energy(
    df: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
) -> float:
    obj = compute_objective_value(df, rounds, existing_allocations, constraint_config, objective_config)
    penalty = float(_quick_violation_score(df, data_loader, rounds, constraint_config) * 1e6)
    return float(obj + penalty)


def _quick_violation_score(
    df: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    constraint_config: Dict[str, Any],
) -> int:
    x = df.loc[:, rounds].fillna(0.0).astype(float).to_numpy()
    demand = df["需求"].fillna(0.0).astype(float).to_numpy() if "需求" in df.columns else np.zeros(len(df))
    supply = df["可用货源"].fillna(0.0).astype(float).to_numpy() if "可用货源" in df.columns else np.zeros(len(df))

    score = 0

    row_sum = x.sum(axis=1)
    score += int(np.sum(np.abs(row_sum - demand) > 1e-3))

    if "第一轮" in rounds:
        j0 = rounds.index("第一轮")
        cap = np.minimum(demand, supply)
        score += int(np.sum(x[:, j0] - cap > 1e-3))

    tolerance = float(constraint_config.get("volume_tolerance", 0.005))
    for j, r in enumerate(rounds):
        try:
            t = float(data_loader.get_round_constraints(r)["total_quantity"])
        except Exception:
            continue
        col = float(x[:, j].sum())
        if col < t * (1.0 - tolerance) - 1e-3 or col > t * (1.0 + tolerance) + 1e-3:
            score += 1

    if constraint_config.get("enable_price_constraints", True):
        if "批发价" in df.columns and "条支比" in df.columns:
            wholesale = df["批发价"].fillna(0.0).astype(float).to_numpy()
            ratio = df["条支比"].replace(0, np.nan).fillna(200.0).astype(float).to_numpy()
            prices = (wholesale * 50000.0) / ratio
            for j, r in enumerate(rounds):
                total = float(x[:, j].sum())
                if total <= 0:
                    continue
                try:
                    c = data_loader.get_round_constraints(r)
                    upper = float(c["upper_price_limit"])
                    lower = float(c["lower_price_limit"])
                except Exception:
                    continue
                avg = float((x[:, j] * prices).sum() / total)
                if avg < lower - 1e-3 or avg > upper + 1e-3:
                    score += 1

    if constraint_config.get("enable_c_type_constraints", True):
        if "C" in df.columns:
            is_c = df["C"].fillna("").astype(str).str.strip().ne("").to_numpy()
            ratio_limit = float(constraint_config.get("c_type_ratio", 0.4))
            volume_limit = float(constraint_config.get("c_type_volume_limit", 4900))
            for j in range(len(rounds)):
                total = float(x[:, j].sum())
                if total <= 0:
                    continue
                c_total = float(x[is_c, j].sum())
                if c_total > volume_limit + 1e-3 or c_total > ratio_limit * total + 1e-3:
                    score += 1

    if constraint_config.get("enable_price_based_constraints", True):
        if "按价" in df.columns:
            ratio_required = float(constraint_config.get("price_based_ratio", 0.3))
            is_price_based = df["按价"].fillna("").astype(str).str.contains("价", na=False).to_numpy()
            for j in range(len(rounds)):
                active = x[:, j] >= 1.0
                total_cnt = int(active.sum())
                if total_cnt <= 0:
                    continue
                price_cnt = int((active & is_price_based).sum())
                if price_cnt < int(np.ceil(ratio_required * total_cnt)):
                    score += 1

    return int(score)


def _repair(
    df: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    fixed_mask: np.ndarray,
    constraint_config: Dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    for _ in range(15):
        _ipf_enforce_demands(out, rounds, fixed_mask)
        _ipf_enforce_round_totals(data_loader, out, rounds, fixed_mask, constraint_config)
        _cap_first_round_supply(out, rounds, fixed_mask)
    _apply_fixed(out, rounds, fixed_mask, df)
    return out


def _apply_fixed(df: pd.DataFrame, rounds: List[str], fixed_mask: np.ndarray, ref: pd.DataFrame) -> None:
    x = df.loc[:, rounds].to_numpy(dtype=float)
    r = ref.loc[:, rounds].to_numpy(dtype=float)
    x = x * (1.0 - fixed_mask) + r * fixed_mask
    df.loc[:, rounds] = np.maximum(0.0, x)


def _cap_first_round_supply(df: pd.DataFrame, rounds: List[str], fixed_mask: np.ndarray) -> None:
    if "第一轮" not in rounds or "需求" not in df.columns or "可用货源" not in df.columns:
        return
    j0 = rounds.index("第一轮")
    x = df.loc[:, rounds].to_numpy(dtype=float)
    demand = df["需求"].fillna(0.0).astype(float).to_numpy()
    supply = df["可用货源"].fillna(0.0).astype(float).to_numpy()
    cap = np.minimum(demand, supply)

    for i in range(len(df)):
        if fixed_mask[i, j0] > 0:
            continue
        if x[i, j0] <= cap[i]:
            continue
        overflow = float(x[i, j0] - cap[i])
        x[i, j0] = float(cap[i])
        if overflow <= 0:
            continue
        remaining = [j for j in range(len(rounds)) if j != j0 and fixed_mask[i, j] == 0]
        if not remaining:
            continue
        denom = float(x[i, remaining].sum())
        if denom <= 0:
            per = overflow / float(len(remaining))
            for j in remaining:
                x[i, j] += per
        else:
            for j in remaining:
                x[i, j] += overflow * (x[i, j] / denom)

    df.loc[:, rounds] = np.maximum(0.0, x)


def _mutate_transfer(
    df: pd.DataFrame,
    rounds: List[str],
    fixed_mask: np.ndarray,
    rng: np.random.Generator,
    tries: int,
) -> pd.DataFrame:
    out = df.copy()
    if not rounds:
        return out
    x = out.loc[:, rounds].to_numpy(dtype=float)

    demand = out["需求"].fillna(0.0).astype(float).to_numpy() if "需求" in out.columns else np.ones(len(out))
    candidates = np.where(demand > 0)[0]
    if len(candidates) == 0:
        return out

    for _ in range(max(1, int(tries))):
        i = int(rng.choice(candidates))
        if len(rounds) < 2:
            break
        j1, j2 = rng.choice(len(rounds), size=2, replace=False).tolist()
        if fixed_mask[i, j1] > 0 or fixed_mask[i, j2] > 0:
            continue
        if x[i, j1] <= 0:
            continue
        delta = float(min(x[i, j1], max(1.0, 0.05 * float(demand[i]))) * float(rng.random()))
        if delta <= 0:
            continue
        x[i, j1] -= delta
        x[i, j2] += delta
        break

    out.loc[:, rounds] = np.maximum(0.0, x)
    return out


def _tournament_select(
    scored: List[Tuple[pd.DataFrame, float]],
    rng: np.random.Generator,
    k: int,
) -> pd.DataFrame:
    if not scored:
        raise ValueError("empty population")
    k = max(1, min(int(k), len(scored)))
    idx = rng.choice(len(scored), size=k, replace=False).tolist()
    best_i = min(idx, key=lambda i: scored[i][1])
    return scored[best_i][0]


def _crossover_rows(
    a: pd.DataFrame,
    b: pd.DataFrame,
    rounds: List[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = a.copy()
    if a.empty or b.empty or not rounds:
        return out
    mask = rng.random(len(out)) < 0.5
    cols = [c for c in rounds if c in out.columns and c in b.columns]
    if not cols:
        return out
    out.loc[mask, cols] = b.loc[mask, cols].to_numpy()
    return out
