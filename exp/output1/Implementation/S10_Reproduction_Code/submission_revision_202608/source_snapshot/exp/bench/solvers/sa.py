from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..metrics import compute_objective_value
from .rule_based import recompute_totals_and_rates, solve_with_rule_based


@dataclass(frozen=True)
class SolverOutput:
    """Common solver-output structure."""

    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame


def solve_with_simulated_annealing(
    data_loader: object,
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    max_time_sec: float = 8.0,
    max_iters: int = 800,
    seed: int = 42,
) -> SolverOutput:
    """Simulated-annealing baseline: neighborhood search initialized by the rule-based solution."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    init = solve_with_rule_based(data_loader, constraint_config, repair_strength=1.0).allocation_matrix_full
    rounds = data_loader.get_rounds()
    existing_allocations = data_loader.get_existing_allocations()

    current = init.copy()
    current_energy = _energy(current, data_loader, rounds, existing_allocations, constraint_config, objective_config)
    best = current.copy()
    best_energy = current_energy

    t0 = max(1.0, abs(current_energy) * 0.01)
    temperature = float(t0)
    alpha = 0.995

    i = 0
    while i < max_iters and (time.perf_counter() - start) < max_time_sec:
        cand = _neighbor(current, data_loader, rounds, existing_allocations, rng)
        cand_energy = _energy(cand, data_loader, rounds, existing_allocations, constraint_config, objective_config)

        accept = False
        if cand_energy <= current_energy:
            accept = True
        else:
            p = math.exp(-(cand_energy - current_energy) / max(1e-9, temperature))
            accept = float(rng.random()) < p

        if accept:
            current = cand
            current_energy = cand_energy
            if cand_energy < best_energy:
                best = cand
                best_energy = cand_energy

        temperature *= alpha
        i += 1

    end = time.perf_counter()
    best = recompute_totals_and_rates(best, rounds)
    obj = compute_objective_value(best, rounds, existing_allocations, constraint_config, objective_config)
    return SolverOutput(
        status="SA",
        solve_time_sec=float(end - start),
        objective_value=float(obj),
        allocation_matrix_full=best,
    )


def _neighbor(
    current: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Transfer volume between two rounds of one SKU while preserving its total."""
    df = current.copy()
    x = df.loc[:, rounds].to_numpy(dtype=float)
    demand = df['Demand'].fillna(0.0).astype(float).to_numpy()
    supply = df['Available_Supply'].fillna(0.0).astype(float).to_numpy()
    fixed_mask = _fixed_mask(existing_allocations, len(df), rounds)

    candidates = np.where(demand > 0)[0]
    if len(candidates) == 0:
        return df

    for _ in range(30):
        i = int(rng.choice(candidates))
        j1, j2 = rng.choice(len(rounds), size=2, replace=False).tolist()
        if fixed_mask[i, j1] > 0 or fixed_mask[i, j2] > 0:
            continue
        if x[i, j1] <= 0:
            continue
        delta = float(min(x[i, j1], max(1.0, 0.05 * demand[i])) * rng.random())
        if delta <= 0:
            continue

        if rounds[j2] == 'Round_1':
            cap = min(float(demand[i]), float(supply[i]))
            if x[i, j2] + delta > cap:
                delta = max(0.0, cap - x[i, j2])
        if rounds[j1] == 'Round_1':
            cap = min(float(demand[i]), float(supply[i]))
            if x[i, j1] - delta > cap:
                delta = min(delta, max(0.0, x[i, j1] - cap))

        if delta <= 0:
            continue
        x[i, j1] -= delta
        x[i, j2] += delta
        break

    df.loc[:, rounds] = np.maximum(0.0, x)
    return df


def _energy(
    df: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    existing_allocations: Dict[str, pd.Series],
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
) -> float:
    """Annealing energy: objective value plus a fast feasibility penalty; lower is better."""
    obj = compute_objective_value(df, rounds, existing_allocations, constraint_config, objective_config)
    penalty = float(_quick_violation_score(df, data_loader, rounds, constraint_config) * 1e6)
    return float(obj + penalty)


def _quick_violation_score(
    df: pd.DataFrame,
    data_loader: object,
    rounds: List[str],
    constraint_config: Dict[str, Any],
) -> int:
    """Approximate violation counts to guide SA search quickly."""
    x = df.loc[:, rounds].fillna(0.0).astype(float).to_numpy()
    demand = df['Demand'].fillna(0.0).astype(float).to_numpy()
    supply = df['Available_Supply'].fillna(0.0).astype(float).to_numpy()

    score = 0

    row_sum = x.sum(axis=1)
    score += int(np.sum(np.abs(row_sum - demand) > 1e-3))

    if 'Round_1' in rounds:
        j0 = rounds.index('Round_1')
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
        if 'Wholesale_Price' in df.columns and 'Carton_Stick_Ratio' in df.columns:
            wholesale = df['Wholesale_Price'].fillna(0.0).astype(float).to_numpy()
            ratio = df['Carton_Stick_Ratio'].replace(0, np.nan).fillna(200.0).astype(float).to_numpy()
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

    return int(score)


def _fixed_mask(existing_allocations: Dict[str, pd.Series], n_rows: int, rounds: List[str]) -> np.ndarray:
    """fixed_mask[i,j] is in {0,1}; only fixed_value > 0 denotes a fixed allocation."""
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
