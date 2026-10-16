"""Unified LP, RuleBased, SA, and GA interfaces under frozen rules.

LP permits no automatic relaxation, disabled constraints, or heuristic fallback. SA/GA reuse snapshot neighborhood and repair operators with both a 60-second limit and a 50,000-objective-evaluation limit, recording actual evaluation counts."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RevisionSolverOutput:
    """Common experiment output including budgets and auditable LP optimality information."""

    status: str
    solve_time_sec: float
    objective_value: float | None
    allocation_matrix_full: pd.DataFrame
    objective_evaluations: int
    solver_status: str | None = None
    optimality_gap: float | None = None
    constraint_diagnostics: list[dict[str, Any]] | None = None


def solve_lp_frozen(month: Any, constraints: dict[str, Any], objective: dict[str, Any], time_limit_sec: int) -> RevisionSolverOutput:
    """Call the snapshot LP directly and report infeasibility without changing frozen rules or using a fallback."""
    from constraint_manager import ConstraintManager
    from linear_programming import LinearProgrammingAllocator

    start = time.perf_counter()
    manager = ConstraintManager(month.data_loader, constraint_config=dict(constraints))
    allocator = LinearProgrammingAllocator(month.data_loader, constraint_manager=manager)
    result = allocator.solve(constraint_config=dict(constraints), objective_config=dict(objective), time_limit=int(time_limit_sec))
    elapsed = time.perf_counter() - start
    status = str(result.get("status", "UNKNOWN"))
    allocation = result.get("allocation_matrix")
    if not isinstance(allocation, pd.DataFrame):
        allocation = pd.DataFrame()
    value = result.get("objective_value")
    # In this project's interface, CBC proves optimality only for Optimal status; leave the gap blank if no numerical gap is available.
    gap = 0.0 if status == "Optimal" else None
    diagnostics: list[dict[str, Any]] = []
    for name, constraint in allocator.model.constraints.items():
        slack = getattr(constraint, "slack", None)
        dual = getattr(constraint, "pi", None)
        if slack is not None:
            diagnostics.append({"constraint": str(name), "slack": float(slack), "dual_value": float(dual) if dual is not None else None})
    return RevisionSolverOutput(
        status=status,
        solver_status=status,
        solve_time_sec=float(elapsed),
        objective_value=float(value) if value is not None else None,
        allocation_matrix_full=allocation,
        objective_evaluations=0,
        optimality_gap=gap,
        constraint_diagnostics=diagnostics,
    )


def solve_rule_based_frozen(month: Any, constraints: dict[str, Any], objective: dict[str, Any]) -> RevisionSolverOutput:
    """Run the deterministic rule baseline and rescore it with the common numerical objective."""
    from bench.metrics import compute_objective_value
    from bench.solvers.rule_based import solve_with_rule_based

    raw = solve_with_rule_based(month.data_loader, dict(constraints), repair_strength=1.0)
    allocation = raw.allocation_matrix_full
    value = compute_objective_value(allocation, month.rounds, month.data_loader.get_existing_allocations(), constraints, objective)
    return RevisionSolverOutput("HEURISTIC", float(raw.solve_time_sec), float(value), allocation, 1)


def _energy(df: pd.DataFrame, month: Any, constraints: dict[str, Any], objective: dict[str, Any]) -> float:
    """Common SA/GA score: unified objective plus the same fast hard-constraint penalties."""
    from bench.metrics import compute_objective_value
    from bench.solvers.ga import _quick_violation_score

    value = compute_objective_value(df, month.rounds, month.data_loader.get_existing_allocations(), constraints, objective)
    violations = _quick_violation_score(df, month.data_loader, month.rounds, constraints)
    return float(value + float(violations) * 1e6)


def solve_sa_budgeted(month: Any, constraints: dict[str, Any], objective: dict[str, Any], seed: int, max_time_sec: float, max_evaluations: int) -> RevisionSolverOutput:
    """Run fixed-seed SA under both time and objective-evaluation limits."""
    from bench.metrics import compute_objective_value
    from bench.solvers.rule_based import recompute_totals_and_rates, solve_with_rule_based
    from bench.solvers.sa import _neighbor

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    current = solve_with_rule_based(month.data_loader, dict(constraints), repair_strength=1.0).allocation_matrix_full.copy()
    current_energy = _energy(current, month, constraints, objective)
    evaluations = 1
    best, best_energy = current.copy(), current_energy
    temperature = max(1.0, abs(current_energy) * 0.01)
    while evaluations < int(max_evaluations) and (time.perf_counter() - start) < float(max_time_sec):
        candidate = _neighbor(current, month.data_loader, month.rounds, month.data_loader.get_existing_allocations(), rng)
        candidate_energy = _energy(candidate, month, constraints, objective)
        evaluations += 1
        if candidate_energy <= current_energy or rng.random() < math.exp(-(candidate_energy - current_energy) / max(1e-9, temperature)):
            current, current_energy = candidate, candidate_energy
            if candidate_energy < best_energy:
                best, best_energy = candidate.copy(), candidate_energy
        temperature *= 0.995
    elapsed = time.perf_counter() - start
    best = recompute_totals_and_rates(best, month.rounds)
    value = compute_objective_value(best, month.rounds, month.data_loader.get_existing_allocations(), constraints, objective)
    cap = "evaluation_cap" if evaluations >= int(max_evaluations) else "time_cap"
    return RevisionSolverOutput(f"SA|{cap}", float(elapsed), float(value), best, evaluations)


def solve_ga_budgeted(month: Any, constraints: dict[str, Any], objective: dict[str, Any], seed: int, max_time_sec: float, max_evaluations: int, population_size: int) -> RevisionSolverOutput:
    """Run fixed-seed GA, counting every score toward the common evaluation budget."""
    from bench.metrics import compute_objective_value
    from bench.solvers.ga import _crossover_rows, _mutate_transfer, _repair, _tournament_select
    from bench.solvers.rule_based import _fixed_mask, recompute_totals_and_rates, solve_with_rule_based

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    base = solve_with_rule_based(month.data_loader, dict(constraints), repair_strength=1.0).allocation_matrix_full
    fixed = _fixed_mask(month.data_loader.get_existing_allocations(), len(base), month.rounds)
    population = [_repair(base.copy(), month.data_loader, month.rounds, fixed, constraints)]
    while len(population) < int(population_size):
        candidate = _mutate_transfer(population[0], month.rounds, fixed, rng, tries=20)
        population.append(_repair(candidate, month.data_loader, month.rounds, fixed, constraints))

    evaluations = 0
    best = population[0].copy()
    best_energy = float("inf")
    while evaluations < int(max_evaluations) and (time.perf_counter() - start) < float(max_time_sec):
        scored: list[tuple[pd.DataFrame, float]] = []
        for candidate in population:
            if evaluations >= int(max_evaluations) or (time.perf_counter() - start) >= float(max_time_sec):
                break
            energy = _energy(candidate, month, constraints, objective)
            evaluations += 1
            scored.append((candidate, energy))
            if energy < best_energy:
                best, best_energy = candidate.copy(), energy
        if not scored:
            break
        scored.sort(key=lambda item: item[1])
        elite_count = max(1, min(5, int(population_size * 0.1)))
        next_population = [item[0].copy() for item in scored[:elite_count]]
        while len(next_population) < int(population_size):
            parent_a = _tournament_select(scored, rng, k=3)
            parent_b = _tournament_select(scored, rng, k=3)
            child = _crossover_rows(parent_a, parent_b, month.rounds, rng)
            if rng.random() < 0.7:
                child = _mutate_transfer(child, month.rounds, fixed, rng, tries=10)
            next_population.append(_repair(child, month.data_loader, month.rounds, fixed, constraints))
        population = next_population
    elapsed = time.perf_counter() - start
    best = recompute_totals_and_rates(best, month.rounds)
    value = compute_objective_value(best, month.rounds, month.data_loader.get_existing_allocations(), constraints, objective)
    cap = "evaluation_cap" if evaluations >= int(max_evaluations) else "time_cap"
    return RevisionSolverOutput(f"GA|{cap}", float(elapsed), float(value), best, evaluations)


