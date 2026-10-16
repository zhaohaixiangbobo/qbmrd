from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..bootstrap import ensure_backend_on_path
from .rule_based import solve_with_rule_based


@dataclass(frozen=True)
class SolverOutput:
    """Common solver-output structure."""

    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame


def solve_with_lp(
    data_loader: object,
    constraint_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    time_limit_sec: int = 300,
) -> SolverOutput:
    """Reuse the backend LP solver as the main-method benchmark."""
    ensure_backend_on_path()
    from constraint_manager import ConstraintManager
    from linear_programming import LinearProgrammingAllocator

    def _run(config: Dict[str, Any]) -> Dict[str, Any]:
        manager = ConstraintManager(data_loader, constraint_config=config)
        allocator = LinearProgrammingAllocator(data_loader, constraint_manager=manager)
        return allocator.solve(
            constraint_config=config,
            objective_config=objective_config,
            time_limit=time_limit_sec,
        )

    start = time.perf_counter()
    applied = dict(constraint_config)
    auto_adjustments: List[str] = []
    result = _run(applied)

    status0 = str(result.get("status", "UNKNOWN"))
    alloc0 = result.get("allocation_matrix")
    need_fallback = status0 == "Infeasible" or not isinstance(alloc0, pd.DataFrame)
    if need_fallback:
        for key in [
            "enable_demand_split_constraints",
            "enable_price_based_constraints",
            "enable_c_type_constraints",
        ]:
            if not applied.get(key, True):
                continue
            applied2 = dict(applied)
            applied2[key] = False
            auto_adjustments.append(f"{key}=False")
            result = _run(applied2)
            applied = applied2
            status_k = str(result.get("status", "UNKNOWN"))
            alloc_k = result.get("allocation_matrix")
            if status_k == "Optimal" and isinstance(alloc_k, pd.DataFrame):
                break

    status1 = str(result.get("status", "UNKNOWN"))
    alloc1 = result.get("allocation_matrix")
    need_more = status1 == "Infeasible" or not isinstance(alloc1, pd.DataFrame)
    if need_more:
        for tol in [0.01, 0.02, 0.05]:
            applied2 = dict(applied)
            applied2["volume_tolerance"] = max(float(applied2.get("volume_tolerance", 0.005)), float(tol))
            auto_adjustments.append(f"volume_tolerance={applied2['volume_tolerance']}")
            result = _run(applied2)
            applied = applied2
            status_k = str(result.get("status", "UNKNOWN"))
            alloc_k = result.get("allocation_matrix")
            if status_k == "Optimal" and isinstance(alloc_k, pd.DataFrame):
                break

    status2 = str(result.get("status", "UNKNOWN"))
    alloc2 = result.get("allocation_matrix")
    need_more2 = status2 == "Infeasible" or not isinstance(alloc2, pd.DataFrame)
    if need_more2 and applied.get("enable_price_constraints", True):
        applied2 = dict(applied)
        applied2["enable_price_constraints"] = False
        auto_adjustments.append("enable_price_constraints=False")
        result = _run(applied2)
        applied = applied2

    end = time.perf_counter()

    allocation_matrix = result.get("allocation_matrix")
    status = str(result.get("status", "UNKNOWN"))
    if auto_adjustments:
        status = f"{status}|auto:{','.join(auto_adjustments)}"

    if not isinstance(allocation_matrix, pd.DataFrame):
        heuristic = solve_with_rule_based(
            data_loader=data_loader,
            constraint_config=constraint_config,
            repair_strength=1.0,
        )
        return SolverOutput(
            status=f"{status}|fallback:RuleBased",
            solve_time_sec=float(end - start),
            objective_value=None,
            allocation_matrix_full=heuristic.allocation_matrix_full,
        )

    if isinstance(allocation_matrix, pd.DataFrame):
        allocation_matrix_full = allocation_matrix
    else:
        allocation_matrix_full = pd.DataFrame()

    objective_value = result.get("objective_value")
    solve_time = float(end - start)

    return SolverOutput(
        status=status,
        solve_time_sec=solve_time,
        objective_value=float(objective_value) if objective_value is not None else None,
        allocation_matrix_full=allocation_matrix_full,
    )
