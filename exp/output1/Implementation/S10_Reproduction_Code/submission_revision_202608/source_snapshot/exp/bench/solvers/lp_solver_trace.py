"""English review copy of bench/solvers/lp_solver.py.
Historical source logic; see HISTORICAL_CODE_GUIDE.md for execution scope and provenance.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from ..bootstrap import ensure_backend_on_path
from .rule_based import solve_with_rule_based

@dataclass(frozen=True)
class SolverOutput:
    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame

@dataclass(frozen=True)
class RelaxationAttempt:
    step_index: int
    action: str
    cumulative_adjustments: List[str]
    applied_constraint_config: Dict[str, Any]
    status: str
    solve_time_sec: float
    objective_value: Optional[float]
    allocation_matrix_full: pd.DataFrame
    is_lp_attempt: bool
    is_optimal: bool

@dataclass(frozen=True)
class RelaxationTraceResult:
    output: SolverOutput
    attempts: List[RelaxationAttempt]
    final_constraint_config: Dict[str, Any]
    used_rule_based_fallback: bool

def solve_with_lp(data_loader: object, constraint_config: Dict[str, Any], objective_config: Dict[str, Any], time_limit_sec: int=300) -> SolverOutput:
    return _solve_with_lp_core(data_loader=data_loader, constraint_config=constraint_config, objective_config=objective_config, time_limit_sec=time_limit_sec, collect_trace=False).output

def solve_with_lp_with_trace(data_loader: object, constraint_config: Dict[str, Any], objective_config: Dict[str, Any], time_limit_sec: int=300) -> RelaxationTraceResult:
    return _solve_with_lp_core(data_loader=data_loader, constraint_config=constraint_config, objective_config=objective_config, time_limit_sec=time_limit_sec, collect_trace=True)

def _solve_with_lp_core(data_loader: object, constraint_config: Dict[str, Any], objective_config: Dict[str, Any], time_limit_sec: int, collect_trace: bool) -> RelaxationTraceResult:
    ensure_backend_on_path()
    from constraint_manager import ConstraintManager
    from linear_programming import LinearProgrammingAllocator
    attempts: List[RelaxationAttempt] = []

    def _run(config: Dict[str, Any], action: str, adjustments: List[str]) -> Dict[str, Any]:
        attempt_start = time.perf_counter()
        manager = ConstraintManager(data_loader, constraint_config=config)
        allocator = LinearProgrammingAllocator(data_loader, constraint_manager=manager)
        result = allocator.solve(constraint_config=config, objective_config=objective_config, time_limit=time_limit_sec)
        allocation = result.get('allocation_matrix')
        status = str(result.get('status', 'UNKNOWN'))
        if collect_trace:
            allocation_frame = allocation.copy() if isinstance(allocation, pd.DataFrame) else pd.DataFrame()
            attempts.append(RelaxationAttempt(step_index=len(attempts), action=action, cumulative_adjustments=list(adjustments), applied_constraint_config=dict(config), status=status, solve_time_sec=float(time.perf_counter() - attempt_start), objective_value=float(result['objective_value']) if result.get('objective_value') is not None else None, allocation_matrix_full=allocation_frame, is_lp_attempt=True, is_optimal=bool(status == 'Optimal' and isinstance(allocation, pd.DataFrame))))
        return result
    start = time.perf_counter()
    applied = dict(constraint_config)
    auto_adjustments: List[str] = []
    result = _run(applied, 'initial', auto_adjustments)
    status0 = str(result.get('status', 'UNKNOWN'))
    alloc0 = result.get('allocation_matrix')
    need_fallback = status0 == 'Infeasible' or not isinstance(alloc0, pd.DataFrame)
    if need_fallback:
        for key in ['enable_demand_split_constraints', 'enable_price_based_constraints', 'enable_c_type_constraints']:
            if not applied.get(key, True):
                continue
            applied2 = dict(applied)
            applied2[key] = False
            adjustment = f'{key}=False'
            auto_adjustments.append(adjustment)
            result = _run(applied2, adjustment, auto_adjustments)
            applied = applied2
            status_k = str(result.get('status', 'UNKNOWN'))
            alloc_k = result.get('allocation_matrix')
            if status_k == 'Optimal' and isinstance(alloc_k, pd.DataFrame):
                break
    status1 = str(result.get('status', 'UNKNOWN'))
    alloc1 = result.get('allocation_matrix')
    need_more = status1 == 'Infeasible' or not isinstance(alloc1, pd.DataFrame)
    if need_more:
        for tol in [0.01, 0.02, 0.05]:
            applied2 = dict(applied)
            applied2['volume_tolerance'] = max(float(applied2.get('volume_tolerance', 0.005)), float(tol))
            adjustment = f"volume_tolerance={applied2['volume_tolerance']}"
            auto_adjustments.append(adjustment)
            result = _run(applied2, adjustment, auto_adjustments)
            applied = applied2
            status_k = str(result.get('status', 'UNKNOWN'))
            alloc_k = result.get('allocation_matrix')
            if status_k == 'Optimal' and isinstance(alloc_k, pd.DataFrame):
                break
    status2 = str(result.get('status', 'UNKNOWN'))
    alloc2 = result.get('allocation_matrix')
    need_more2 = status2 == 'Infeasible' or not isinstance(alloc2, pd.DataFrame)
    if need_more2 and applied.get('enable_price_constraints', True):
        applied2 = dict(applied)
        applied2['enable_price_constraints'] = False
        adjustment = 'enable_price_constraints=False'
        auto_adjustments.append(adjustment)
        result = _run(applied2, adjustment, auto_adjustments)
        applied = applied2
    end = time.perf_counter()
    allocation_matrix = result.get('allocation_matrix')
    status = str(result.get('status', 'UNKNOWN'))
    if auto_adjustments:
        status = f"{status}|auto:{','.join(auto_adjustments)}"
    if not isinstance(allocation_matrix, pd.DataFrame):
        heuristic = solve_with_rule_based(data_loader=data_loader, constraint_config=constraint_config, repair_strength=1.0)
        output = SolverOutput(status=f'{status}|fallback:RuleBased', solve_time_sec=float(end - start), objective_value=None, allocation_matrix_full=heuristic.allocation_matrix_full)
        if collect_trace:
            attempts.append(RelaxationAttempt(step_index=len(attempts), action='fallback:RuleBased', cumulative_adjustments=list(auto_adjustments), applied_constraint_config=dict(constraint_config), status=output.status, solve_time_sec=float(heuristic.solve_time_sec), objective_value=heuristic.objective_value, allocation_matrix_full=heuristic.allocation_matrix_full.copy(), is_lp_attempt=False, is_optimal=False))
        return RelaxationTraceResult(output=output, attempts=attempts, final_constraint_config=dict(applied), used_rule_based_fallback=True)
    output = SolverOutput(status=status, solve_time_sec=float(end - start), objective_value=float(result['objective_value']) if result.get('objective_value') is not None else None, allocation_matrix_full=allocation_matrix)
    return RelaxationTraceResult(output=output, attempts=attempts, final_constraint_config=dict(applied), used_rule_based_fallback=False)
