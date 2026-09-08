from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class MonthMethodReportRow:
    """
    单月×单方法的汇总行。
    """

    month_id: str
    method: str
    status: str
    feasible: bool
    total_violations: int
    solve_time_sec: float
    objective_value: Optional[float]
    gnae: Optional[float]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key_obj = k
            if hasattr(key_obj, "item"):
                try:
                    key_obj = key_obj.item()
                except Exception:
                    pass
            if not isinstance(key_obj, str):
                key_obj = str(key_obj)
            out[key_obj] = _to_jsonable(v)
        return out

    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def write_summary_csv(rows: List[MonthMethodReportRow], output_path: Path) -> None:
    """
    输出汇总 CSV。
    """
    df = pd.DataFrame([r.__dict__ for r in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for suffix in ["", "_retry1", "_retry2", "_retry3", "_retry4", "_retry5"]:
        target = output_path
        if suffix:
            target = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        try:
            df.to_csv(target, index=False, encoding="utf-8-sig")
            return
        except PermissionError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error


def write_rows_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for suffix in ["", "_retry1", "_retry2", "_retry3", "_retry4", "_retry5"]:
        target = output_path
        if suffix:
            target = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        try:
            df.to_csv(target, index=False, encoding="utf-8-sig")
            return
        except PermissionError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error


def write_allocations_xlsx(
    allocation_df: pd.DataFrame,
    output_path: Path,
    extra_sheets: Dict[str, pd.DataFrame] | None = None,
) -> None:
    """
    输出分配矩阵到 xlsx（便于人工抽查）。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for suffix in ["", "_retry1", "_retry2", "_retry3", "_retry4", "_retry5"]:
        target = output_path
        if suffix:
            target = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        try:
            with pd.ExcelWriter(target) as writer:
                allocation_df.to_excel(writer, index=False, sheet_name="allocation")
                if extra_sheets:
                    used: set[str] = {"allocation"}
                    for raw_name, df in extra_sheets.items():
                        name = str(raw_name)[:31] if raw_name else "sheet"
                        if not name:
                            name = "sheet"
                        base = name
                        k = 1
                        while name in used:
                            suffix2 = f"_{k}"
                            name = f"{base[: 31 - len(suffix2)]}{suffix2}"
                            k += 1
                        used.add(name)
                        df.to_excel(writer, index=False, sheet_name=name)
            return
        except PermissionError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error


def write_violations_json(violations: Dict[str, Any], output_path: Path) -> None:
    """
    输出约束违背明细到 json。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_violations = _to_jsonable(violations)
    output_path.write_text(json.dumps(safe_violations, ensure_ascii=False, indent=2), encoding="utf-8")


def violations_to_count_rows(violations: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    constraint_results = violations.get("constraint_results", {}) if isinstance(violations, dict) else {}
    if not isinstance(constraint_results, dict):
        return rows
    for name, result in constraint_results.items():
        if not isinstance(result, dict):
            continue
        details = result.get("details", {})
        count = None
        if isinstance(details, dict):
            count = details.get("total_violations")
        if count is None:
            vio_list = result.get("violations", [])
            count = len(vio_list) if isinstance(vio_list, list) else 0
        try:
            count_int = int(count)
        except Exception:
            count_int = 0
        rows.append({"constraint_name": str(name), "count": count_int})
    return rows


def write_violation_counts_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for suffix in ["", "_retry1", "_retry2", "_retry3", "_retry4", "_retry5"]:
        target = output_path
        if suffix:
            target = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        try:
            df.to_csv(target, index=False, encoding="utf-8-sig")
            return
        except PermissionError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error


def violations_to_detail_rows(violations: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    constraint_results = violations.get("constraint_results", {}) if isinstance(violations, dict) else {}
    if not isinstance(constraint_results, dict):
        return rows
    for name, result in constraint_results.items():
        if not isinstance(result, dict):
            continue
        vio_list = result.get("violations", [])
        if not isinstance(vio_list, list):
            continue
        for i, v in enumerate(vio_list):
            v2 = _to_jsonable(v)
            rows.append(
                {
                    "constraint_name": str(name),
                    "violation_index": int(i),
                    "violation": json.dumps(v2, ensure_ascii=False, default=str),
                }
            )
    return rows


def write_violation_details_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for suffix in ["", "_retry1", "_retry2", "_retry3", "_retry4", "_retry5"]:
        target = output_path
        if suffix:
            target = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        try:
            df.to_csv(target, index=False, encoding="utf-8-sig")
            return
        except PermissionError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
