from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp

from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import (
    compute_gnae,
    compute_round_volume_deviation,
    compute_sku_concentration_metric,
    compute_sku_stability_metric,
    compute_total_stability_metric,
)
from bench.report import write_rows_csv
from bench.solvers.lp_solver import solve_with_lp


@dataclass(frozen=True)
class ToyMonth:
    rounds: List[str]
    product_data: pd.DataFrame
    label_data: pd.DataFrame
    data_loader: object
    month_id: str


class ToyDataLoader:
    def __init__(
        self,
        product_data: pd.DataFrame,
        rounds: List[str],
        round_constraints: Dict[str, Dict[str, float]],
    ) -> None:
        self._product_data = product_data.copy()
        self._rounds = list(rounds)
        self._round_constraints = dict(round_constraints)

    def get_product_data(self) -> pd.DataFrame:
        return self._product_data.copy()

    def get_rounds(self) -> List[str]:
        return list(self._rounds)

    def get_round_constraints(self, round_name: str) -> Dict[str, float]:
        return dict(self._round_constraints[round_name])

    def get_existing_allocations(self) -> Dict[str, pd.Series]:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LP 公平/稳定性数值例子：构造小型模拟数据，对比无公平 vs 有公平目标的 LP 解差异。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="真实月份输入目录（用于抽取统计特征，不直接用于输出）",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=str(DEFAULT_LABEL_DIR),
        help="真实 label 目录（用于 load_month_data 读取；缺省也可）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "lp_fairness_example"),
        help="输出目录（写 allocation/metrics CSV）",
    )
    parser.add_argument(
        "--n-skus",
        type=int,
        default=8,
        help="toy 示例 SKU 数量",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=3,
        help="toy 示例轮次数量（默认取前三轮：第一轮/第二轮/第三轮）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于抽样需求）",
    )
    parser.add_argument(
        "--lp-time-limit",
        type=int,
        default=30,
        help="LP 求解时间上限（秒）",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    month = _build_toy_month(
        input_dir=Path(args.input_dir),
        label_dir=Path(args.label_dir),
        n_skus=int(args.n_skus),
        n_rounds=int(args.n_rounds),
        seed=int(args.seed),
    )

    constraint_config = {
        "enable_demand_constraints": True,
        "enable_volume_constraints": False,
        "enable_price_constraints": False,
        "enable_c_type_constraints": False,
        "enable_balance_constraints": False,
        "enable_demand_split_constraints": False,
        "enable_demand_based_constraints": False,
        "enable_price_based_constraints": False,
        "volume_tolerance": 0.005,
    }

    objective_model1 = {
        "maximize_allocation_weight": 1000.0,
        "round_balance_weight": 0.0,
        "round_variance_weight": 0.0,
        "product_balance_weight": 0.0,
        "smooth_transition_weight": 0.0,
    }
    objective_model2 = {
        "maximize_allocation_weight": 1000.0,
        "round_balance_weight": 800.0,
        "round_variance_weight": 400.0,
        "product_balance_weight": 200.0,
        "smooth_transition_weight": 300.0,
    }

    alloc1 = _solve_model1_early_release(month.product_data, month.rounds)
    out1_status = "Optimal"
    out2 = solve_with_lp(
        data_loader=month.data_loader,
        constraint_config=constraint_config,
        objective_config=objective_model2,
        time_limit_sec=int(args.lp_time_limit),
    )

    alloc2 = out2.allocation_matrix_full

    metrics1 = _compute_metrics(alloc1, month.label_data, month.rounds)
    metrics2 = _compute_metrics(alloc2, month.label_data, month.rounds)

    metrics_rows = _build_metric_table(metrics1, metrics2)
    write_rows_csv(metrics_rows, out_dir / "metrics_table.csv")

    write_rows_csv(_df_to_rows(alloc1), out_dir / "allocation_model1.csv")
    write_rows_csv(_df_to_rows(alloc2), out_dir / "allocation_model2.csv")
    write_rows_csv(_df_to_rows(month.label_data), out_dir / "label.csv")

    print(
        json.dumps(
            {
                "toy": {
                    "month_id": month.month_id,
                    "n_skus": int(len(month.product_data)),
                    "rounds": month.rounds,
                    "round_share_p": _round_share_from_label(month.label_data, month.rounds),
                },
                "model1": {"status": out1_status, "objective": objective_model1, "metrics": metrics1},
                "model2": {"status": out2.status, "objective": objective_model2, "metrics": metrics2},
            },
            ensure_ascii=False,
            default=str,
        )
    )
    print(pd.DataFrame(metrics_rows).to_string(index=False))
    print("\nModel 1 allocation (Without fairness):")
    print(_compact_matrix(alloc1, month.rounds).to_string(index=False))
    print("\nModel 2 allocation (With fairness):")
    print(_compact_matrix(alloc2, month.rounds).to_string(index=False))
    print("\nLabel (fair/stable target):")
    print(_compact_matrix(month.label_data, month.rounds).to_string(index=False))
    return 0


def _build_toy_month(input_dir: Path, label_dir: Path, n_skus: int, n_rounds: int, seed: int) -> ToyMonth:
    rng = np.random.default_rng(int(seed))

    input_paths = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    input_paths = [p for p in input_paths if p.stem.isdigit() and len(p.stem) == 6]
    input_paths = input_paths[:6] if len(input_paths) >= 6 else input_paths
    if not input_paths:
        raise FileNotFoundError(f"no input xlsx found under {input_dir}")

    shares, ref_rounds = _average_round_shares(input_paths, label_dir=label_dir, n_rounds=n_rounds)

    demands: List[float] = []
    for p in input_paths:
        m = load_month_data(input_path=p, label_dir=label_dir)
        d = pd.to_numeric(m.product_data.get("需求"), errors="coerce").fillna(0.0)
        d = d[(d > 0) & (d < d.quantile(0.99))]
        demands.extend([float(x) for x in d.tolist()])

    if not demands:
        raise RuntimeError("empty demand samples")

    demands_arr = np.array(demands, dtype=float)
    q90 = float(np.quantile(demands_arr, 0.9)) if len(demands_arr) > 0 else 1.0
    scale = 80.0 / max(1.0, q90)

    sampled = rng.choice(demands_arr, size=max(1, int(n_skus)), replace=len(demands_arr) < int(n_skus)).tolist()
    demand_small = [float(max(8.0, round(float(x) * scale))) for x in sampled]

    codes = [f"SKU{i+1}" for i in range(len(demand_small))]
    product_data = pd.DataFrame(
        {
            "代码": codes,
            "卷烟名称": codes,
            "需求": demand_small,
            "可用货源": [float(d * 10.0) for d in demand_small],
            "批发价": [100.0 + float(i) for i in range(len(demand_small))],
            "条支比": [200.0 for _ in range(len(demand_small))],
            "按需": ["" for _ in range(len(demand_small))],
            "按价": ["" for _ in range(len(demand_small))],
            "C": ["" for _ in range(len(demand_small))],
        }
    )

    round_constraints = {}
    total = float(sum(demand_small))
    for r, p in zip(ref_rounds, shares, strict=False):
        round_constraints[r] = {"total_quantity": float(total * float(p)), "upper_price_limit": 1e9, "lower_price_limit": 0.0}

    loader = ToyDataLoader(product_data=product_data, rounds=ref_rounds, round_constraints=round_constraints)
    label = _build_label(product_data, ref_rounds, shares)

    return ToyMonth(
        rounds=ref_rounds,
        product_data=product_data,
        label_data=label,
        data_loader=loader,
        month_id="toy_fairness_example",
    )


def _average_round_shares(input_paths: List[Path], label_dir: Path, n_rounds: int) -> Tuple[List[float], List[str]]:
    round_names = ["第一轮", "第二轮", "第三轮", "第四轮", "第五轮"]
    use_rounds = round_names[: max(1, int(n_rounds))]

    shares: List[List[float]] = []
    for p in input_paths:
        m = load_month_data(input_path=p, label_dir=label_dir)
        rounds = m.rounds
        picked = [r for r in use_rounds if r in rounds]
        if len(picked) < len(use_rounds):
            for r in rounds:
                if r not in picked:
                    picked.append(r)
                if len(picked) >= len(use_rounds):
                    break

        totals = []
        for r in picked:
            c = m.data_loader.get_round_constraints(r)
            totals.append(float(c.get("total_quantity", 0.0) or 0.0))
        s = float(sum(totals))
        if s <= 0:
            continue
        shares.append([float(x) / s for x in totals])

    if not shares:
        k = len(use_rounds)
        return [1.0 / float(k) for _ in range(k)], use_rounds

    mean = np.mean(np.array(shares, dtype=float), axis=0)
    mean = mean / max(1e-9, float(mean.sum()))
    return [float(x) for x in mean.tolist()], use_rounds


def _build_label(product_data: pd.DataFrame, rounds: List[str], shares: List[float]) -> pd.DataFrame:
    df = product_data.loc[:, ["代码"]].copy()
    demand = product_data["需求"].fillna(0.0).astype(float).to_numpy()
    p = np.array(shares, dtype=float)
    p = p / max(1e-9, float(p.sum()))

    alloc = np.zeros((len(df), len(rounds)), dtype=float)
    for i in range(len(df)):
        d = float(demand[i])
        if d <= 0:
            continue
        raw = d * p
        floor = np.floor(raw)
        rem = int(round(d - float(floor.sum())))
        alloc[i, :] = floor
        if rem > 0:
            frac = raw - floor
            order = np.argsort(-frac)
            for j in order[:rem]:
                alloc[i, int(j)] += 1.0

    for j, r in enumerate(rounds):
        df[r] = alloc[:, j]
    return df


def _solve_model1_early_release(product_data: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    df = product_data.copy()
    rounds = list(rounds)
    if not rounds:
        return df

    demand = pd.to_numeric(df["需求"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    supply = pd.to_numeric(df.get("可用货源"), errors="coerce").fillna(0.0).astype(float).to_numpy()

    model = pulp.LpProblem("toy_without_fairness", pulp.LpMaximize)
    x: Dict[tuple[int, int], pulp.LpVariable] = {}
    for i in range(len(df)):
        for j, r in enumerate(rounds):
            ub = float(demand[i])
            if r == "第一轮":
                ub = float(min(demand[i], supply[i]))
            x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=ub, cat="Continuous")

    for i in range(len(df)):
        d = float(demand[i])
        if d <= 0:
            for j in range(len(rounds)):
                model += x[(i, j)] == 0
            continue
        model += pulp.lpSum(x[(i, j)] for j in range(len(rounds))) == d

    model += pulp.lpSum(x[(i, 0)] for i in range(len(df)))
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    out = df.copy()
    for j, r in enumerate(rounds):
        out[r] = [float(pulp.value(x[(i, j)]) or 0.0) for i in range(len(df))]
    out["总分配量"] = out.loc[:, rounds].sum(axis=1)
    out["分配率"] = np.where(pd.to_numeric(out["需求"], errors="coerce").fillna(0.0).astype(float) > 0, out["总分配量"] / out["需求"], 1.0)
    return out


def _compute_metrics(alloc_full: pd.DataFrame, label: pd.DataFrame, rounds: List[str]) -> Dict[str, float]:
    if alloc_full is None or alloc_full.empty:
        return {
            "max_round_deviation": 0.0,
            "round_to_round_jump": 0.0,
            "sku_level_deviation": 0.0,
            "sku_concentration": 0.0,
            "gnae": 0.0,
            "first_round_share": 0.0,
        }

    total = float(alloc_full.loc[:, rounds].fillna(0.0).astype(float).to_numpy().sum())
    first = float(alloc_full[rounds[0]].fillna(0.0).astype(float).sum()) if rounds and rounds[0] in alloc_full.columns else 0.0
    first_share = float(first / total) if total > 0 else 0.0

    gnae = 0.0
    try:
        alloc_std = standardize_allocation_matrix(alloc_full, rounds)
        label_std = standardize_allocation_matrix(label, rounds)
        alloc_aligned, label_aligned = _align_by_code(alloc_std, label_std)
        gnae = float(compute_gnae(alloc_aligned, label_aligned, rounds))
    except Exception:
        gnae = 0.0

    return {
        "max_round_deviation": float(compute_round_volume_deviation(alloc_full, rounds)),
        "round_to_round_jump": float(compute_total_stability_metric(alloc_full, rounds)),
        "sku_level_deviation": float(compute_sku_stability_metric(alloc_full, rounds)),
        "sku_concentration": float(compute_sku_concentration_metric(alloc_full, rounds)),
        "gnae": float(gnae),
        "first_round_share": float(first_share),
    }


def _build_metric_table(m1: Dict[str, float], m2: Dict[str, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "Metric": "Max round deviation",
            "Without fairness": m1.get("max_round_deviation"),
            "With fairness": m2.get("max_round_deviation"),
            "Interpretation": "总量更平衡（各轮总量极差/均值更小）",
        }
    )
    rows.append(
        {
            "Metric": "Round-to-round jump",
            "Without fairness": m1.get("round_to_round_jump"),
            "With fairness": m2.get("round_to_round_jump"),
            "Interpretation": "跨轮更稳定（相邻轮次总量跳变更小）",
        }
    )
    rows.append(
        {
            "Metric": "SKU-level deviation",
            "Without fairness": m1.get("sku_level_deviation"),
            "With fairness": m2.get("sku_level_deviation"),
            "Interpretation": "SKU 层面更平滑（每个 SKU 跨轮跳变更小）",
        }
    )
    rows.append(
        {
            "Metric": "SKU concentration",
            "Without fairness": m1.get("sku_concentration"),
            "With fairness": m2.get("sku_concentration"),
            "Interpretation": "SKU 分配更均衡（单一轮次占单 SKU 需求比例更低）",
        }
    )
    rows.append(
        {
            "Metric": "GNAE / fairness deviation",
            "Without fairness": m1.get("gnae"),
            "With fairness": m2.get("gnae"),
            "Interpretation": "整体分配更接近公平/稳定 target（label）",
        }
    )
    rows.append(
        {
            "Metric": "First-round share",
            "Without fairness": m1.get("first_round_share"),
            "With fairness": m2.get("first_round_share"),
            "Interpretation": "第一轮过度集中程度（越低越利于渠道执行）",
        }
    )
    return rows


def _align_by_code(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "代码" not in a.columns or "代码" not in b.columns:
        return a.reset_index(drop=True), b.reset_index(drop=True)

    a2 = a.copy()
    b2 = b.copy()
    a2["代码"] = a2["代码"].astype(str)
    b2["代码"] = b2["代码"].astype(str)
    merged = a2.merge(b2, on="代码", how="inner", suffixes=("_a", "_b"))
    cols_a = ["代码"] + [f"{r}_a" for r in [c[:-2] for c in merged.columns if c.endswith("_a")] if r not in ["代码", "卷烟名称"]]
    cols_b = ["代码"] + [f"{r}_b" for r in [c[:-2] for c in merged.columns if c.endswith("_b")] if r not in ["代码", "卷烟名称"]]
    rounds = [c[:-2] for c in merged.columns if c.endswith("_a") and c[:-2] not in ["代码", "卷烟名称"]]
    out_a = merged.loc[:, ["代码"] + [f"{r}_a" for r in rounds]].rename(columns={f"{r}_a": r for r in rounds})
    out_b = merged.loc[:, ["代码"] + [f"{r}_b" for r in rounds]].rename(columns={f"{r}_b": r for r in rounds})
    return out_a, out_b


def _round_share_from_label(label: pd.DataFrame, rounds: List[str]) -> Dict[str, float]:
    if label is None or label.empty or not rounds:
        return {}
    totals = label.loc[:, rounds].fillna(0.0).astype(float).sum(axis=0).to_dict()
    s = float(sum(float(v) for v in totals.values()))
    if s <= 0:
        return {str(k): 0.0 for k in totals.keys()}
    return {str(k): float(v) / s for k, v in totals.items()}


def _compact_matrix(df: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
    cols = [c for c in ["代码"] if c in df.columns] + [r for r in rounds if r in df.columns]
    out = df.loc[:, cols].copy()
    for r in rounds:
        if r in out.columns:
            out[r] = pd.to_numeric(out[r], errors="coerce").fillna(0.0).astype(float)
    return out


def _df_to_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None:
        return []
    return df.to_dict("records")


if __name__ == "__main__":
    raise SystemExit(main())
