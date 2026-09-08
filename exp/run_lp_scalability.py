from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bench.configs import default_experiment_configs
from bench.io import DEFAULT_INPUT_DIR, DEFAULT_LABEL_DIR, load_month_data, standardize_allocation_matrix
from bench.metrics import compute_gnae, validate_constraints
from bench.report import write_rows_csv
from bench.solvers.lp_solver import solve_with_lp


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    demand_scale: float
    sku_scale: float
    rounds: int


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
        description="LP 规模扩展/压力测试：基于202501~202506逐月生成synthetic实例（S1~S4），仅运行LP，输出可用于论文的可扩展性表。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="真实月份输入目录（用于抽取统计特征）",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default=str(DEFAULT_LABEL_DIR),
        help="真实 label 目录（用于 load_month_data；本实验主要用 synthetic label）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "output" / "lp_scalability"),
        help="输出目录（生成 lp_scalability_summary.csv 与 lp_scalability_agg.csv）",
    )
    parser.add_argument(
        "--base-months",
        type=str,
        default="all",
        help="基准月份：all 或逗号分隔列表（例如 202501,202504）",
    )
    parser.add_argument(
        "--lp-time-limit",
        type=int,
        default=60,
        help="每个实例的 LP 求解时间上限（秒）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="resampling 随机种子",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每个 base month × scale 的重复次数（用于平滑随机性）",
    )
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = default_experiment_configs()

    input_paths = sorted(p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    input_paths = [p for p in input_paths if p.stem.isdigit() and len(p.stem) == 6]
    if not input_paths:
        raise FileNotFoundError(f"no input xlsx found under {input_dir}")

    allowed = _parse_base_months(args.base_months)
    if allowed is not None:
        input_paths = [p for p in input_paths if p.stem in allowed]

    scales = [
        ScaleSpec(name="S1 Small", demand_scale=0.5, sku_scale=0.5, rounds=3),
        ScaleSpec(name="S2 Field-scale", demand_scale=1.0, sku_scale=1.0, rounds=4),
        ScaleSpec(name="S3 Large", demand_scale=2.0, sku_scale=1.5, rounds=5),
        ScaleSpec(name="S4 Extra-large", demand_scale=5.0, sku_scale=2.0, rounds=6),
    ]

    constraint_config = {
        "enable_demand_constraints": True,
        "enable_volume_constraints": True,
        "enable_price_constraints": False,
        "enable_c_type_constraints": False,
        "enable_balance_constraints": False,
        "enable_demand_split_constraints": False,
        "enable_demand_based_constraints": False,
        "enable_price_based_constraints": False,
        "volume_tolerance": float(cfg.constraint_config.get("volume_tolerance", 0.005) or 0.005),
    }

    objective_config = {
        "maximize_allocation_weight": float(cfg.objective_config.get("maximize_allocation_weight", 1000.0) or 1000.0),
        "smooth_transition_weight": float(cfg.objective_config.get("smooth_transition_weight", 300.0) or 300.0),
        "round_balance_weight": 0.0,
        "round_variance_weight": 0.0,
        "product_balance_weight": 0.0,
    }

    rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        base = load_month_data(input_path=input_path, label_dir=label_dir)
        base_month = base.month_id

        base_product = base.product_data.copy()
        base_product = base_product[pd.to_numeric(base_product.get("需求"), errors="coerce").fillna(0.0) > 0].copy()
        base_product = base_product.reset_index(drop=True)
        if base_product.empty:
            continue

        base_p = _round_share_from_month(base.data_loader, base.rounds)
        for scale in scales:
            for rep in range(max(1, int(args.repeats))):
                instance = _build_scaled_instance(
                    base_month=base_month,
                    base_product=base_product,
                    base_round_share=base_p,
                    scale=scale,
                    seed=int(args.seed) + rep,
                )

                out = solve_with_lp(
                    data_loader=instance.data_loader,
                    constraint_config=constraint_config,
                    objective_config=objective_config,
                    time_limit_sec=int(args.lp_time_limit),
                )

                rounds = instance.rounds
                has_round_cols = isinstance(out.allocation_matrix_full, pd.DataFrame) and all(
                    r in out.allocation_matrix_full.columns for r in rounds
                )
                if has_round_cols:
                    constraint_check = validate_constraints(
                        data_loader=instance.data_loader,
                        allocation_matrix_full=out.allocation_matrix_full,
                        constraint_config=constraint_config,
                    )
                else:
                    constraint_check = validate_constraints(
                        data_loader=instance.data_loader,
                        allocation_matrix_full=pd.DataFrame(),
                        constraint_config=constraint_config,
                    )

                gnae = None
                if has_round_cols:
                    alloc_std = standardize_allocation_matrix(out.allocation_matrix_full, rounds)
                    label_std = standardize_allocation_matrix(instance.label_data, rounds)
                    a2, y2 = _align_by_code(alloc_std, label_std, rounds)
                    gnae = compute_gnae(a2, y2, rounds)

                rows.append(
                    {
                        "base_month": base_month,
                        "scale": scale.name,
                        "method": "LP",
                        "number_of_skus": int(len(instance.product_data)),
                        "rounds": int(len(instance.rounds)),
                        "demand_scale": float(scale.demand_scale),
                        "gnae": float(gnae) if gnae is not None else None,
                        "rule_deviations": int(constraint_check.total_violations),
                        "solve_time_sec": float(out.solve_time_sec),
                        "status": str(out.status),
                        "replicate_id": int(rep),
                    }
                )

    summary_path = out_dir / "lp_scalability_summary.csv"
    write_rows_csv(rows, summary_path)

    agg_rows = _aggregate(rows)
    agg_path = out_dir / "lp_scalability_agg.csv"
    write_rows_csv(agg_rows, agg_path)

    print(
        json.dumps(
            {"output_dir": str(out_dir), "summary_csv": str(summary_path), "agg_csv": str(agg_path)},
            ensure_ascii=False,
        )
    )
    return 0


@dataclass(frozen=True)
class ScaledInstance:
    rounds: List[str]
    product_data: pd.DataFrame
    label_data: pd.DataFrame
    data_loader: object


def _parse_base_months(text: str) -> Optional[set[str]]:
    t = (text or "").strip()
    if not t or t.lower() == "all":
        return None
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return set(parts) if parts else None


def _round_names(k: int) -> List[str]:
    names = ["第一轮", "第二轮", "第三轮", "第四轮", "第五轮", "第六轮"]
    return names[: max(1, min(k, len(names)))]


def _round_share_from_month(data_loader: object, rounds: List[str]) -> List[float]:
    totals = []
    for r in rounds:
        try:
            c = data_loader.get_round_constraints(r)
            totals.append(float(c.get("total_quantity", 0.0) or 0.0))
        except Exception:
            totals.append(0.0)
    s = float(sum(totals))
    if s <= 0:
        return [1.0 / float(len(rounds)) for _ in rounds]
    return [float(x) / s for x in totals]


def _expand_round_share(p_base: List[float], k: int) -> List[float]:
    p = [float(x) for x in (p_base or []) if float(x) >= 0]
    if not p:
        return [1.0 / float(k) for _ in range(k)]

    while len(p) < k:
        last = float(p[-1])
        p = p[:-1] + [0.6 * last, 0.4 * last]

    p = p[:k]
    s = float(sum(p))
    if s <= 0:
        return [1.0 / float(k) for _ in range(k)]
    return [float(x) / s for x in p]


def _build_scaled_instance(
    base_month: str,
    base_product: pd.DataFrame,
    base_round_share: List[float],
    scale: ScaleSpec,
    seed: int,
) -> ScaledInstance:
    rng = np.random.default_rng(int(seed))

    n_base = int(len(base_product))
    n_target = int(math.ceil(float(n_base) * float(scale.sku_scale)))
    n_target = max(1, n_target)

    replace = n_target > n_base
    idx = rng.choice(n_base, size=n_target, replace=replace).tolist()
    df = base_product.iloc[idx, :].copy().reset_index(drop=True)

    if "代码" not in df.columns:
        df["代码"] = [f"{base_month}_SKU{i+1}" for i in range(len(df))]
    df["代码"] = [f"{str(code)}__rep{i}" for i, code in enumerate(df["代码"].astype(str).tolist())]

    if "卷烟名称" not in df.columns:
        df["卷烟名称"] = df["代码"].astype(str)
    else:
        df["卷烟名称"] = df["卷烟名称"].astype(str)

    for col in ["按需", "按价", "C"]:
        if col not in df.columns:
            df[col] = ""

    demand = pd.to_numeric(df["需求"], errors="coerce").fillna(0.0).astype(float) * float(scale.demand_scale)
    demand = demand.round(1)
    demand = demand.where(demand > 0, 0.0)
    df["需求"] = demand

    if "可用货源" in df.columns:
        supply = pd.to_numeric(df["可用货源"], errors="coerce").fillna(0.0).astype(float) * float(scale.demand_scale)
        supply = np.maximum(supply, demand)
        df["可用货源"] = pd.Series(supply).round(1)
    else:
        df["可用货源"] = demand * 10.0

    total_demand = float(pd.to_numeric(df["需求"], errors="coerce").fillna(0.0).sum())
    k = int(scale.rounds)
    rounds = _round_names(k)
    p = _expand_round_share(base_round_share, k)

    targets = [total_demand * float(x) for x in p]
    if targets:
        targets[-1] = max(0.0, total_demand - float(sum(targets[:-1])))

    round_constraints: Dict[str, Dict[str, float]] = {}
    for r, t in zip(rounds, targets, strict=False):
        round_constraints[r] = {
            "total_quantity": float(t),
            "upper_price_limit": 1e9,
            "lower_price_limit": 0.0,
        }

    loader = ToyDataLoader(product_data=df, rounds=rounds, round_constraints=round_constraints)
    label = _build_label(df, rounds, p)

    return ScaledInstance(rounds=rounds, product_data=df, label_data=label, data_loader=loader)


def _build_label(product_data: pd.DataFrame, rounds: List[str], shares: List[float]) -> pd.DataFrame:
    df = product_data.loc[:, ["代码", "卷烟名称", "需求"]].copy()
    demand = pd.to_numeric(df["需求"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    p = np.array(shares, dtype=float)
    p = p / max(1e-9, float(p.sum()))

    alloc = np.zeros((len(df), len(rounds)), dtype=float)
    for i in range(len(df)):
        d = float(demand[i])
        if d <= 0:
            continue
        raw = d * p
        floor = np.floor(raw)
        rem = float(d - float(floor.sum()))
        alloc[i, :] = floor
        if rem > 0:
            frac = raw - floor
            order = np.argsort(-frac)
            k = int(round(rem))
            for j in order[:k]:
                alloc[i, int(j)] += 1.0

    for j, r in enumerate(rounds):
        df[r] = alloc[:, j]
    return df


def _align_by_code(a: pd.DataFrame, b: pd.DataFrame, rounds: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "代码" not in a.columns or "代码" not in b.columns:
        return a.reset_index(drop=True), b.reset_index(drop=True)
    a2 = a.copy()
    b2 = b.copy()
    a2["代码"] = a2["代码"].astype(str)
    b2["代码"] = b2["代码"].astype(str)
    merged = a2.merge(b2, on="代码", how="inner", suffixes=("_a", "_b"))
    out_a = merged.loc[:, ["代码"] + [f"{r}_a" for r in rounds]].rename(columns={f"{r}_a": r for r in rounds})
    out_b = merged.loc[:, ["代码"] + [f"{r}_b" for r in rounds]].rename(columns={f"{r}_b": r for r in rounds})
    return out_a, out_b


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    numeric_cols = ["number_of_skus", "rounds", "demand_scale", "gnae", "rule_deviations", "solve_time_sec"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.groupby("scale", dropna=False)
    mean_df = g[numeric_cols].mean(numeric_only=True).add_suffix("_mean")
    std_df = g[numeric_cols].std(numeric_only=True).fillna(0.0).add_suffix("_std")
    out = mean_df.join(std_df, how="outer").reset_index()
    return out.to_dict("records")


if __name__ == "__main__":
    raise SystemExit(main())
