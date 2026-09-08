from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .bootstrap import ensure_backend_on_path


@dataclass(frozen=True)
class ExperimentConfigs:
    """
    实验使用的约束与目标配置。

    Attributes:
        constraint_config: 传入后端 LP/约束校验器的约束配置字典
        objective_config: 传入后端 LP 的目标权重配置字典
    """

    constraint_config: Dict[str, Any]
    objective_config: Dict[str, Any]


def default_experiment_configs() -> ExperimentConfigs:
    """
    构造默认实验配置，默认值与后端 Pydantic 模型保持一致。

    Returns:
        ExperimentConfigs: 默认配置
    """
    ensure_backend_on_path()
    from models import ConstraintConfig, ObjectiveConfig

    constraints = ConstraintConfig()
    objective = ObjectiveConfig()
    constraint_config = constraints.model_dump() if hasattr(constraints, "model_dump") else constraints.dict()
    objective_config = objective.model_dump() if hasattr(objective, "model_dump") else objective.dict()

    constraint_config["enable_demand_split_constraints"] = False

    objective_config["round_balance_weight"] = 0.0
    objective_config["round_variance_weight"] = 0.0
    objective_config["product_balance_weight"] = 0.0
    objective_config["smooth_transition_weight"] = float(objective_config.get("smooth_transition_weight", 300.0) or 300.0)
    objective_config["maximize_allocation_weight"] = float(objective_config.get("maximize_allocation_weight", 1000.0) or 1000.0)

    return ExperimentConfigs(
        constraint_config=constraint_config,
        objective_config=objective_config,
    )
