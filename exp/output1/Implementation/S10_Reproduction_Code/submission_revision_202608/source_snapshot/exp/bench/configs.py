from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .bootstrap import ensure_backend_on_path


@dataclass(frozen=True)
class ExperimentConfigs:
    """Experimental constraint and objective configurations.

Attributes:
    constraint_config: Configuration passed to the backend LP and validator.
    objective_config: Objective weights passed to the backend LP."""

    constraint_config: Dict[str, Any]
    objective_config: Dict[str, Any]


def default_experiment_configs() -> ExperimentConfigs:
    """Construct default experimental configurations matching the backend Pydantic defaults.

Returns:
    ExperimentConfigs: Default configurations."""
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
