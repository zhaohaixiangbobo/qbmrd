"""Data-model definitions."""
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

# Request models
class ConstraintConfig(BaseModel):
    """Constraint configuration."""
    enable_demand_constraints: bool = True
    enable_volume_constraints: bool = True
    enable_price_constraints: bool = True
    enable_c_type_constraints: bool = True
    enable_balance_constraints: bool = True
    enable_demand_split_constraints: bool = True      # Demand concentration constraint
    enable_demand_based_constraints: bool = True      # Demand-based priority constraint
    enable_price_based_constraints: bool = True       # Price-based proportion constraint
    
    # Tolerance settings
    volume_tolerance: float = 0.005  # Total allocation tolerance ±0.5%
    
    # Configurable constraint parameters
    price_upper_limits: Optional[Dict[str, float]] = None  # Upper mean price per case in each round (override dataloader defaults)
    price_lower_limits: Optional[Dict[str, float]] = None  # Lower mean price per case in each round (override dataloader defaults)
    volume_limits: Optional[Dict[str, float]] = None       # Total allocation limits for each round (override dataloader defaults)
    price_based_ratio: float = 0.3                    # Required proportion of price-based SKUs (default 30%)
    
    # C-category cigarette settings
    c_type_ratio: float = 0.4                       # Required C-category cigarette proportion (default 40%)
    c_type_volume_limit: Optional[float] = 4900       # Total C-category cigarette allocation limit
    
    # Long and slim subtype settings
    chang_type_ratio: float = 0.2                   # Maximum long-subtype share of C-category cigarettes (default 20%)
    chang_type_volume_limit: float = 1000           # Maximum long-subtype quantity per round (default 1000 cases)
    xi_type_ratio: float = 0.6                      # Maximum slim-subtype share of C-category cigarettes (default 60%)
    xi_type_volume_limit: float = 3000              # Maximum slim-subtype quantity per round (default 3000 cases)

class ObjectiveConfig(BaseModel):
    """Objective-weight configuration."""
    maximize_allocation_weight: float = 1000.0      # Weight for maximising total allocation
    round_balance_weight: float = 800.0             # Weight for balancing total allocation across rounds
    round_variance_weight: float = 0 #400.0            # Weight for minimising variance across rounds
    product_balance_weight: float = 0# 100.0           # Weight for SKU-level balance
    smooth_transition_weight: float =0# 300.0         # Weight for smooth transitions across rounds

class SolveConfig(BaseModel):
    """Solver configuration."""
    constraints: ConstraintConfig
    objective: ObjectiveConfig

# Response models
class UploadData(BaseModel):
    """Uploaded-data response."""
    total_products: int
    rounds: List[str]
    round_constraints: Dict[str, Dict[str, float]]
    upload_time: str

class UploadResponse(BaseModel):
    """Upload response."""
    success: bool
    message: str
    data: Optional[UploadData] = None

class SolveResult(BaseModel):
    """Solver result."""
    status: str
    objective_value: Optional[float] = None
    solve_time: float
    total_allocated: float
    constraint_violations: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = {}

class SolveResponse(BaseModel):
    """Solver response."""
    success: bool
    message: str
    data: Optional[SolveResult] = None

class AllocationDetail(BaseModel):
    """Allocation details."""
    product_code: str
    product_name: str
    category: str
    demand: float
    wholesale_price: float
    available_supply: float
    total_allocation: float
    allocation_rate: float
    # Add allocation fields for each round dynamically

class RoundSummary(BaseModel):
    """Round summary."""
    round_name: str
    total_allocation: float
    average_price: float
    product_count: int

class ResultData(BaseModel):
    """Result data."""
    allocation_details: List[Dict[str, Any]]
    round_summary: List[RoundSummary]
    total_products: int
    total_allocation: float

class ResultResponse(BaseModel):
    """Result response."""
    success: bool
    message: str
    data: Optional[ResultData] = None