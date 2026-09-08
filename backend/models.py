"""
数据模型定义
"""
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime

# 请求模型
class ConstraintConfig(BaseModel):
    """约束配置"""
    enable_demand_constraints: bool = True
    enable_volume_constraints: bool = True
    enable_price_constraints: bool = True
    enable_c_type_constraints: bool = True
    enable_balance_constraints: bool = True
    enable_demand_split_constraints: bool = True      # 需求量集中约束
    enable_demand_based_constraints: bool = True      # 按需优先约束
    enable_price_based_constraints: bool = True       # 按价比例约束
    
    # 容差配置
    volume_tolerance: float = 0.005  # 投放总量容差 ±0.5%
    
    # 可配置的约束参数
    price_upper_limits: Optional[Dict[str, float]] = None  # 各轮单箱均价上限（覆盖dataloader默认值）
    price_lower_limits: Optional[Dict[str, float]] = None  # 各轮单箱均价下限（覆盖dataloader默认值）
    volume_limits: Optional[Dict[str, float]] = None       # 各轮投放总量限制（覆盖dataloader默认值）
    price_based_ratio: float = 0.3                    # 按价品规比例要求（默认30%）
    
    # C类烟配置参数
    c_type_ratio: float = 0.4                       # C类烟比例要求（默认40%）
    c_type_volume_limit: Optional[float] = 4900       # C类烟总量限制
    
    # 长和细类型配置参数
    chang_type_ratio: float = 0.2                   # 长型占C类烟的最大比例（默认20%）
    chang_type_volume_limit: float = 1000           # 长型每轮最大量限制（默认1000箱）
    xi_type_ratio: float = 0.6                      # 细型占C类烟的最大比例（默认60%）
    xi_type_volume_limit: float = 3000              # 细型每轮最大量限制（默认3000箱）

class ObjectiveConfig(BaseModel):
    """目标函数权重配置"""
    maximize_allocation_weight: float = 1000.0      # 最大化总分配量权重
    round_balance_weight: float = 800.0             # 轮次间总量均衡权重
    round_variance_weight: float = 0 #400.0            # 轮次间方差最小化权重
    product_balance_weight: float = 0# 100.0           # 品规级别均衡权重
    smooth_transition_weight: float =0# 300.0         # 轮次间平滑过渡权重

class SolveConfig(BaseModel):
    """求解配置"""
    constraints: ConstraintConfig
    objective: ObjectiveConfig

# 响应模型
class UploadData(BaseModel):
    """上传数据响应"""
    total_products: int
    rounds: List[str]
    round_constraints: Dict[str, Dict[str, float]]
    upload_time: str

class UploadResponse(BaseModel):
    """上传响应"""
    success: bool
    message: str
    data: Optional[UploadData] = None

class SolveResult(BaseModel):
    """求解结果"""
    status: str
    objective_value: Optional[float] = None
    solve_time: float
    total_allocated: float
    constraint_violations: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = {}

class SolveResponse(BaseModel):
    """求解响应"""
    success: bool
    message: str
    data: Optional[SolveResult] = None

class AllocationDetail(BaseModel):
    """分配明细"""
    product_code: str
    product_name: str
    category: str
    demand: float
    wholesale_price: float
    available_supply: float
    total_allocation: float
    allocation_rate: float
    # 动态添加各轮次分配量字段

class RoundSummary(BaseModel):
    """轮次汇总"""
    round_name: str
    total_allocation: float
    average_price: float
    product_count: int

class ResultData(BaseModel):
    """结果数据"""
    allocation_details: List[Dict[str, Any]]
    round_summary: List[RoundSummary]
    total_products: int
    total_allocation: float

class ResultResponse(BaseModel):
    """结果响应"""
    success: bool
    message: str
    data: Optional[ResultData] = None