"""
卷烟货源分配平台 - FastAPI主应用
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import tempfile
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from models import *
from data_loader import DataLoader
from constraint_manager import ConstraintManager
from linear_programming import LinearProgrammingAllocator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，解决Pydantic序列化问题
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

app = FastAPI(
    title="卷烟货源分配平台",
    description="基于线性规划的卷烟货源分配优化系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储当前会话数据
current_session = {
    "data_loader": None,
    "constraint_manager": None,
    "allocator": None,
    "last_result": None,
    "upload_time": None
}

@app.get("/")
async def root():
    """根路径"""
    return {"message": "卷烟货源分配平台API", "version": "1.0.0"}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    """
    上传Excel文件并解析数据
    """
    try:
        # 验证文件类型
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="请上传Excel文件(.xlsx或.xls)")
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 加载数据
        data_loader = DataLoader(temp_file_path)
        constraint_manager = ConstraintManager(data_loader)
        
        # 更新全局会话
        current_session["data_loader"] = data_loader
        current_session["constraint_manager"] = constraint_manager
        current_session["upload_time"] = datetime.now()
        
        # 获取基本信息
        product_data = data_loader.get_product_data()
        rounds = data_loader.get_rounds()
        
        # 获取轮次约束信息
        round_constraints = {}
        for round_name in rounds:
            try:
                constraints = data_loader.get_round_constraints(round_name)
                round_constraints[round_name] = {
                    "total_quantity": constraints["total_quantity"],
                    "upper_price_limit": constraints["upper_price_limit"],
                    "lower_price_limit": constraints["lower_price_limit"]
                }
            except Exception as e:
                logger.warning(f"无法获取轮次 {round_name} 的约束: {e}")
        
        # 清理临时文件
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        return UploadResponse(
            success=True,
            message="文件上传成功",
            data=UploadData(
                total_products=len(product_data),
                rounds=rounds,
                round_constraints=round_constraints,
                upload_time=current_session["upload_time"].isoformat()
            )
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

@app.post("/api/solve", response_model=SolveResponse)
async def solve_allocation(config: SolveConfig):
    """
    执行货源分配计算
    """
    try:
        logger.info(f"接收到前端配置: {config.json()}")

        if current_session["data_loader"] is None:
            raise HTTPException(status_code=400, detail="请先上传Excel文件")
        
        # 前端请求的约束参数（requested）
        requested_constraint_config = {
            "enable_demand_constraints": config.constraints.enable_demand_constraints,
            "enable_volume_constraints": config.constraints.enable_volume_constraints,
            "enable_price_constraints": config.constraints.enable_price_constraints,
            "enable_c_type_constraints": config.constraints.enable_c_type_constraints,
            "enable_balance_constraints": config.constraints.enable_balance_constraints,
            "enable_demand_split_constraints": config.constraints.enable_demand_split_constraints,
            "enable_demand_based_constraints": config.constraints.enable_demand_based_constraints,
            "enable_price_based_constraints": config.constraints.enable_price_based_constraints,
            "volume_tolerance": config.constraints.volume_tolerance,
            "price_upper_limits": getattr(config.constraints, 'price_upper_limits', None),
            "price_lower_limits": getattr(config.constraints, 'price_lower_limits', None),
            "volume_limits": getattr(config.constraints, 'volume_limits', None),
            "price_based_ratio": getattr(config.constraints, 'price_based_ratio', 0.3),
            "c_type_ratio": getattr(config.constraints, 'c_type_ratio', 0.4),
            "c_type_volume_limit": getattr(config.constraints, 'c_type_volume_limit', 4900),
            "chang_type_ratio": getattr(config.constraints, 'chang_type_ratio', 0.2),
            "chang_type_volume_limit": getattr(config.constraints, 'chang_type_volume_limit',1000),
            "xi_type_ratio": getattr(config.constraints, 'xi_type_ratio', 0.6),
            "xi_type_volume_limit": getattr(config.constraints, 'xi_type_volume_limit', 3000)
        }

        # 目标函数权重
        objective_config = {
            "maximize_allocation_weight": config.objective.maximize_allocation_weight,
            "round_balance_weight": config.objective.round_balance_weight,
            "round_variance_weight": config.objective.round_variance_weight,
            "product_balance_weight": config.objective.product_balance_weight,
            "smooth_transition_weight": config.objective.smooth_transition_weight
        }

        # 实际用于求解的约束参数（applied）：允许后端根据数据做最小必要的防呆调整
        applied_constraint_config = dict(requested_constraint_config)
        auto_adjustments: List[str] = []

        try:
            product_data = current_session["data_loader"].get_product_data()
            if "按价" in product_data.columns:
                price_based_cnt = int(
                    product_data["按价"]
                    .fillna("")
                    .astype(str)
                    .str.contains("价", na=False)
                    .sum()
                )
                if price_based_cnt <= 1 and applied_constraint_config.get("enable_price_based_constraints", True):
                    applied_constraint_config["enable_price_based_constraints"] = False
                    auto_adjustments.append("enable_price_based_constraints=False（按价品规<=1）")
        except Exception:
            pass
        
        def _apply_round_constraint_overrides(constraint_config: Dict[str, Any]) -> None:
            has_constraint_updates = any(
                [
                    constraint_config.get("price_upper_limits") is not None,
                    constraint_config.get("price_lower_limits") is not None,
                    constraint_config.get("volume_limits") is not None,
                ]
            )
            if has_constraint_updates:
                logger.info("检测到约束配置更新，正在更新data_loader中的约束值")
                current_session["data_loader"].update_all_round_constraints(constraint_config)

        def _run_lp(constraint_config: Dict[str, Any]) -> tuple[Dict[str, Any], ConstraintManager, LinearProgrammingAllocator]:
            _apply_round_constraint_overrides(constraint_config)
            cm = ConstraintManager(current_session["data_loader"], constraint_config)
            alloc = LinearProgrammingAllocator(current_session["data_loader"], cm)
            res = alloc.solve(constraint_config, objective_config)
            return res, cm, alloc

        # 执行求解（支持无解回退重试）
        start_time = datetime.now()
        result, constraint_manager, allocator = _run_lp(applied_constraint_config)

        if str(result.get("status")) == "Infeasible":
            if applied_constraint_config.get("enable_demand_split_constraints", True):
                retry_cfg = dict(applied_constraint_config)
                retry_cfg["enable_demand_split_constraints"] = False
                retry_result, retry_cm, retry_alloc = _run_lp(retry_cfg)
                if str(retry_result.get("status")) != "Infeasible":
                    auto_adjustments.append("enable_demand_split_constraints=False（无解自动回退）")
                    applied_constraint_config = retry_cfg
                    result, constraint_manager, allocator = retry_result, retry_cm, retry_alloc

        if str(result.get("status")) == "Infeasible":
            if applied_constraint_config.get("enable_price_based_constraints", True):
                retry_cfg = dict(applied_constraint_config)
                retry_cfg["enable_price_based_constraints"] = False
                retry_result, retry_cm, retry_alloc = _run_lp(retry_cfg)
                if str(retry_result.get("status")) != "Infeasible":
                    auto_adjustments.append("enable_price_based_constraints=False（无解自动回退）")
                    applied_constraint_config = retry_cfg
                    result, constraint_manager, allocator = retry_result, retry_cm, retry_alloc

        solve_time = (datetime.now() - start_time).total_seconds()

        # 保存结果（以最终一次求解为准）
        current_session["constraint_manager"] = constraint_manager
        current_session["allocator"] = allocator
        current_session["last_result"] = result
        
        # 验证约束
        constraint_validation = None
        if result.get("status") == "Optimal":
            allocation_matrix = result["allocation_matrix"]
            constraint_validation = constraint_manager.validate_all_constraints(allocation_matrix[constraint_manager.rounds])
            # 转换numpy类型
            constraint_validation = convert_numpy_types(constraint_validation)
            # 显示验证结果
            logger.info(f"整体验证结果: {'通过' if constraint_validation['overall_valid'] else '失败'}")
            logger.info(f"总违反数: {constraint_validation['summary']['total_violations']}")
            logger.info(f"违反的约束: {constraint_validation['summary']['violated_constraints']}")
            logger.info(f"通过的约束: {constraint_validation['summary']['passed_constraints']}")
        
        # 计算总分配量（只对轮次列求和）
        total_allocated = 0
        if result.get("allocation_matrix") is not None:
            allocation_matrix = result["allocation_matrix"]
            rounds = current_session["data_loader"].get_rounds()
            allocation_cols = [col for col in rounds if col in allocation_matrix.columns]
            if allocation_cols:
                total_allocated = float(allocation_matrix[allocation_cols].sum().sum())
        
        # 转换result中的numpy类型
        converted_result = convert_numpy_types(result)
        
        summary = converted_result.get("summary", {}) or {}
        summary["requested_constraint_config"] = requested_constraint_config
        summary["applied_constraint_config"] = applied_constraint_config
        if auto_adjustments:
            summary["auto_adjustments"] = auto_adjustments

        return SolveResponse(
            success=True,
            message="计算完成",
            data=SolveResult(
                status=converted_result["status"],
                objective_value=converted_result.get("objective_value"),
                solve_time=solve_time,
                total_allocated=total_allocated,
                constraint_violations=constraint_validation,
                summary=summary
            )
        )
        
    except Exception as e:
        logger.error(f"求解失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@app.get("/api/result", response_model=ResultResponse)
async def get_result():
    """
    获取最新的计算结果
    """
    try:
        if current_session["last_result"] is None:
            raise HTTPException(status_code=400, detail="暂无计算结果")
        
        result = current_session["last_result"]
        allocation_matrix = result["allocation_matrix"]
        
        if allocation_matrix is None:
            raise HTTPException(status_code=400, detail="分配矩阵为空")
        
        # 转换为前端需要的格式
        product_data = current_session["data_loader"].get_product_data()
        rounds = current_session["data_loader"].get_rounds()
        
        # 构建分配明细数据
        allocation_details = []
        for idx, row in product_data.iterrows():
            detail = {
                "product_code": row["代码"],
                "product_name": row["卷烟名称"],
                "category": row["类"],
                "demand": float(row["需求"]),
                "wholesale_price": float(row["批发价"]),
                "available_supply": float(row["可用货源"]),
                # 添加output4.xlsx中的其他字段
                "attribute": row.get("属", ""),
                "c_category": row.get("C类", ""),
                "brand": row.get("品牌", ""),
                "stick_ratio": row.get("条支比", ""),
                "c_type": row.get("C", ""),
                "demand_based": row.get("按需", ""),
                "price_based": row.get("按价", "")
            }
            
            # 添加各轮次分配量
            for round_name in rounds:
                if round_name in allocation_matrix.columns:
                    detail[f"allocation_{round_name}"] = float(allocation_matrix.loc[idx, round_name])
                else:
                    detail[f"allocation_{round_name}"] = 0.0
            
            # 计算总分配量
            detail["total_allocation"] = sum([detail[f"allocation_{round_name}"] for round_name in rounds])
            detail["allocation_rate"] = detail["total_allocation"] / detail["demand"] if detail["demand"] > 0 else 0
            
            # 计算单箱价格：批发价 * 50000 / 条支比
            stick_ratio = row.get("条支比", 1)
            if stick_ratio and stick_ratio != 0:
                detail["unit_price"] = float((row["批发价"] * 50000) / stick_ratio)
            else:
                detail["unit_price"] = 0.0
            
            allocation_details.append(detail)
        
        # 构建轮次汇总数据
        round_summary = []
        for round_name in rounds:
            if round_name in allocation_matrix.columns:
                round_total = float(allocation_matrix[round_name].sum())
                
                # 计算加权平均价格
                weighted_price = 0
                if round_total > 0:
                    for idx, row in product_data.iterrows():
                        allocation = allocation_matrix.loc[idx, round_name]
                        if allocation > 0:
                            weighted_price += allocation * row["批发价"]
                    weighted_price = weighted_price / round_total
                
                round_summary.append({
                    "round_name": round_name,
                    "total_allocation": round_total,
                    "average_price": float(weighted_price),
                    "product_count": int((allocation_matrix[round_name] > 0).sum())
                })
        
        # 计算总分配量（只对轮次列求和）
        total_allocation = 0
        allocation_cols = [col for col in rounds if col in allocation_matrix.columns]
        if allocation_cols:
            total_allocation = float(allocation_matrix[allocation_cols].sum().sum())
        
        return ResultResponse(
            success=True,
            message="获取结果成功",
            data=ResultData(
                allocation_details=allocation_details,
                round_summary=round_summary,
                total_products=len(allocation_details),
                total_allocation=total_allocation
            )
        )
        
    except Exception as e:
        logger.error(f"获取结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取结果失败: {str(e)}")

@app.get("/api/constraints")
async def get_constraints():
    """
    获取约束验证结果
    """
    try:
        if current_session["last_result"] is None:
            raise HTTPException(status_code=400, detail="暂无计算结果")
        
        result = current_session["last_result"]
        allocation_matrix = result["allocation_matrix"]
        
        if allocation_matrix is None:
            raise HTTPException(status_code=400, detail="分配矩阵为空")
        
        constraint_manager = current_session["constraint_manager"]
        constraint_validation = constraint_manager.validate_all_constraints(allocation_matrix[constraint_manager.rounds])
        # 转换numpy类型
        validations = convert_numpy_types(constraint_validation)
        
        return {
            "success": True,
            "message": "约束验证完成",
            "data": validations
        }
        
    except Exception as e:
        logger.error(f"约束验证失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"约束验证失败: {str(e)}")

@app.get("/api/download-template")
async def download_template():
    """
    下载Excel模板文件
    """
    try:
        template_path = "d:/2-code/huoyuan/web_huoyuan/huoyuanfenpei.xlsx"
        
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail="模板文件不存在")
        
        return FileResponse(
            path=template_path,
            filename="huoyuanfenpei.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        logger.error(f"下载模板失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载模板失败: {str(e)}")

@app.get("/api/export")
async def export_result(format: str = "xlsx"):
    """
    导出计算结果
    """
    try:
        if current_session["last_result"] is None:
            raise HTTPException(status_code=400, detail="暂无计算结果")
        
        result = current_session["last_result"]
        allocation_matrix = result["allocation_matrix"]
        
        if allocation_matrix is None:
            raise HTTPException(status_code=400, detail="分配矩阵为空")
        
        # 准备导出数据
        product_data = current_session["data_loader"].get_product_data()
        export_data = product_data.copy()
        
        # 添加分配结果
        rounds = current_session["data_loader"].get_rounds()
        allocation_cols = []
        for round_name in rounds:
            if round_name in allocation_matrix.columns:
                export_data[round_name] = allocation_matrix[round_name].astype(float)
                allocation_cols.append(round_name)
        
        # 添加汇总列 - 确保数据类型正确
        if allocation_cols:
            export_data["总分配量"] = export_data[allocation_cols].sum(axis=1).astype(float)
            # 避免除零错误
            demand_values = pd.to_numeric(export_data["需求"], errors='coerce').fillna(0)
            export_data["分配率"] = export_data["总分配量"] / demand_values.replace(0, 1)  # 避免除零
        else:
            export_data["总分配量"] = 0.0
            export_data["分配率"] = 0.0
        
        # 生成临时文件
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "xlsx":
            filename = f"allocation_result_{timestamp}.xlsx"
            filepath = os.path.join(temp_dir, filename)
            export_data.to_excel(filepath, index=False)
        else:
            filename = f"allocation_result_{timestamp}.csv"
            filepath = os.path.join(temp_dir, filename)
            export_data.to_csv(filepath, index=False, encoding="utf-8-sig")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"导出失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

@app.get("/api/export-statistics")
async def export_statistics(format: str = "xlsx"):
    """
    导出分配结果统计表
    """
    try:
        if current_session["last_result"] is None:
            raise HTTPException(status_code=400, detail="暂无计算结果")
        
        result = current_session["last_result"]
        allocation_matrix = result["allocation_matrix"]
        
        if allocation_matrix is None:
            raise HTTPException(status_code=400, detail="分配矩阵为空")
        
        # 获取分配详情数据
        product_data = current_session["data_loader"].get_product_data()
        rounds = current_session["data_loader"].get_rounds()
        
        # 构建分配详情数据
        allocation_details = []
        for idx, row in product_data.iterrows():
            # 安全获取字符串字段，处理可能的NaN或非字符串值
            c_type_val = row.get("C", "")
            c_category_val = row.get("C类", "")
            
            # 确保字符串类型
            if pd.isna(c_type_val):
                c_type_val = ""
            else:
                c_type_val = str(c_type_val)
                
            if pd.isna(c_category_val):
                c_category_val = ""
            else:
                c_category_val = str(c_category_val)
            
            detail = {
                "product_code": row["代码"],
                "product_name": row["卷烟名称"],
                "category": row["类"],
                "demand": float(row["需求"]),
                "wholesale_price": float(row["批发价"]),
                "c_type": c_type_val,
                "c_category": c_category_val,
            }
            
            # 添加各轮次分配量
            for round_name in rounds:
                if round_name in allocation_matrix.columns:
                    detail[f"allocation_{round_name}"] = float(allocation_matrix.loc[idx, round_name])
                else:
                    detail[f"allocation_{round_name}"] = 0.0
            
            # 计算总分配量
            detail["total_allocation"] = sum([detail[f"allocation_{round_name}"] for round_name in rounds])
            
            # 计算单箱价格：批发价 * 50000 / 条支比
            stick_ratio = row.get("条支比", 1)
            if stick_ratio and stick_ratio != 0:
                detail["unit_price"] = float((row["批发价"] * 50000) / stick_ratio)
            else:
                detail["unit_price"] = 0.0
            
            allocation_details.append(detail)
        
        # 计算统计数据
        stats = []
        
        # 1. 单箱（单箱均价）
        unit_price_row = {"指标": "单箱"}
        for round_name in rounds:
            round_key = f"allocation_{round_name}"
            total_allocation = sum(item[round_key] for item in allocation_details)
            total_value = sum(item[round_key] * item["unit_price"] for item in allocation_details)
            unit_price_row[round_name] = round(total_value / total_allocation, 1) if total_allocation > 0 else 0.0
        
        # 总计单箱价格
        total_allocation_all = sum(item["total_allocation"] for item in allocation_details)
        total_value_all = sum(item["total_allocation"] * item["unit_price"] for item in allocation_details)
        unit_price_row["合计"] = round(total_value_all / total_allocation_all, 1) if total_allocation_all > 0 else 0.0
        stats.append(unit_price_row)
        
        # 2. 总分配量
        total_allocation_row = {"指标": "总量"}
        for round_name in rounds:
            round_key = f"allocation_{round_name}"
            round_total = sum(item[round_key] for item in allocation_details)
            total_allocation_row[round_name] = round(round_total, 2)
        total_allocation_row["合计"] = round(sum(item["total_allocation"] for item in allocation_details), 2)
        stats.append(total_allocation_row)
        
        # 3. 总数（投放大于0的品规数量）
        total_count_row = {"指标": "总数"}
        for round_name in rounds:
            round_key = f"allocation_{round_name}"
            count = sum(1 for item in allocation_details if item[round_key] > 0)
            total_count_row[round_name] = count
        total_count_row["合计"] = sum(1 for item in allocation_details if item["total_allocation"] > 0)
        stats.append(total_count_row)
        
        # 4. C类量
        c_category_row = {"指标": "C类量"}
        for round_name in rounds:
            round_key = f"allocation_{round_name}"
            c_total = sum(item[round_key] for item in allocation_details if item["c_type"] and str(item["c_type"]).strip())
            c_category_row[round_name] = round(c_total, 2)
        c_category_row["合计"] = round(sum(item["total_allocation"] for item in allocation_details if item["c_type"] and str(item["c_type"]).strip()), 2)
        stats.append(c_category_row)
        
        # 5. C类占比
        c_ratio_row = {"指标": "占比"}
        for round_name in rounds:
            round_total = total_allocation_row[round_name]
            c_total = c_category_row[round_name]
            c_ratio_row[round_name] = f"{round(c_total / round_total * 100, 1)}%" if round_total > 0 else "0.0%"
        total_total = total_allocation_row["合计"]
        c_total_all = c_category_row["合计"]
        c_ratio_row["合计"] = f"{round(c_total_all / total_total * 100, 1)}%" if total_total > 0 else "0.0%"
        stats.append(c_ratio_row)
        
        # 6. C类数
        c_count_row = {"指标": "C类数"}
        for round_name in rounds:
            round_key = f"allocation_{round_name}"
            count = sum(1 for item in allocation_details if item["c_type"] and str(item["c_type"]).strip() and item[round_key] > 0)
            c_count_row[round_name] = count
        c_count_row["合计"] = sum(1 for item in allocation_details if item["c_type"] and str(item["c_type"]).strip() and item["total_allocation"] > 0)
        stats.append(c_count_row)
        
        # 7-9. 方块、长片、细支
        categories = [
            {"name": "方块", "match": "方"},
            {"name": "长片", "match": "长"},
            {"name": "细支", "match": "细"}
        ]
        
        for category in categories:
            # 量
            category_amount_row = {"指标": f"{category['name']}量"}
            for round_name in rounds:
                round_key = f"allocation_{round_name}"
                category_total = sum(item[round_key] for item in allocation_details 
                                   if item["c_category"] and category["match"] in str(item["c_category"]))
                category_amount_row[round_name] = round(category_total, 2)
            category_amount_row["合计"] = round(sum(item["total_allocation"] for item in allocation_details 
                                               if item["c_category"] and category["match"] in str(item["c_category"])), 2)
            stats.append(category_amount_row)
            
            # 数量
            category_count_row = {"指标": f"{category['name']}数"}
            for round_name in rounds:
                round_key = f"allocation_{round_name}"
                count = sum(1 for item in allocation_details 
                          if item["c_category"] and category["match"] in str(item["c_category"]) and item[round_key] > 0)
                category_count_row[round_name] = count
            category_count_row["合计"] = sum(1 for item in allocation_details 
                                         if item["c_category"] and category["match"] in str(item["c_category"]) and item["total_allocation"] > 0)
            stats.append(category_count_row)
            
            # 占比（占C类的占比）
            category_ratio_row = {"指标": f"{category['name']}占比"}
            for round_name in rounds:
                c_total = c_category_row[round_name]
                category_total = category_amount_row[round_name]
                category_ratio_row[round_name] = f"{round(category_total / c_total * 100, 1)}%" if c_total > 0 else "0.0%"
            c_total_all = c_category_row["合计"]
            category_total_all = category_amount_row["合计"]
            category_ratio_row["合计"] = f"{round(category_total_all / c_total_all * 100, 1)}%" if c_total_all > 0 else "0.0%"
            stats.append(category_ratio_row)
        
        # 转换为DataFrame
        stats_df = pd.DataFrame(stats)
        
        # 生成临时文件
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "xlsx":
            filename = f"allocation_statistics_{timestamp}.xlsx"
            filepath = os.path.join(temp_dir, filename)
            stats_df.to_excel(filepath, index=False)
        else:
            filename = f"allocation_statistics_{timestamp}.csv"
            filepath = os.path.join(temp_dir, filename)
            stats_df.to_csv(filepath, index=False, encoding="utf-8-sig")
        
        return FileResponse(
            path=filepath,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"导出统计表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出统计表失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
