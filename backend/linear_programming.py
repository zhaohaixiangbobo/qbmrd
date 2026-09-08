"""
卷烟货源分配算法系统 - 线性规划算法模块

该模块实现基于PuLP的线性规划算法，用于求解卷烟货源分配问题。
主要处理线性约束条件，通过线性目标函数优化分配方案。
"""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LinearProgrammingAllocator:
    """
    线性规划分配器类
    
    使用PuLP库实现线性规划算法，求解卷烟货源分配问题
    """
    
    def __init__(self, data_loader, constraint_manager=None):
        """
        初始化线性规划分配器
        
        Args:
            data_loader: 数据加载器实例
            constraint_manager: 约束管理器实例（可选）
        """
        self.data_loader = data_loader
        self.constraint_manager = constraint_manager
        self.product_data = data_loader.get_product_data()
        self.rounds = data_loader.get_rounds()
        
        # 获取已有分配数据
        self.existing_allocations = data_loader.get_existing_allocations()
        
        # 线性规划模型
        self.model = None
        self.variables = {}
        self.solution = None
        
    def create_variables(self) -> Dict[str, Dict[int, pulp.LpVariable]]:
        """
        创建决策变量
        
        Returns:
            Dict[str, Dict[int, pulp.LpVariable]]: 决策变量字典
        """
        variables = {}
        
        for round_name in self.rounds:
            variables[round_name] = {}
            
            for idx in range(len(self.product_data)):
                product_name = self.product_data.iloc[idx]['卷烟名称']
                demand = self.product_data.iloc[idx]['需求']
                supply = self.product_data.iloc[idx]['可用货源']
                
                # 创建非负连续变量
                var_name = f"{round_name}_{idx}_{product_name}"
                
                # 约束4：已确定的投放品规不能更改
                if round_name in self.existing_allocations and idx in self.existing_allocations[round_name].index:
                    fixed_value = self.existing_allocations[round_name][idx]
                    variables[round_name][idx] = pulp.LpVariable(
                        var_name,
                        lowBound=fixed_value,
                        upBound=fixed_value,  # 固定值
                        cat='Continuous'
                    )
                else:
                    # 约束7：第一轮不超过可用货源上限
                    upper_bound = min(demand, supply if round_name == '第一轮' else demand)
                    variables[round_name][idx] = pulp.LpVariable(
                        var_name,
                        lowBound=0,
                        upBound=upper_bound,
                        cat='Continuous'
                    )
        
        logger.info(f"创建了{len(self.rounds) * len(self.product_data)}个决策变量")
        return variables
    
    def add_demand_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]],
                             constraint_config: Dict[str, Any] = None) -> None:
        """
        添加需求满足约束
        
        约束1: 各品规在所有轮的投放量之和 = 需求量
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['需求']
            
            if demand > 0:
                # 该品规在所有轮的分配量之和不超过需求
                total_allocation = pulp.lpSum([
                    variables[round_name][idx] 
                    for round_name in self.rounds 
                    if round_name in variables
                ])
                
                model += total_allocation == demand, f"Demand_Product_{idx}"
        
        logger.info("添加需求满足约束完成")
    
    def add_volume_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]], 
                             constraint_config: Dict[str, Any] = None) -> None:
        """
        添加投放总量约束
        
        约束3: 每轮投放总量在约束范围内
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # 该轮总分配量
            total_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in range(len(self.product_data))
            ])
            
            # 获取约束条件
            try:
                # 优先使用轮次级别的约束参数
                if (constraint_config and 
                    constraint_config.get('volume_limits') and 
                    round_name in constraint_config['volume_limits']):
                    total_quantity = constraint_config['volume_limits'][round_name]
                else:
                    constraints = self.data_loader.get_round_constraints(round_name)
                    total_quantity = constraints['total_quantity']
                
                # 获取容差配置
                tolerance = constraint_config.get('volume_tolerance', 0.005) if constraint_config else 0.005
                
                # 添加约束（允许配置的容差浮动）
                upper_limit = total_quantity * (1 + tolerance)
                lower_limit = total_quantity * (1 - tolerance)
                
                model += total_round_allocation <= upper_limit, f"Volume_Upper_{round_name}"
                model += total_round_allocation >= lower_limit, f"Volume_Lower_{round_name}"
                
                logger.info(f"{round_name}投放总量约束: {lower_limit:.2f} <= 总量 <= {upper_limit:.2f} (目标: {total_quantity:.2f})")
                
            except Exception as e:
                logger.warning(f"无法获取{round_name}的约束条件: {e}")
                continue
        
        logger.info("添加投放总量约束完成")
    
    def add_c_type_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]],
                             constraint_config: Dict[str, Any] = None) -> None:
        """
        添加C类烟约束（线性部分）
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        # 确保'C'列为字符串类型，并判断C类烟
        type_column = self.product_data['C'].astype(str)
        c_type_mask = type_column.str.contains('C', na=False)
        c_type_indices = c_type_mask[c_type_mask].index.tolist()
        
        if not c_type_indices:
            logger.info("未发现C类烟数据，跳过C类烟约束")
            return
        
        # 获取C类烟配置
        c_type_ratio = 0.4  # 默认40%
        c_type_volume_limit = 4900  # 默认4900箱
        
        if constraint_config:
            c_type_ratio = constraint_config.get('c_type_ratio', 0.4)
            c_type_volume_limit = constraint_config.get('c_type_volume_limit', 4900)
        
        # 约束10.1: 每轮C类烟不得超过总量的配置比例
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # 该轮C类分配
            c_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in c_type_indices
            ])
            
            # 该轮总分配
            total_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in range(len(self.product_data))
            ])
            
            # C类占比不超过配置比例的线性化约束
            model += (c_round_allocation <= c_type_ratio * total_round_allocation), f"C_Type_Ratio_{round_name}"
            
            # 每轮C类烟总量不得超过配置限制
            model += c_round_allocation <= c_type_volume_limit, f"C_Type_Round_Limit_{round_name}"
        
        logger.info(f"添加C类烟约束完成 (比例: {c_type_ratio*100:.1f}%, 量限制: {c_type_volume_limit}箱)")
    
    def _add_c_subtype_constraints(self, model: pulp.LpProblem, 
                                 variables: Dict[str, Dict[int, pulp.LpVariable]],
                                 constraint_config: Dict[str, Any] = None) -> None:
        """
        添加C类烟子类型约束
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        # 获取配置参数
        chang_type_ratio = 0.2  # 默认20%
        chang_type_volume_limit = 1000  # 默认1000箱
        xi_type_ratio = 0.6  # 默认60%
        xi_type_volume_limit = 3000  # 默认3000箱
        
        if constraint_config:
            chang_type_ratio = constraint_config.get('chang_type_ratio', chang_type_ratio)
            chang_type_volume_limit = constraint_config.get('chang_type_volume_limit', chang_type_volume_limit)
            xi_type_ratio = constraint_config.get('xi_type_ratio', xi_type_ratio)
            xi_type_volume_limit = constraint_config.get('xi_type_volume_limit', xi_type_volume_limit)
        
        # 获取各子类型索引
        fang_indices = self.product_data[self.product_data['C类'] == '方'].index.tolist()
        chang_indices = self.product_data[self.product_data['C类'] == '长'].index.tolist()
        xi_indices = self.product_data[self.product_data['C类'] == '细'].index.tolist()
        
        # 约束10.4: "长"型每轮≤配置限制，且每轮不超过C类总投放的配置比例
        if chang_indices:
            # 每轮长型不超过配置限制
            # 每轮长型不超过C类总投放的配置比例
            for round_name in self.rounds:
                if round_name not in variables:
                    continue
                    
                round_chang_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in chang_indices
                ])
                
                round_c_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in range(len(self.product_data))
                    if str(self.product_data.iloc[idx]['C']) == 'C'
                ])
                
                model += round_chang_allocation <= chang_type_ratio * round_c_allocation, f"Chang_Type_Ratio_{round_name}"
                model += round_chang_allocation <= chang_type_volume_limit, f"Chang_Type_Round_Limit_{round_name}"

        
        # 约束10.5: "细"型总量≤配置限制，且每轮不超过C类总投放的配置比例
        if xi_indices:
            # 每轮细型不超过配置限制
            # 每轮细型不超过C类总投放的配置比例
            for round_name in self.rounds:
                if round_name not in variables:
                    continue
                    
                round_xi_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in xi_indices
                ])
                
                round_c_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in range(len(self.product_data))
                    if str(self.product_data.iloc[idx]['C']) == 'C'
                ])
                
                model += round_xi_allocation <= xi_type_ratio * round_c_allocation, f"Xi_Type_Ratio_{round_name}"
                model += round_xi_allocation <= xi_type_volume_limit, f"Xi_Type_Round_Limit_{round_name}"

        # 约束10.3: "方"型集中一轮投放（排除existing_allocations数据）
        # 方型基本都是cx 都是已经分配过的
        if fang_indices:
            # 过滤出非existing_allocations的方型品规
            non_existing_fang_indices = []
            for idx in fang_indices:
                is_existing = False
                for round_name in self.rounds:
                    if (round_name in self.existing_allocations and 
                        idx in self.existing_allocations[round_name].index and
                        self.existing_allocations[round_name][idx] > 0):
                        is_existing = True
                        break
                if not is_existing:
                    non_existing_fang_indices.append(idx)
            
            # 只对非existing_allocations的方型品规应用集中投放约束
            if non_existing_fang_indices:
                # 为每轮创建二进制变量，表示是否在该轮投放方型
                round_binary = {}
                for round_name in self.rounds:
                    if round_name not in variables:
                        continue
                    round_binary[round_name] = pulp.LpVariable(
                        f"Fang_Round_{round_name}",
                        cat='Binary'
                    )
            
                # 只能在一轮投放
                model += pulp.lpSum(round_binary.values()) == 1, "Fang_Single_Round"
            
                # 如果该轮不投放，则所有方型变量为0
                M = 1000000  # 一个足够大的数
                for round_name in self.rounds:
                    if round_name not in variables:
                        continue
                    round_fang_allocation = pulp.lpSum([
                        variables[round_name][idx]
                        for idx in non_existing_fang_indices
                    ])
                    model += round_fang_allocation <= M * round_binary[round_name], f"Fang_Round_Link_{round_name}"

        
    def add_price_based_constraints(self, model: pulp.LpProblem,
                                   variables: Dict[str, Dict[int, pulp.LpVariable]],
                                   constraint_config: Dict[str, Any] = None) -> None:
        """
        添加按价品规约束
        
        约束9: "按价"列含"价"的品规在各轮中需占品规总数 ≥ 配置比例
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        # 获取按价品规索引
        price_based_indices = self.product_data[self.product_data['按价'].str.contains('价', na=False)].index.tolist()
        
        if not price_based_indices:
            logger.info("未发现按价品规数据，跳过按价品规约束")
            return
        
        # 获取按价比例配置
        price_based_ratio = 0.3  # 默认30%
        if constraint_config and constraint_config.get('price_based_ratio') is not None:
            price_based_ratio = constraint_config['price_based_ratio']
        
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # 创建二进制变量，表示每个品规在该轮是否投放（投放量>0）
            allocation_binary = {}
            for idx in range(len(self.product_data)):
                allocation_binary[idx] = pulp.LpVariable(
                    f"Allocation_Binary_{round_name}_{idx}",
                    cat='Binary'
                )
                # 如果投放量>0，则二进制变量=1
                M = 1000000  # 一个足够大的数
                model += variables[round_name][idx] <= M * allocation_binary[idx], f"Binary_Link_Upper_{round_name}_{idx}"
                model += variables[round_name][idx] >= 1.0 * allocation_binary[idx], f"Binary_Link_Lower_{round_name}_{idx}"
            
            # 按价品规数量
            price_based_count = pulp.lpSum([allocation_binary[idx] for idx in price_based_indices])
            # 总投放品规数量
            total_count = pulp.lpSum(allocation_binary.values())
            
            # 按价品规占比≥配置比例
            model += price_based_count >= price_based_ratio * total_count, f"Price_Based_Ratio_{round_name}"
        
        logger.info(f"添加按价品规约束完成 (比例: {price_based_ratio*100:.1f}%)")
    
    def add_demand_based_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """
        添加按需品规约束（软约束）
        
        约束8: "按需"列含"需"字的品规应优先投放在前两轮
        改为软约束：越往后投放惩罚越大
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
        """
        # 获取按需品规索引
        demand_based_indices = self.product_data[self.product_data['按需'].str.contains('需', na=False)].index.tolist()
        
        if not demand_based_indices:
            logger.info("未发现按需品规数据，跳过按需品规约束")
            return
        
        # 为每个按需品规在后续轮次创建惩罚变量
        self.demand_penalty_vars = {}
        for idx in demand_based_indices:
            self.demand_penalty_vars[idx] = {}
            for i, round_name in enumerate(self.rounds[2:], start=3):  # 从第三轮开始
                if round_name in variables:
                    # 创建惩罚变量，表示在该轮的分配量
                    penalty_var = pulp.LpVariable(
                        f"Demand_Penalty_{idx}_{round_name}",
                        lowBound=0,
                        cat='Continuous'
                    )
                    self.demand_penalty_vars[idx][round_name] = penalty_var
                    
                    # 惩罚变量等于该轮的分配量
                    model += penalty_var == variables[round_name][idx], f"Demand_Penalty_Link_{idx}_{round_name}"
        
        logger.info("添加按需品规软约束完成")
    
    def add_demand_split_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """
        添加需求量分轮约束（仅针对非existing_allocations的数据）
        
        约束6: 需求量<50箱的尽量集中分在一轮；50-100的最多分2轮；100-250的至少分2轮
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
        """
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['需求']
            if demand <= 0:
                continue
            
            # 检查是否为已确定分配的品规
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] >= 0):
                    is_existing = True
                    break
            
            # 如果是已确定分配的品规，跳过此约束
            if is_existing:
                continue
            
            # 创建二进制变量，表示每轮是否有分配
            round_binary = {}
            for round_name in self.rounds:
                if round_name not in variables:
                    continue
                round_binary[round_name] = pulp.LpVariable(
                    f"Round_Binary_{idx}_{round_name}",
                    cat='Binary'
                )
                # 如果该轮有分配，则二进制变量=1
                M = 1000000  # 一个足够大的数
                model += variables[round_name][idx] <= M * round_binary[round_name], f"Round_Binary_Link_Upper_{idx}_{round_name}"
                model += variables[round_name][idx] >= 0.01 * round_binary[round_name], f"Round_Binary_Link_Lower_{idx}_{round_name}"
            
            # 计算使用的轮数
            rounds_used = pulp.lpSum(round_binary.values())
            
            if demand < 50:
                # 尽量集中在一轮，使用软约束
                model += rounds_used <= 2, f"Small_Demand_Split_{idx}"
            elif demand <= 100:
                # 最多分2轮
                model += rounds_used <= 2, f"Medium_Demand_Split_{idx}"
            elif demand <= 250:
                # 至少分2轮
                model += rounds_used >= 2, f"Large_Demand_Split_{idx}"
        
        logger.info("添加需求量分轮约束完成（仅针对非existing_allocations数据）")
    
    def add_average_price_constraints(self, model: pulp.LpProblem,
                                     variables: Dict[str, Dict[int, pulp.LpVariable]], 
                                     constraint_config: Dict[str, Any] = None) -> None:
        """
        添加单箱均价约束
        
        约束2: 每轮的单箱均价应在Sheet2中指定的上下限区间内
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
            constraint_config (Dict): 约束配置，可覆盖默认值
        """
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # 获取该轮的约束条件
            try:
                constraints = self.data_loader.get_round_constraints(round_name)
                
                # 使用配置参数覆盖默认值
                # 优先使用轮次级别的约束参数
                if (constraint_config and 
                    constraint_config.get('price_upper_limits') and 
                    round_name in constraint_config['price_upper_limits']):
                    upper_price_limit = constraint_config['price_upper_limits'][round_name]
                elif constraint_config and constraint_config.get('price_upper_limit') is not None:
                    upper_price_limit = constraint_config['price_upper_limit']
                else:
                    upper_price_limit = constraints['upper_price_limit']
                    
                if (constraint_config and 
                    constraint_config.get('price_lower_limits') and 
                    round_name in constraint_config['price_lower_limits']):
                    lower_price_limit = constraint_config['price_lower_limits'][round_name]
                elif constraint_config and constraint_config.get('price_lower_limit') is not None:
                    lower_price_limit = constraint_config['price_lower_limit']
                else:
                    lower_price_limit = constraints['lower_price_limit']
                
                # 计算该轮总销售额
                total_sales = pulp.lpSum([
                    variables[round_name][idx] * self.product_data.iloc[idx]['批发价'] * 50000 / self.product_data.iloc[idx]['条支比']
                    for idx in range(len(self.product_data))
                ])
                
                # 计算该轮总投放量
                total_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in range(len(self.product_data))
                ])
                # 添加单箱均价约束
                model += total_sales <= upper_price_limit * total_allocation, f"Average_Price_Upper_{round_name}"
                model += total_sales >= lower_price_limit * total_allocation, f"Average_Price_Lower_{round_name}"
                
                logger.info(f"{round_name}单箱均价约束: {lower_price_limit:.2f} <= 均价 <= {upper_price_limit:.2f}")
                
            except Exception as e:
                logger.warning(f"无法获取{round_name}的约束条件: {e}")
                continue
    
    def add_first_round_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """
        添加第一轮约束
        
        约束7: 第一轮总投放量不得超过可用货源上限
        注意：该约束已在创建变量时通过设置上界实现，此处仅作为文档说明
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
        """
        pass
    
    def add_balance_constraints(self, model: pulp.LpProblem, 
                                variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """
        添加分配均衡约束（软约束）
        
        约束11：分配均衡，避免单轮过度集中或稀疏
        - 相邻轮次分配量差异不超过20%（改为软约束）
        
        Args:
            model (pulp.LpProblem): 线性规划模型
            variables (Dict): 决策变量字典
        """
        # 1. 创建辅助变量表示各轮次的分配总量
        round_totals = {}
        for round_name in self.rounds:
            if round_name in variables:
                round_totals[round_name] = pulp.lpSum([variables[round_name][idx] for idx in range(len(self.product_data))])
        
        # 2. 创建软约束惩罚变量
        self.balance_penalty_vars = {}
        for i in range(len(self.rounds)-1):
            round_name = self.rounds[i]
            next_round = self.rounds[i+1]
            if round_name in round_totals and next_round in round_totals:
                # 创建正向和负向偏差变量
                pos_deviation = pulp.LpVariable(
                    f"Balance_Pos_Dev_{round_name}_{next_round}",
                    lowBound=0,
                    cat='Continuous'
                )
                neg_deviation = pulp.LpVariable(
                    f"Balance_Neg_Dev_{round_name}_{next_round}",
                    lowBound=0,
                    cat='Continuous'
                )
                
                self.balance_penalty_vars[f"{round_name}_{next_round}"] = {
                    'pos': pos_deviation,
                    'neg': neg_deviation
                }
                
                # 软约束：相邻轮次分配量差异
                # round_totals[round_name] - 1.2 * round_totals[next_round] <= pos_deviation
                model += round_totals[round_name] - 1.2 * round_totals[next_round] <= pos_deviation, f"Balance_Upper_Soft_{round_name}_{next_round}"
                
                # round_totals[round_name] - 0.8 * round_totals[next_round] >= -neg_deviation
                model += round_totals[round_name] - 0.8 * round_totals[next_round] >= -neg_deviation, f"Balance_Lower_Soft_{round_name}_{next_round}"
        
        logger.info("添加分配均衡软约束完成")
    
    def create_objective_function(self, variables: Dict[str, Dict[int, pulp.LpVariable]], objective_config: Dict[str, Any] = None) -> pulp.LpAffineExpression:
        """
        创建目标函数
        
        支持多种优化目标的权重配置：
        1. 最大化总分配量
        2. 轮次间总量均衡
        3. 轮次间方差最小化
        4. 品规级别均衡
        5. 轮次间平滑过渡
        
        Args:
            variables (Dict): 决策变量字典
            objective_config (Dict): 目标函数配置
            
        Returns:
            pulp.LpAffineExpression: 目标函数表达式
        """
        objective = 0
        
        # 使用默认配置如果没有提供
        if objective_config is None:
            objective_config = {
                "maximize_allocation_weight": 1000.0,      # 最大化总分配量权重
                "round_balance_weight": 800.0,             # 轮次间总量均衡权重
                "round_variance_weight": 400.0,            # 轮次间方差最小化权重
                "product_balance_weight": 100.0,           # 品规级别均衡权重
                "smooth_transition_weight": 300.0          # 轮次间平滑过渡权重
            }
        
        logger.info(f"目标函数配置: {objective_config}")
        
        # 1. 最大化总分配量（尽量满足需求）
        if objective_config.get("maximize_allocation_weight", 0) > 0:
            total_allocation = pulp.lpSum([
                variables[round_name][idx]
                for round_name in self.rounds
                for idx in range(len(self.product_data))
                if round_name in variables
            ])
            weight = objective_config["maximize_allocation_weight"]
            objective -= weight * total_allocation  # 负号表示最大化
            logger.info(f"添加最大化总分配量目标，权重: {weight}")
        
        # 计算轮次总量用于后续均衡计算
        round_totals = []
        round_names_with_vars = []
        for round_name in self.rounds:
            if round_name in variables:
                round_total = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in range(len(self.product_data))
                ])
                round_totals.append(round_total)
                round_names_with_vars.append(round_name)

        if len(round_totals) >= 2:
            # 2. 轮次间总量均衡：最小化最大轮次与最小轮次的差异
            if objective_config.get("round_balance_weight", 0) > 0:
                max_round = pulp.LpVariable("max_round", cat='Continuous')
                min_round = pulp.LpVariable("min_round", cat='Continuous')
                
                for round_total in round_totals:
                    self.model += max_round >= round_total
                    self.model += min_round <= round_total
                
                weight = objective_config["round_balance_weight"]
                objective += weight * (max_round - min_round)
                logger.info(f"添加轮次间总量均衡目标，权重: {weight}")
            
            # 3. 轮次间方差最小化：惩罚各轮次与平均值的偏差
            if objective_config.get("round_variance_weight", 0) > 0:
                avg_round = pulp.lpSum(round_totals) / len(round_totals)
                weight = objective_config["round_variance_weight"]
                
                for i, round_total in enumerate(round_totals):
                    deviation_pos = pulp.LpVariable(f"round_deviation_pos_{i}", lowBound=0)
                    deviation_neg = pulp.LpVariable(f"round_deviation_neg_{i}", lowBound=0)
                
                    self.model += round_total - avg_round == deviation_pos - deviation_neg
                    objective += weight * (deviation_pos + deviation_neg)
                
                logger.info(f"添加轮次间方差最小化目标，权重: {weight}")

            # 4. 品规级别均衡：鼓励品规在多轮分配，避免过度集中
            if objective_config.get("product_balance_weight", 0) > 0:
                weight = objective_config["product_balance_weight"]
                self._add_product_level_balance(objective, variables, weight)
                logger.info(f"添加品规级别均衡目标，权重: {weight}")

            # 5. 轮次间平滑过渡：相邻轮次分配量不应差异过大
            if objective_config.get("smooth_transition_weight", 0) > 0:
                weight = objective_config["smooth_transition_weight"]
                self._add_smooth_transition(objective, variables, weight)
                logger.info(f"添加轮次间平滑过渡目标，权重: {weight}")

        # 添加按需品规软约束惩罚（越往后惩罚越大）
        if hasattr(self, 'demand_penalty_vars'):
            for idx, round_penalties in self.demand_penalty_vars.items():
                for i, (round_name, penalty_var) in enumerate(round_penalties.items(), start=3):
                    # 第三轮惩罚系数为50，第四轮为100，第五轮为200，依此类推
                    penalty_weight = 50 * (2 ** (i - 3))
                    objective += penalty_weight * penalty_var
        
        # 添加分配均衡软约束惩罚
        if hasattr(self, 'balance_penalty_vars'):
            for round_pair, deviations in self.balance_penalty_vars.items():
                # 均衡约束惩罚系数
                balance_penalty_weight = 500
                objective += balance_penalty_weight * (deviations['pos'] + deviations['neg'])
        
        logger.info("目标函数创建完成（支持参数化权重配置）")
        return objective
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        生成求解结果摘要
        
        Returns:
            Dict[str, Any]: 结果摘要
        """
        if self.solution is None:
            return {}
        
        allocation_matrix = self.solution['allocation_matrix']
        
        # 计算基本统计信息
        total_demand = allocation_matrix['需求'].sum()
        total_allocated = allocation_matrix['总分配量'].sum()
        allocation_rate = total_allocated / total_demand if total_demand > 0 else 0
        
        # 计算各轮次分配量
        round_allocations = {}
        for round_name in self.rounds:
            if round_name in allocation_matrix.columns:
                round_allocations[round_name] = allocation_matrix[round_name].sum()
        
        # 计算分配品规数量
        allocated_products = (allocation_matrix['总分配量'] > 0).sum()
        total_products = len(allocation_matrix)
        
        return {
            'total_demand': total_demand,
            'total_allocated': total_allocated,
            'allocation_rate': allocation_rate,
            'round_allocations': round_allocations,
            'allocated_products': allocated_products,
            'total_products': total_products,
            'product_allocation_rate': allocated_products / total_products if total_products > 0 else 0
        }
    
    def _add_product_level_balance(self, objective: pulp.LpAffineExpression, 
                                  variables: Dict[str, Dict[int, pulp.LpVariable]], 
                                  weight: float) -> None:
        """
        添加品规级别的均衡机制
        
        鼓励品规在多轮分配，避免过度集中在某几轮
        仅针对非existing_allocations的品规
        
        Args:
            objective: 目标函数表达式
            variables: 决策变量字典
            weight: 权重系数
        """
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['需求']
            if demand <= 0:
                continue
            
            # 检查是否为已确定分配的品规
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] > 0):
                    is_existing = True
                    break
            
            # 如果是已确定分配的品规，跳过均衡约束
            if is_existing:
                continue
            
            # 获取该品规在各轮次的分配量
            product_allocations = []
            for round_name in self.rounds:
                if round_name in variables:
                    product_allocations.append(variables[round_name][idx])
            
            if len(product_allocations) >= 2:
                # 方法1：最小化该品规的最大分配量与最小分配量的差异
                product_max = pulp.LpVariable(f"product_max_{idx}", cat='Continuous')
                product_min = pulp.LpVariable(f"product_min_{idx}", cat='Continuous')
                
                for allocation in product_allocations:
                    self.model += product_max >= allocation
                    self.model += product_min <= allocation
                
                # 惩罚品规分配的不均衡（使用传入的权重）
                objective += weight * (product_max - product_min)
                
                # 方法2：更强力地惩罚单轮分配量超过需求量的情况
                for round_name in self.rounds:
                    if round_name in variables:
                        # 惩罚超过60%的集中分配
                        concentration_penalty_60 = pulp.LpVariable(f"concentration_penalty_60_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.6 * demand <= concentration_penalty_60
                        objective += weight * 3 * concentration_penalty_60  # 60%集中度惩罚
                        
                        # 更强力惩罚超过80%的集中分配
                        concentration_penalty_80 = pulp.LpVariable(f"concentration_penalty_80_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.8 * demand <= concentration_penalty_80
                        objective += weight * 5 * concentration_penalty_80  # 80%集中度惩罚
                        
                        # 极强惩罚超过90%的集中分配
                        concentration_penalty_90 = pulp.LpVariable(f"concentration_penalty_90_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.9 * demand <= concentration_penalty_90
                        objective += weight * 10 * concentration_penalty_90  # 90%集中度惩罚
                
                # 方法3：鼓励分配到更多轮次（通过二进制变量）
                if demand >= 50:  # 只对需求量较大的品规应用
                    round_binary_vars = []
                    for round_name in self.rounds:
                        if round_name in variables:
                            binary_var = pulp.LpVariable(f"product_round_binary_{idx}_{round_name}", cat='Binary')
                            round_binary_vars.append(binary_var)
                            
                            # 如果该轮有分配，则二进制变量=1
                            M = 10000  # 足够大的数
                            self.model += variables[round_name][idx] <= M * binary_var
                            self.model += variables[round_name][idx] >= 0.1 * binary_var  # 最小分配阈值
                    
                    # 鼓励分配到更多轮次
                    rounds_used = pulp.lpSum(round_binary_vars)
                    if demand >= 100:
                        # 需求量大的品规，鼓励分配到至少2轮
                        rounds_shortage = pulp.LpVariable(f"rounds_shortage_{idx}", lowBound=0)
                        self.model += 2 - rounds_used <= rounds_shortage
                        objective += weight * 1.5 * rounds_shortage
        
        logger.info(f"添加品规级别均衡机制完成，权重: {weight}")
    
    def _add_smooth_transition(self, objective: pulp.LpAffineExpression, 
                              variables: Dict[str, Dict[int, pulp.LpVariable]], 
                              weight: float) -> None:
        """
        添加轮次间平滑过渡机制
        
        避免相邻轮次分配量差异过大，确保分配的平滑性
        
        Args:
            objective: 目标函数表达式
            variables: 决策变量字典
            weight: 权重系数
        """
        round_names = list(self.rounds)
        
        # 计算各轮次的总分配量
        round_totals = {}
        for round_name in round_names:
            if round_name in variables:
                round_totals[round_name] = pulp.lpSum([
                    variables[round_name][idx] for idx in range(len(self.product_data))
                    if self.product_data.iloc[idx]['需求'] > 0
                ])
        
        # 添加相邻轮次平滑约束
        for i in range(len(round_names) - 1):
            current_round = round_names[i]
            next_round = round_names[i + 1]
            
            if current_round in round_totals and next_round in round_totals:
                # 创建差异变量
                diff_var = pulp.LpVariable(f"round_diff_{i}_{i+1}", cat='Continuous')
                
                # 差异的绝对值约束
                self.model += round_totals[next_round] - round_totals[current_round] <= diff_var
                self.model += round_totals[current_round] - round_totals[next_round] <= diff_var
                
                # 惩罚相邻轮次差异过大（使用传入的权重）
                objective += weight * diff_var
        
        # 添加品规级别的相邻轮次平滑约束
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['需求']
            if demand <= 0:
                continue
            
            # 检查是否为已确定分配的品规
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] > 0):
                    is_existing = True
                    break
            
            # 如果是已确定分配的品规，跳过平滑约束
            if is_existing:
                continue
            
            # 对需求量较大的品规应用品规级别平滑约束
            if demand >= 30:
                for i in range(len(round_names) - 1):
                    current_round = round_names[i]
                    next_round = round_names[i + 1]
                    
                    if current_round in variables and next_round in variables:
                        # 品规级别的相邻轮次差异
                        product_diff = pulp.LpVariable(f"product_diff_{idx}_{i}_{i+1}", cat='Continuous')
                        
                        self.model += variables[next_round][idx] - variables[current_round][idx] <= product_diff
                        self.model += variables[current_round][idx] - variables[next_round][idx] <= product_diff
                        
                        # 惩罚品规在相邻轮次的分配差异过大（使用传入的权重）
                        objective += weight * 0.17 * product_diff  # 相对较小的权重
        
        logger.info(f"添加轮次间平滑过渡机制完成，权重: {weight}")
    
    def solve(self, constraint_config: Dict[str, Any] = None, objective_config: Dict[str, Any] = None, time_limit: int = 300) -> Dict[str, Any]:
        """
        求解线性规划问题
        
        Args:
            constraint_config (Dict[str, Any]): 约束配置
            objective_config (Dict[str, Any]): 目标函数配置
            time_limit (int): 求解时间限制（秒）
            
        Returns:
            Dict[str, Any]: 求解结果
        """
        logger.info("开始构建线性规划模型")
        
        # 设置默认配置
        if constraint_config is None:
            constraint_config = {
                "enable_demand_constraints": True,
                "enable_volume_constraints": True,
                "enable_price_constraints": True,
                "enable_c_type_constraints": True,
                "enable_balance_constraints": True,
                "enable_demand_split_constraints": True,
                "enable_demand_based_constraints": True,
                "enable_price_based_constraints": True,
                "volume_tolerance": 0.005,
                "price_upper_limits": None,
                "price_lower_limits": None,
                "volume_limits": None,
                "price_based_ratio": 0.3,
                "c_type_ratio": 0.4,
                "c_type_volume_limit": 4900,
                "chang_type_ratio": 0.2,
                "chang_type_volume_limit": 1000,
                "xi_type_ratio": 0.6,
                "xi_type_volume_limit": 3000
            }
        
        if objective_config is None:
            objective_config = {
                "maximize_allocation_weight": 1000.0,      # 最大化总分配量权重
                "round_balance_weight": 800.0,             # 轮次间总量均衡权重
                "round_variance_weight": 400.0,            # 轮次间方差最小化权重
                "product_balance_weight": 100.0,           # 品规级别均衡权重
                "smooth_transition_weight": 300.0          # 轮次间平滑过渡权重
            }
        
        # 创建模型
        self.model = pulp.LpProblem("Cigarette_Allocation", pulp.LpMinimize)
        
        # 创建变量
        self.variables = self.create_variables()
        
        # 根据配置添加约束
        if constraint_config.get("enable_demand_constraints", True):
            self.add_demand_constraints(self.model, self.variables, constraint_config)  # 约束1：需求满足 
        
        if constraint_config.get("enable_volume_constraints", True):
            self.add_volume_constraints(self.model, self.variables, constraint_config)  # 约束3：投放总量
        
        if constraint_config.get("enable_price_constraints", True):
            self.add_average_price_constraints(self.model, self.variables, constraint_config)  # 约束2：单箱均价
        
        # 根据配置添加分配策略约束
        if constraint_config.get("enable_demand_split_constraints", True):
            self.add_demand_split_constraints(self.model, self.variables)  # 约束6：需求量集中 50 00 150（仅非existing_allocations）
        # 约束7：第一轮货源上限已在创建变量时通过设置上界实现
        if constraint_config.get("enable_demand_based_constraints", True):
            self.add_demand_based_constraints(self.model, self.variables)  # 约束8：按需优先（软约束）
        if constraint_config.get("enable_price_based_constraints", True):
            self.add_price_based_constraints(self.model, self.variables, constraint_config)  # 约束9：按价比例
        
        # 根据配置添加C类烟约束
        if constraint_config.get("enable_c_type_constraints", True):
            self.add_c_type_constraints(self.model, self.variables, constraint_config)  # 约束10.1-10.2
            self._add_c_subtype_constraints(self.model, self.variables, constraint_config)  # 约束10.3-10.5
        
        # 根据配置添加均衡约束
        if constraint_config.get("enable_balance_constraints", True):
            self.add_balance_constraints(self.model, self.variables)  # 约束11：分配均衡（软约束）
        
        # 设置目标函数
        objective = self.create_objective_function(self.variables, objective_config)
        self.model += objective
        
        logger.info(f"模型构建完成，开始求解（时间限制：{time_limit}秒）")
        
        # 求解
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
        self.model.solve(solver)
        
        # 处理求解结果
        status = pulp.LpStatus[self.model.status]
        logger.info(f"求解完成，状态：{status}")
        
        if self.model.status == pulp.LpStatusOptimal:
            self.solution = self._extract_solution()
            
            return {
                'status': 'Optimal',
                'objective_value': pulp.value(self.model.objective),
                'allocation_matrix': self.solution['allocation_matrix'],
                'solve_time': self.model.solutionTime if hasattr(self.model, 'solutionTime') else None,
                'summary': self._generate_summary()
            }
        
        elif self.model.status == pulp.LpStatusInfeasible:
            logger.error("模型不可行")
            return {
                'status': 'Infeasible',
                'message': '约束条件冲突，无可行解',
                'allocation_matrix': None
            }
        
        else:
            logger.warning(f"求解未完成，状态：{status}")
            return {
                'status': status,
                'message': f'求解状态：{status}',
                'allocation_matrix': None
            }
    
    def _extract_solution(self) -> Dict[str, Any]:
        """
        提取求解结果
        
        Returns:
            Dict[str, Any]: 解决方案字典，包含完整的分配矩阵（包含原始表格所有字段）
        """
        # 创建一个包含原始数据的DataFrame
        allocation_matrix = self.data_loader.get_product_data().copy()
        
        # 更新分配结果
        for round_name in self.rounds:
            if round_name in self.variables:
                allocation_matrix[round_name] = 0.0  # 初始化轮次列
                for idx in range(len(self.product_data)):
                    var_value = self.variables[round_name][idx].varValue
                    if var_value is not None:
                        allocation_matrix.loc[idx, round_name] = float(var_value)
        
        # 处理微小分配（小于0.1的分配值）会有很多0.01出现 因为最小 这样可以处理为0
        allocation_matrix = self._handle_small_allocations(allocation_matrix)
        
        # 后处理：处理未分配品规中与需求差值很小的情况
        allocation_matrix = self._handle_tiny_unallocated_demand(allocation_matrix)
        
        # 计算总分配量
        allocation_cols = [col for col in self.rounds if col in allocation_matrix.columns]
        if allocation_cols:
            allocation_matrix['总分配量'] = allocation_matrix[allocation_cols].sum(axis=1)
        else:
            allocation_matrix['总分配量'] = 0
        
        # 计算分配率
        allocation_matrix['分配率'] = np.where(
            allocation_matrix['需求'] > 0,
            allocation_matrix['总分配量'] / allocation_matrix['需求'],
            1
        )
        
        return {
            'allocation_matrix': allocation_matrix,
            'variable_values': {
                round_name: {
                    idx: var.varValue for idx, var in round_vars.items()
                }
                for round_name, round_vars in self.variables.items()
            }
        }
    
    def _handle_small_allocations(self, allocation_matrix: pd.DataFrame, min_threshold: float = 0.1) -> pd.DataFrame:
        """
        处理微小分配问题
        后处理
        将小于阈值的分配量合并到同一品规的其他轮次中，或者清零
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            min_threshold (float): 最小分配阈值
            
        Returns:
            pd.DataFrame: 处理后的分配矩阵
        """
        processed_matrix = allocation_matrix.copy()
        
        for idx in range(len(processed_matrix)):
            # 获取该品规在各轮次的分配量
            round_allocations = {}
            for round_name in self.rounds:
                if round_name in processed_matrix.columns:
                    round_allocations[round_name] = processed_matrix.loc[idx, round_name]

            fixed_rounds = set()
            for round_name in self.rounds:
                if round_name not in self.existing_allocations:
                    continue
                try:
                    exist_val = float(self.existing_allocations[round_name].get(idx, 0.0))
                except Exception:
                    exist_val = 0.0
                if exist_val > 0:
                    fixed_rounds.add(round_name)
            
            # 找出小于阈值的分配
            small_allocations = {
                k: v for k, v in round_allocations.items() if (k not in fixed_rounds) and 0 < v < min_threshold
            }
            large_allocations = {
                k: v for k, v in round_allocations.items() if (k not in fixed_rounds) and v >= min_threshold
            }
            
            if small_allocations and large_allocations:
                # 将小分配合并到最大的分配轮次中
                max_round = max(large_allocations.keys(), key=lambda k: large_allocations[k])
                total_small = sum(small_allocations.values())
                
                # 合并分配
                processed_matrix.loc[idx, max_round] += total_small
                
                # 清零小分配
                for round_name in small_allocations.keys():
                    processed_matrix.loc[idx, round_name] = 0
                    
            elif small_allocations and not large_allocations:
                # 如果只有小分配，将它们合并到第一个有分配的轮次
                if small_allocations:
                    first_round = list(small_allocations.keys())[0]
                    total_small = sum(small_allocations.values())
                    
                    processed_matrix.loc[idx, first_round] = total_small
                    for round_name in list(small_allocations.keys())[1:]:
                        processed_matrix.loc[idx, round_name] = 0
        
        logger.info(f"微小分配处理完成，阈值：{min_threshold}")
        return processed_matrix
    
    def _handle_tiny_unallocated_demand(self, allocation_matrix: pd.DataFrame, tiny_threshold: float = 0.01) -> pd.DataFrame:
        """
        处理未分配品规中与需求差值很小的情况
        
        对于未分配品规，如果跟需求差值很小（如0.01），将这个微小差值分配到任意已分配的轮次里
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            tiny_threshold (float): 微小差值阈值，默认0.01
            
        Returns:
            pd.DataFrame: 处理后的分配矩阵
        """
        processed_matrix = allocation_matrix.copy()
        
        for idx in range(len(processed_matrix)):
            demand = processed_matrix.loc[idx, '需求']
            if demand <= 0:
                continue
            
            # 计算当前总分配量
            round_allocations = {}
            total_allocated = 0
            for round_name in self.rounds:
                if round_name in processed_matrix.columns:
                    allocation = processed_matrix.loc[idx, round_name]
                    round_allocations[round_name] = allocation
                    total_allocated += allocation
            
            # 计算未分配量
            unallocated = demand - total_allocated
            
            # 如果未分配量很小且为正值，将其分配到已有分配的轮次中
            if unallocated <= tiny_threshold:
                # 找到已有分配的轮次
                allocated_rounds = {k: v for k, v in round_allocations.items() if v > 0}
                
                if allocated_rounds:
                    # 分配到分配量最大的轮次
                    max_round = max(allocated_rounds.keys(), key=lambda k: allocated_rounds[k])
                    processed_matrix.loc[idx, max_round] += unallocated
                    
                    logger.debug(f"品规{idx}的微小未分配量{unallocated:.3f}已分配到{max_round}")
                else:
                    # 如果没有已分配的轮次，分配到第一轮
                    first_round = self.rounds[0] if self.rounds else None
                    if first_round and first_round in processed_matrix.columns:
                        processed_matrix.loc[idx, first_round] = unallocated
                        logger.debug(f"品规{idx}的微小未分配量{unallocated:.3f}已分配到{first_round}")
        
        logger.info(f"微小未分配需求处理完成，阈值：{tiny_threshold}")
        return processed_matrix
    
    def post_process_solution(self, allocation_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        后处理解决方案
        
        处理一些线性规划难以处理的约束，如方型集中投放等
        暂未使用
        Args:
            allocation_matrix (pd.DataFrame): 原始分配矩阵
            
        Returns:
            pd.DataFrame: 后处理后的分配矩阵
        """
        processed_matrix = allocation_matrix.copy()
        
        # 处理C类烟集中投放约束
        type_column = self.product_data['类'].astype(str)
        c_type_mask = type_column.str.contains('C', na=False)
        c_type_indices = c_type_mask[c_type_mask].index.tolist()
        
        for idx in c_type_indices:
            # 找到分配量最大的轮次
            round_allocations = processed_matrix.loc[idx, self.rounds]
            if round_allocations.sum() > 0:
                max_round = round_allocations.idxmax()
                total_allocation = round_allocations.sum()
                
                # 将所有分配集中到最大轮次
                for round_name in self.rounds:
                    if round_name == max_round:
                        processed_matrix.loc[idx, round_name] = total_allocation
                    else:
                        processed_matrix.loc[idx, round_name] = 0
        
        # 整数化处理（保留三位小数）
        processed_matrix = processed_matrix.round(3)
        
        logger.info("解决方案后处理完成")
        return processed_matrix


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    import traceback

    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    try:
        from src.data.data_loader import DataLoader

        # 创建数据加载器
        data_loader = DataLoader(r"D:\2-code\huoyuan\data\huoyuanfenpei.xlsx")
        
        # 创建线性规划分配器
        allocator = LinearProgrammingAllocator(data_loader)
        
        # 求解
        result = allocator.solve(time_limit=300)
        
        print("求解结果:", result['status'])
        if result['status'] == 'optimal':
            print("目标函数值:", result['objective_value'])
            print("分配矩阵形状:", result['allocation_matrix'].shape)
            print("\n前5行分配矩阵:")
            print(result['allocation_matrix'].head())
            result['allocation_matrix'].to_excel('output5.xlsx', index=False)

        
    except Exception as e:
        print(f"测试失败: {e}")
        traceback.print_exc()
