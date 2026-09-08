"""
卷烟货源分配算法系统 - 约束管理模块

该模块负责管理和验证所有约束条件，包括：
- 需求满足约束
- 单箱均价约束  
- 投放总量约束
- C类烟特殊约束
- 分配策略约束

"""
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from data_loader import DataLoader


class ConstraintManager:
    """
    约束管理器类
    
    负责管理和验证卷烟货源分配的所有约束条件
    """
    
    def __init__(self, data_loader, constraint_config=None):
        """
        初始化约束管理器
        
        Args:
            data_loader: 数据加载器实例
            constraint_config: 约束配置字典，包含volume_tolerance等参数
        """
        self.data_loader = data_loader
        self.product_data = data_loader.get_product_data()  # 获取完整产品数据
        self.constraint_config = constraint_config or {}
        
        # 初始化约束参数（支持动态配置）
        self._init_constraint_parameters()
        
        # 动态获取轮次（与data_loader保持一致）
        self.rounds = data_loader.get_rounds()
        self.round_constraints = {}
        for round_name in self.rounds:
            try:
                # 从data_loader获取默认约束值
                constraints = data_loader.get_round_constraints(round_name)
                
                # 优先使用constraint_config中的轮次级别约束参数
                price_upper = constraints['upper_price_limit']
                price_lower = constraints['lower_price_limit']
                total_quantity = constraints['total_quantity']
                
                # 如果constraint_config中有轮次级别的约束参数，则覆盖默认值
                if self.price_upper_limits and round_name in self.price_upper_limits:
                    price_upper = self.price_upper_limits[round_name]
                    
                if self.price_lower_limits and round_name in self.price_lower_limits:
                    price_lower = self.price_lower_limits[round_name]
                    
                if self.volume_limits and round_name in self.volume_limits:
                    total_quantity = self.volume_limits[round_name]
                
                # 修正约束字段名以匹配data_loader的返回格式
                self.round_constraints[round_name] = {
                    'price_upper': price_upper,
                    'price_lower': price_lower,
                    'total_target': total_quantity,
                    'total_upper': total_quantity * (1 + self.volume_tolerance),  # 动态容差
                    'total_lower': total_quantity * (1 - self.volume_tolerance)   # 动态容差
                }
                
                logger.debug(f"初始化轮次 {round_name} 约束: price_upper={price_upper}, "
                           f"price_lower={price_lower}, total_quantity={total_quantity}")
                           
            except (ValueError, KeyError) as e:
                logger.warning(f"无法获取轮次 {round_name} 的约束条件: {e}")
                continue
        
        self.existing_allocations = data_loader.get_existing_allocations()
        
        # 创建C类烟和按价品规标识
        self._create_auxiliary_flags()
    
    def _init_constraint_parameters(self):
        """
        初始化约束参数，支持从constraint_config动态配置
        """
        # 投放总量容差，默认为0.5%
        self.volume_tolerance = self.constraint_config.get('volume_tolerance', 0.005)
        
        # 按价品规比例要求，默认30%
        self.price_based_ratio = self.constraint_config.get('price_based_ratio', 0.3)
        
        # C类烟约束参数
        self.c_type_ratio = self.constraint_config.get('c_type_ratio', 0.4)  # 默认40%
        self.c_type_volume_limit = self.constraint_config.get('c_type_volume_limit', 4900)  # 默认4900箱
        
        # 长型约束参数
        self.chang_type_ratio = self.constraint_config.get('chang_type_ratio', 0.2)  # 默认20%
        self.chang_type_volume_limit = self.constraint_config.get('chang_type_volume_limit', 1000)  # 默认1000箱
        
        # 细型约束参数
        self.xi_type_ratio = self.constraint_config.get('xi_type_ratio', 0.6)  # 默认60%
        self.xi_type_volume_limit = self.constraint_config.get('xi_type_volume_limit', 3000)  # 默认3000箱
        
        # 轮次级别的约束参数（覆盖dataloader默认值）
        self.price_upper_limits = self.constraint_config.get('price_upper_limits', None)  # 各轮单箱均价上限
        self.price_lower_limits = self.constraint_config.get('price_lower_limits', None)  # 各轮单箱均价下限
        self.volume_limits = self.constraint_config.get('volume_limits', None)  # 各轮投放总量限制
        
        logger.debug(f"约束参数初始化完成: volume_tolerance={self.volume_tolerance}, "
                    f"price_based_ratio={self.price_based_ratio}, c_type_ratio={self.c_type_ratio}, "
                    f"c_type_volume_limit={self.c_type_volume_limit}, chang_type_ratio={self.chang_type_ratio}, "
                    f"chang_type_volume_limit={self.chang_type_volume_limit}, xi_type_ratio={self.xi_type_ratio}, "
                    f"xi_type_volume_limit={self.xi_type_volume_limit}, price_upper_limits={self.price_upper_limits}, "
                    f"price_lower_limits={self.price_lower_limits}, volume_limits={self.volume_limits}")
    
    def _create_auxiliary_flags(self):
        """
        创建辅助标识字段：is_c_type（C类烟）和is_price_based（按价品规）
        根据huoyuanfenpei.xlsx的C列和属列获取C类烟信息
        """
        # 创建C类烟标识：C列有值的为C类烟
        if 'C' in self.product_data.columns:
            self.product_data['is_c_type'] = self.product_data['C'].notna() & (self.product_data['C'].str.strip() != '')
        else:
            logger.warning("未找到C列，无法识别C类烟")
            self.product_data['is_c_type'] = False
        
        # 创建按价品规标识：按价列含"价"字的品规
        if '按价' in self.product_data.columns:
            self.product_data['is_price_based'] = self.product_data['按价'].notna() & self.product_data['按价'].str.contains('价', na=False)
        else:
            logger.warning("未找到按价列，无法识别按价品规")
            self.product_data['is_price_based'] = False
        
        # 创建按需品规标识：按需列含"需"字的品规
        if '按需' in self.product_data.columns:
            self.product_data['is_demand_based'] = self.product_data['按需'].notna() & self.product_data['按需'].str.contains('需', na=False)
        else:
            logger.warning("未找到按需列，无法识别按需品规")
            self.product_data['is_demand_based'] = False
        
        # 获取C类烟的子类型（从属列获取）
        if 'C类' in self.product_data.columns:
            self.product_data['c_subtype'] = self.product_data['C类'].fillna('')
        else:
            logger.warning("未找到属列，无法识别C类烟子类型")
            self.product_data['c_subtype'] = ''
        
        logger.info(f"识别到 {self.product_data['is_c_type'].sum()} 个C类烟品规")
        logger.info(f"识别到 {self.product_data['is_price_based'].sum()} 个按价品规")
        logger.info(f"识别到 {self.product_data['is_demand_based'].sum()} 个按需品规")
        
    def update_config(self, constraint_config):
        """
        待修改  不太对
        根据constraint_config更新ConstraintManager的所有相关参数
        
        Args:
            constraint_config (Dict): 约束配置字典，包含各种约束参数
        """
        logger.info("更新ConstraintManager配置参数")
        
        # 更新约束配置
        if constraint_config:
            self.constraint_config.update(constraint_config)
            
            # 重新初始化约束参数
            self._init_constraint_parameters()
            
            # # 更新data_loader中的轮次约束（如果有轮次级别的配置）
            # if any(key in constraint_config for key in ['price_upper_limits', 'price_lower_limits', 'volume_limits']):
            #     self.data_loader.update_all_round_constraints(constraint_config)
            #     logger.debug("更新data_loader中的轮次约束配置")
            
            # 重置轮次约束以应用新的配置
            self.reset_round_constraints()
            
            logger.info("ConstraintManager配置更新完成")
    
    def reset_round_constraints(self, constraint_config=None):
        """
        待修改  不太对
        重置轮次约束值，重新从data_loader获取最新的约束数据
        避免约束应用错误的值
        
        Args:
            constraint_config: 约束配置字典，包含volume_tolerance等参数
        """
        logger.info("重置轮次约束值")
        
        # 更新约束配置和容差
        if constraint_config:
            self.constraint_config = constraint_config
            self._init_constraint_parameters()
        
        self.round_constraints = {}
        for round_name in self.rounds:
            try:
                constraints = self.data_loader.get_round_constraints(round_name)
                # 修正约束字段名以匹配data_loader的返回格式
                self.round_constraints[round_name] = {
                    'price_upper': constraints['upper_price_limit'],
                    'price_lower': constraints['lower_price_limit'],
                    'total_target': constraints['total_quantity'],
                    'total_upper': constraints['total_quantity'] * (1 + self.volume_tolerance),  # 动态容差
                    'total_lower': constraints['total_quantity'] * (1 - self.volume_tolerance)   # 动态容差
                }
                logger.debug(f"重置轮次 {round_name} 约束: {self.round_constraints[round_name]}")
            except (ValueError, KeyError) as e:
                logger.warning(f"无法重置轮次 {round_name} 的约束条件: {e}")
                continue
        
    def validate_demand_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证需求满足约束
        
        约束1: 各品规在所有轮的投放量之和 == Sheet1中的"需求"列
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵，行为品规，列为轮次
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        # 计算每个品规的总分配量
        total_allocations = allocation_matrix.sum(axis=1)
        demands = self.product_data['需求']
        
        # 检查需求满足情况
        for idx, (total_alloc, demand) in enumerate(zip(total_allocations, demands)):
            product_name = self.product_data.iloc[idx]['卷烟名称']
            diff = abs(total_alloc - demand)
            
            # 允许小的数值误差（0.001）
            if diff > 0.001:
                results['is_valid'] = False
                results['violations'].append({
                    'product': product_name,
                    'demand': demand,
                    'allocated': total_alloc,
                    'difference': diff
                })
        
        results['details']['total_violations'] = len(results['violations'])
        results['details']['max_violation'] = max([v['difference'] for v in results['violations']], default=0)
        
        return results
    
    def validate_volume_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证投放总量约束
        
        约束3: 每轮投放总量可上下浮动±0.5%，以Sheet2中数值为基准
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }

        eps = 0.01
        
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            total_allocation = allocation_matrix[round_name].sum()
            
            # 获取约束条件，优先使用配置参数覆盖默认值
            target = None
            upper_limit = None
            lower_limit = None
            
            # 优先使用constraint_config中的轮次级别约束参数
            if self.volume_limits and round_name in self.volume_limits:
                target = self.volume_limits[round_name]
                upper_limit = target * (1 + self.volume_tolerance)
                lower_limit = target * (1 - self.volume_tolerance)
            elif round_name in self.round_constraints:
                target = self.round_constraints[round_name]['total_target']
                upper_limit = self.round_constraints[round_name]['total_upper']
                lower_limit = self.round_constraints[round_name]['total_lower']
            
            # 检查是否违反约束
            if target is not None and upper_limit is not None and lower_limit is not None:
                if total_allocation - eps > upper_limit or total_allocation + eps < lower_limit:
                    results['is_valid'] = False
                    results['violations'].append({
                        'round': round_name,
                        'actual_volume': total_allocation,
                        'target_volume': target,
                        'upper_limit': upper_limit,
                        'lower_limit': lower_limit,
                        'violation_type': 'upper' if total_allocation > upper_limit else 'lower'
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_price_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证单箱均价约束
        
        约束2: 每轮的单箱均价应在Sheet2中指定的上下限区间内
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            allocation = allocation_matrix[round_name]
            total_quantity = allocation.sum()
            
            if total_quantity == 0:
                continue
            
            # 本地计算该轮单箱均价
            total_revenue = 0
            for idx, quantity in allocation.items():
                if quantity > 0:
                    wholesale_price = self.product_data.loc[idx, '批发价']
                    stick_ratio = self.product_data.loc[idx, '条支比']
                    # 单轮某品规销售额 = 投放量 × 批发价 × 50000 / 条支比
                    revenue = quantity * wholesale_price * 50000 / stick_ratio
                    total_revenue += revenue
            
            # 单箱均价 = 单轮销售额 / 总投放量
            avg_price = total_revenue / total_quantity
            
            # 获取约束条件，优先使用配置参数覆盖默认值
            price_upper = None
            price_lower = None
            
            # 优先使用constraint_config中的轮次级别约束参数
            if self.price_upper_limits and round_name in self.price_upper_limits:
                price_upper = self.price_upper_limits[round_name]
            elif round_name in self.round_constraints:
                price_upper = self.round_constraints[round_name]['price_upper']
            
            if self.price_lower_limits and round_name in self.price_lower_limits:
                price_lower = self.price_lower_limits[round_name]
            elif round_name in self.round_constraints:
                price_lower = self.round_constraints[round_name]['price_lower']
            
            # 检查是否违反约束
            if price_upper is not None and price_lower is not None:
                if avg_price-0.01 > price_upper or avg_price+0.01 < price_lower:
                    results['is_valid'] = False
                    results['violations'].append({
                        'round': round_name,
                        'actual_price': avg_price,
                        'upper_limit': price_upper,
                        'lower_limit': price_lower,
                        'violation_type': 'upper' if avg_price > price_upper else 'lower'
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_fixed_allocation_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证固定分配约束
        
        约束4: Sheet1中已确定的有数值的投放品规不能更改
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }

        eps = 1e-6
        
        for round_name in self.rounds:
            if round_name not in self.existing_allocations:
                continue
                
            existing = self.existing_allocations[round_name]
            current = allocation_matrix[round_name] if round_name in allocation_matrix.columns else pd.Series(0, index=existing.index)
            
            # 检查已有分配是否被更改
            for idx in existing.index:
                exist_val = existing[idx]
                curr_val = current[idx]
                if exist_val > 0 and abs(exist_val - curr_val) > eps:
                    product_name = self.product_data.loc[idx, '卷烟名称']
                    results['is_valid'] = False
                    results['violations'].append({
                        'product': product_name,
                        'round': round_name,
                        'original_value': exist_val,
                        'current_value': curr_val
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results

    def validate_first_round_supply_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证第一轮货源上限约束
        
        约束7: 第一轮总投放量不得超过可用货源上限
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }

        eps = 1e-6
        
        # 获取第一轮
        first_round = min(self.rounds) if self.rounds else None
        if not first_round or first_round not in allocation_matrix.columns:
            return results
        
        first_round_allocation = allocation_matrix[first_round]
        
        # 检查每个品规的第一轮分配是否超过可用货源
        for idx, allocation in first_round_allocation.items():
            if allocation > 0:
                available_supply = self.product_data.loc[idx, '可用货源']
                if allocation > available_supply + eps:
                    product_name = self.product_data.loc[idx, '卷烟名称']
                    results['is_valid'] = False
                    results['violations'].append({
                        'product': product_name,
                        'allocated': allocation,
                        'available_supply': available_supply,
                        'excess': allocation - available_supply
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_demand_based_priority_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证按需品规优先约束
        
        约束8: "按需"列含"需"字的品规应优先投放在前两轮
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        demand_based_mask = self.product_data['is_demand_based']
        
        if len(self.rounds) < 2:
            return results
        
        first_two_rounds = self.rounds[:2]
        later_rounds = self.rounds[2:] if len(self.rounds) > 2 else []
        
        for idx in self.product_data[demand_based_mask].index:
            product_name = self.product_data.loc[idx, '卷烟名称']
            
            # 计算前两轮和后续轮次的分配量
            first_two_allocation = 0
            later_allocation = 0
            
            for round_name in first_two_rounds:
                if round_name in allocation_matrix.columns:
                    first_two_allocation += allocation_matrix.loc[idx, round_name]
            
            for round_name in later_rounds:
                if round_name in allocation_matrix.columns:
                    later_allocation += allocation_matrix.loc[idx, round_name]
            
            total_allocation = first_two_allocation + later_allocation
            
            # 如果有分配，检查是否优先在前两轮
            if total_allocation > 0:
                first_two_ratio = first_two_allocation / total_allocation
                
                # 要求至少100%的分配在前两轮（可调整这个阈值）
                if first_two_ratio < 1.0:
                    results['is_valid'] = False
                    results['violations'].append({
                        'product': product_name,
                        'first_two_allocation': first_two_allocation,
                        'later_allocation': later_allocation,
                        'first_two_ratio': first_two_ratio,
                        'required_ratio': 1.0
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    
    def validate_c_type_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证C类烟约束
        
        约束10: C类烟的复杂约束条件
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        # 获取C类烟数据
        c_type_mask = self.product_data['is_c_type']
        c_type_data = self.product_data[c_type_mask]
        
        if len(c_type_data) == 0:
            return results
        
        # 约束10.1: 每轮不得超过总量的指定比例
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            round_allocation = allocation_matrix[round_name]
            total_round = round_allocation.sum()
            c_type_round = round_allocation[c_type_mask].sum()
            
            if total_round > 0 and c_type_round / total_round > self.c_type_ratio:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'C类烟每轮占比',
                    'round': round_name,
                    'actual_ratio': c_type_round / total_round,
                    'limit': self.c_type_ratio
                })
        
        # 约束10.2: 每轮不得超过指定总量
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
            if c_type_round > self.c_type_volume_limit:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'C类烟每轮总量',
                    'round': round_name,
                    'actual_total': c_type_round,
                    'limit': self.c_type_volume_limit
                })
        
        # 约束10.3: "方"型集中一轮投
        fang_mask = self.product_data['c_subtype'].str.contains('方', na=False)
        if fang_mask.any():
            fang_allocation = allocation_matrix.loc[fang_mask]
            non_zero_rounds = (fang_allocation > 0).sum(axis=0)
            if (non_zero_rounds > 0).sum() > 1:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'C类方型集中投放',
                    'actual_rounds': (non_zero_rounds > 0).sum(),
                    'limit': 1
                })
        
        # 约束10.4: "长"型每轮不得超过C类总投放的20%，每轮≤1000
        chang_mask = self.product_data['c_subtype'].str.contains('长', na=False)
        if chang_mask.any():
            chang_allocation = allocation_matrix.loc[chang_mask]
            
            for round_name in self.rounds:
                if round_name not in allocation_matrix.columns:
                    continue
                    
                c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
                chang_round = chang_allocation[round_name].sum()
                
                # 每轮总量限制
                if chang_round > self.chang_type_volume_limit:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'C类长型每轮总量',
                        'round': round_name,
                        'actual_total': chang_round,
                        'limit': self.chang_type_volume_limit
                    })
                
                # 占比限制
                if c_type_round > 0 and chang_round / c_type_round > self.chang_type_ratio:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'C类长型每轮占比',
                        'round': round_name,
                        'actual_ratio': chang_round / c_type_round,
                        'limit': self.chang_type_ratio
                    })
        
        # 约束10.5: "细"型每轮不得超过C类总投放的60%，每轮≤3000
        xi_mask = self.product_data['c_subtype'].str.contains('细', na=False)
        if xi_mask.any():
            xi_allocation = allocation_matrix.loc[xi_mask]
            
            for round_name in self.rounds:
                if round_name not in allocation_matrix.columns:
                    continue
                    
                c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
                xi_round = xi_allocation[round_name].sum()
                
                # 每轮总量限制
                if xi_round > self.xi_type_volume_limit:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'C类细型每轮总量',
                        'round': round_name,
                        'actual_total': xi_round,
                        'limit': self.xi_type_volume_limit
                    })
                
                # 占比限制
                if c_type_round > 0 and xi_round / c_type_round > self.xi_type_ratio:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'C类细型每轮占比',
                        'round': round_name,
                        'actual_ratio': xi_round / c_type_round,
                        'limit': self.xi_type_ratio
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_price_based_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证按价品规约束
        
        约束9: "按价"列含"价"的品规在各轮中需占品规总数≥30%
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        price_based_mask = self.product_data['is_price_based']
        
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            round_allocation = allocation_matrix[round_name]
            
            # 该轮有投放的品规
            allocated_products = round_allocation > 0
            total_allocated_products = allocated_products.sum()
            
            # 该轮有投放的按价品规
            price_based_allocated = allocated_products & price_based_mask
            price_based_count = price_based_allocated.sum()
            
            if total_allocated_products > 0:
                ratio = price_based_count / total_allocated_products
                if ratio < self.price_based_ratio:
                    results['is_valid'] = False
                    results['violations'].append({
                        'round': round_name,
                        'actual_ratio': ratio,
                        'required_ratio': self.price_based_ratio,
                        'price_based_count': price_based_count,
                        'total_count': total_allocated_products
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_all_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        验证所有约束条件
        
        Args:
            allocation_matrix (pd.DataFrame): 分配矩阵
            
        Returns:
            Dict[str, Any]: 完整验证结果
        """
        logger.info("开始验证所有约束条件")
        
        all_results = {
            'overall_valid': True,
            'constraint_results': {},
            'summary': {}
        }
        
        # 根据约束配置决定需要验证的约束
        constraint_validators = {}
        
        # 需求满足约束 - 通常总是启用
        if self.constraint_config.get('enable_demand_constraints', True):
            constraint_validators['demand'] = self.validate_demand_constraints
            
        # 单箱均价约束
        if self.constraint_config.get('enable_price_constraints', True):
            constraint_validators['price'] = self.validate_price_constraints
            
        # 投放总量约束
        if self.constraint_config.get('enable_volume_constraints', True):
            constraint_validators['volume'] = self.validate_volume_constraints
            
        # 固定分配约束 - 通常总是启用（保护已有分配）
        constraint_validators['fixed_allocation'] = self.validate_fixed_allocation_constraints
        
        # 第一轮货源上限约束 - 通常总是启用（保护货源限制）
        constraint_validators['first_round_supply'] = self.validate_first_round_supply_constraints
        
        # 按需优先约束
        if self.constraint_config.get('enable_demand_based_constraints', True):
            constraint_validators['demand_based_priority'] = self.validate_demand_based_priority_constraints
            
        # 按价比例约束
        if self.constraint_config.get('enable_price_based_constraints', True):
            constraint_validators['price_based'] = self.validate_price_based_constraints
            
        # C类烟约束
        if self.constraint_config.get('enable_c_type_constraints', True):
            constraint_validators['c_type'] = self.validate_c_type_constraints
        
        total_violations = 0
        
        # 记录跳过的约束
        skipped_constraints = []
        all_constraint_names = [
            'demand', 'price', 'volume', 'fixed_allocation', 'first_round_supply',
            'demand_based_priority', 'price_based', 'c_type'
        ]
        
        for constraint_name in all_constraint_names:
            if constraint_name not in constraint_validators:
                skipped_constraints.append(constraint_name)
                logger.info(f"跳过{constraint_name}约束验证（已禁用）")
        
        for constraint_name, validator in constraint_validators.items():
            try:
                result = validator(allocation_matrix)
                all_results['constraint_results'][constraint_name] = result
                
                if not result['is_valid']:
                    all_results['overall_valid'] = False
                    total_violations += result['details'].get('total_violations', 0)
                    
                logger.info(f"{constraint_name}约束验证完成: {'通过' if result['is_valid'] else '违反'}")
                
            except Exception as e:
                logger.error(f"验证{constraint_name}约束时出错: {str(e)}")
                all_results['constraint_results'][constraint_name] = {
                    'is_valid': False,
                    'error': str(e)
                }
                all_results['overall_valid'] = False
        
        # 生成摘要
        all_results['summary'] = {
            'total_violations': total_violations,
            'violated_constraints': [name for name, result in all_results['constraint_results'].items() 
                                   if not result.get('is_valid', False)],
            'passed_constraints': [name for name, result in all_results['constraint_results'].items() 
                                 if result.get('is_valid', False)],
            'skipped_constraints': skipped_constraints,
            'enabled_constraints': list(constraint_validators.keys())
        }
        
        logger.info(f"约束验证完成: {'全部通过' if all_results['overall_valid'] else f'发现{total_violations}个违反'}")
        logger.info(f"启用的约束: {list(constraint_validators.keys())}")
        logger.info(f"跳过的约束: {skipped_constraints}")
        
        return all_results
    
    def get_constraint_weights(self) -> Dict[str, float]:
        """
        获取约束权重（用于优化目标函数）
        
        Returns:
            Dict[str, float]: 约束权重字典
        """
        return {
            'demand_satisfaction': 1000.0,      # 需求满足权重最高
            'price_deviation': 100.0,           # 价格偏差权重
            'volume_deviation': 50.0,           # 总量偏差权重
            'balance_penalty': 10.0,            # 分配均衡权重
            'c_type_penalty': 200.0,            # C类烟约束权重
            'priority_bonus': 20.0              # 优先级奖励权重
        }

if __name__ == "__main__":
    # 1. 加载数据
    logger.info("=== 开始测试约束管理器 ===")
    data_loader = DataLoader(r"D:\2-code\huoyuan\data\huoyuanfenpei.xlsx")

    # 2. 创建约束管理器
    constraint_manager = ConstraintManager(data_loader)

    output_file = r"D:\2-code\huoyuan\output4.xlsx"
    output_data = pd.read_excel(output_file)

    # 验证所有约束
    validation_results = constraint_manager.validate_all_constraints(output_data[constraint_manager.rounds])
    # 显示验证结果
    logger.info(f"整体验证结果: {'通过' if validation_results['overall_valid'] else '失败'}")
    logger.info(f"总违反数: {validation_results['summary']['total_violations']}")
    logger.info(f"违反的约束: {validation_results['summary']['violated_constraints']}")
    logger.info(f"通过的约束: {validation_results['summary']['passed_constraints']}")

    # 详细显示违反情况
    for constraint_name, result in validation_results['constraint_results'].items():
        if not result.get('is_valid', True):
            logger.warning(f"约束 {constraint_name} 违反:")
            for violation in result.get('violations', []):
                logger.warning(f"  - {violation}")
