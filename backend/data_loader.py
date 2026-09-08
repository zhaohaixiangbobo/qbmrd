"""
卷烟货源分配算法系统 - 数据加载模块

该模块负责从Excel文件中加载货源数据和分配限制，
并进行数据验证和预处理。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器类
    
    负责从Excel文件加载卷烟货源分配数据，包括：
    - Sheet1: 货源预分配数据
    - Sheet2: 各轮控制要求
    """
    
    def __init__(self, file_path: str):
        """
        初始化数据加载器
        
        Args:
            file_path (str): Excel文件路径
        """
        self.file_path = file_path
        self.sheet1_data = None
        self.sheet2_data = None
        self.rounds = []  # 动态检测轮次
        self.round_constraints = {}  # 缓存轮次约束数据
        
        # 自动加载数据
        self.load_data()
        
    def load_data(self) -> None:
        """
        加载数据
        
        从Excel文件加载所有数据并进行预处理
        """
        try:
            logger.info(f"开始加载数据文件: {self.file_path}")
            
            # 加载Sheet1数据
            self.sheet1_data = pd.read_excel(self.file_path, sheet_name='Sheet1')
            logger.info(f"Sheet1数据加载完成，共{len(self.sheet1_data)}行数据")
            
            # 加载Sheet2数据
            self.sheet2_data = pd.read_excel(self.file_path, sheet_name='Sheet2', index_col=0)
            logger.info(f"Sheet2数据加载完成，共{len(self.sheet2_data)}行限制条件")
            
            # 动态检测轮次
            self._detect_rounds()
            
            # 数据验证
            self._validate_data()
            
            # 数据预处理
            self._preprocess_data()
            
            # 初始化约束缓存
            self._initialize_constraints_cache()
            
            logger.info("数据加载和预处理完成")
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def _detect_rounds(self) -> None:
        """
        动态检测轮次
        """
        # 从Sheet1检测轮次（作为备选）
        sheet1_rounds = [col for col in self.sheet1_data.columns 
                        if '轮' in col and col.startswith('第')]
        
        # 取两者的交集，确保数据一致性
        cn_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}
        self.rounds = sorted(list(set(sheet1_rounds)),key=lambda x: cn_num_map.get(x[1], 0))
        
        if not self.rounds:
            # 如果没有检测到轮次，使用默认的前5轮
            self.rounds = ['第一轮', '第二轮', '第三轮', '第四轮', '第五轮']
            logger.warning("未检测到轮次信息，使用默认轮次: " + str(self.rounds))
        else:
            logger.info(f"检测到轮次: {self.rounds}")
    
    def _validate_data(self) -> None:
        """
        验证数据格式和完整性
        
        Raises:
            ValueError: 数据格式错误
        """
        # 验证Sheet1必需列
        required_sheet1_cols = ['代码', '批发价', '卷烟名称', '类', '需求', '可用货源'] + self.rounds
        missing_cols = [col for col in required_sheet1_cols if col not in self.sheet1_data.columns]
        if missing_cols:
            raise ValueError(f"Sheet1缺少必需列: {missing_cols}")
        
        # 验证Sheet2必需行
        required_sheet2_rows = ['单箱均价上限', '单箱均价下限', '总量']
        missing_rows = [row for row in required_sheet2_rows if row not in self.sheet2_data.index]
        if missing_rows:
            raise ValueError(f"Sheet2缺少必需行: {missing_rows}")
        
        # 验证Sheet2必需列（轮次）
        missing_round_cols = [col for col in self.rounds if col not in self.sheet2_data.columns]
        if missing_round_cols:
            raise ValueError(f"Sheet2缺少轮次列: {missing_round_cols}")
        
        logger.info("数据验证通过")
    
    def _preprocess_data(self) -> None:
        """
        数据预处理
        
        清理数据，创建辅助列，处理缺失值
        """
        # 处理Sheet1数据
        # 清理列名中的换行符
        # self.sheet1_data.columns = [col.replace('\n', '') for col in self.sheet1_data.columns]
        
        # 重新检测轮次（因为列名可能已更改）
        sheet1_rounds = [col for col in self.sheet1_data.columns 
                        if '轮' in col and col.startswith('第')]
        cn_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}
        self.rounds = sorted(list(set(self.rounds) & set(sheet1_rounds)),key=lambda x: cn_num_map.get(x[1], 0))

        
        # 处理缺失值
        self.sheet1_data['需求'] = pd.to_numeric(self.sheet1_data['需求'], errors='coerce').fillna(0)
        self.sheet1_data['可用货源'] = pd.to_numeric(self.sheet1_data['可用货源'], errors='coerce').fillna(0)
        self.sheet1_data['批发价'] = pd.to_numeric(self.sheet1_data['批发价'], errors='coerce').fillna(0)
        
        # 处理轮次分配数据
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                self.sheet1_data[round_name] = pd.to_numeric(self.sheet1_data[round_name], errors='coerce').fillna(0)
        
        # 创建辅助列
        self._create_auxiliary_columns()
        
        logger.info("数据预处理完成")
    
    def _create_auxiliary_columns(self) -> None:
        """
        创建辅助列
        
        计算单箱价格、总分配量等辅助信息
        """
        # 计算单箱价格（假设每箱50条，每条20支）
        if '条支比' in self.sheet1_data.columns:
            # 如果有条支比列，使用实际条支比
            self.sheet1_data['条支比'] = pd.to_numeric(self.sheet1_data['条支比'], errors='coerce').fillna(200)
            self.sheet1_data['单箱价格'] = self.sheet1_data['批发价'] * 50000 / self.sheet1_data['条支比']
        else:
            # 默认每条20支
            self.sheet1_data['单箱价格'] = self.sheet1_data['批发价'] * 2500  # 50条 * 50元/条
        
        # 计算总分配量
        allocation_cols = [col for col in self.rounds if col in self.sheet1_data.columns]
        if allocation_cols:
            self.sheet1_data['总分配量'] = self.sheet1_data[allocation_cols].sum(axis=1)
        else:
            self.sheet1_data['总分配量'] = 0
        
        # 计算分配率
        self.sheet1_data['分配率'] = np.where(
            self.sheet1_data['需求'] > 0,
            self.sheet1_data['总分配量'] / self.sheet1_data['需求'],
            1
        )
        
        logger.info("辅助列创建完成")
    
    def _initialize_constraints_cache(self) -> None:
        """
        初始化约束缓存
        
        从Sheet2数据中加载所有轮次的约束条件到缓存中
        """
        logger.info("初始化约束缓存")
        self.round_constraints = {}
        for round_name in self.rounds:
            if round_name in self.sheet2_data.columns:
                try:
                    self.round_constraints[round_name] = {
                        'upper_price_limit': float(self.sheet2_data.loc['单箱均价上限', round_name]),
                        'lower_price_limit': float(self.sheet2_data.loc['单箱均价下限', round_name]),
                        'total_quantity': float(self.sheet2_data.loc['总量', round_name])
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"无法加载轮次 {round_name} 的约束条件: {e}")
                    continue
        logger.info(f"约束缓存初始化完成，共加载 {len(self.round_constraints)} 个轮次的约束")
    
    def update_round_constraints(self, round_name: str, constraint_config: Dict[str, Any]) -> None:
        """
        根据constraint_config更新指定轮次的约束值
        
        Args:
            round_name (str): 轮次名称
            constraint_config (Dict[str, Any]): 约束配置，可能包含price_upper_limits, price_lower_limits, volume_limits等
        """
        if round_name not in self.round_constraints:
            logger.warning(f"轮次 {round_name} 不存在于约束缓存中")
            return
        
        # 更新价格上限（支持新旧字段名）
        price_upper_limits = constraint_config.get('price_upper_limits')
        if price_upper_limits and round_name in price_upper_limits:
            self.round_constraints[round_name]['upper_price_limit'] = float(price_upper_limits[round_name])
            logger.debug(f"更新轮次 {round_name} 价格上限: {price_upper_limits[round_name]}")
        elif constraint_config.get('price_upper_limit') is not None:
            # 兼容旧的单值格式
            self.round_constraints[round_name]['upper_price_limit'] = float(constraint_config['price_upper_limit'])
            logger.debug(f"更新轮次 {round_name} 价格上限: {constraint_config['price_upper_limit']}")
        
        # 更新价格下限（支持新旧字段名）
        price_lower_limits = constraint_config.get('price_lower_limits')
        if price_lower_limits and round_name in price_lower_limits:
            self.round_constraints[round_name]['lower_price_limit'] = float(price_lower_limits[round_name])
            logger.debug(f"更新轮次 {round_name} 价格下限: {price_lower_limits[round_name]}")
        elif constraint_config.get('price_lower_limit') is not None:
            # 兼容旧的单值格式
            self.round_constraints[round_name]['lower_price_limit'] = float(constraint_config['price_lower_limit'])
            logger.debug(f"更新轮次 {round_name} 价格下限: {constraint_config['price_lower_limit']}")
        
        # 更新投放总量限制
        volume_limits = constraint_config.get('volume_limits')
        if volume_limits and round_name in volume_limits:
            self.round_constraints[round_name]['total_quantity'] = float(volume_limits[round_name])
            logger.debug(f"更新轮次 {round_name} 投放总量: {volume_limits[round_name]}")
    
    def update_all_round_constraints(self, constraint_config: Dict[str, Any]) -> None:
        """
        根据constraint_config更新所有轮次的约束值
        
        Args:
            constraint_config (Dict[str, Any]): 约束配置
        """
        for round_name in self.rounds:
            self.update_round_constraints(round_name, constraint_config)
    
    def get_existing_allocations(self) -> Dict[str, pd.Series]:
        """
        获取已有分配数据
        >0 是因为非空的值都已经给赋值0了
        Returns:
            Dict[str, pd.Series]: 各轮次的已有分配数据
        """
        existing_allocations = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                # 获取非零分配
                allocation = self.sheet1_data[round_name]
                non_zero_allocation = allocation[allocation > 0]
                if len(non_zero_allocation) > 0:
                    existing_allocations[round_name] = non_zero_allocation
        
        return existing_allocations
    
    def get_round_constraints(self, round_name: str) -> Dict[str, float]:
        """
        获取指定轮次的约束条件
        
        优先从缓存中获取，如果缓存中没有则从Sheet2中读取
        
        Args:
            round_name (str): 轮次名称
            
        Returns:
            Dict[str, float]: 约束条件字典
        """
        # 优先从缓存中获取
        if round_name in self.round_constraints:
            return self.round_constraints[round_name].copy()
        
        # 如果缓存中没有，从Sheet2中读取
        if round_name not in self.sheet2_data.columns:
            raise ValueError(f"轮次 {round_name} 不存在")
        
        constraints = {
            'upper_price_limit': float(self.sheet2_data.loc['单箱均价上限', round_name]),
            'lower_price_limit': float(self.sheet2_data.loc['单箱均价下限', round_name]),
            'total_quantity': float(self.sheet2_data.loc['总量', round_name])
        }
        
        # 更新缓存
        self.round_constraints[round_name] = constraints.copy()
        
        return constraints
    
    def get_product_data(self) -> pd.DataFrame:
        """
        获取产品基础数据
        
        Returns:
            pd.DataFrame: 产品基础信息
        """
        return self.sheet1_data.copy()
    
    def get_constraint_data(self) -> pd.DataFrame:
        """
        获取约束数据
        
        Returns:
            pd.DataFrame: 约束条件数据
        """
        return self.sheet2_data.copy()
    
    def get_rounds(self) -> List[str]:
        """
        获取轮次列表
        
        Returns:
            List[str]: 轮次名称列表
        """
        return self.rounds.copy()
    
    def get_allocation_matrix(self) -> pd.DataFrame:
        """
        获取分配矩阵
        
        Returns:
            pd.DataFrame: 分配矩阵（产品 x 轮次）
        """
        allocation_cols = [col for col in self.rounds if col in self.sheet1_data.columns]
        return self.sheet1_data[allocation_cols].copy()
    
    def get_demand_vector(self) -> pd.Series:
        """
        获取需求向量
        
        Returns:
            pd.Series: 需求数据
        """
        return self.sheet1_data['需求'].copy()
    
    def get_supply_vector(self) -> pd.Series:
        """
        获取供应向量
        
        Returns:
            pd.Series: 可用货源数据
        """
        return self.sheet1_data['可用货源'].copy()
    
    def get_price_vector(self) -> pd.Series:
        """
        获取价格向量
        
        Returns:
            pd.Series: 单箱价格数据
        """
        return self.sheet1_data['单箱价格'].copy()
    
    def get_product_info(self) -> pd.DataFrame:
        """
        获取产品基本信息
        
        Returns:
            pd.DataFrame: 产品基本信息
        """
        info_cols = ['代码', '卷烟名称', '品牌', '类', '单箱价格']
        available_cols = [col for col in info_cols if col in self.sheet1_data.columns]
        return self.sheet1_data[available_cols].copy()
    
    def calculate_total_allocation_by_round(self) -> pd.Series:
        """
        计算各轮次总分配量
        
        Returns:
            pd.Series: 各轮次总分配量
        """
        round_totals = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                round_totals[round_name] = self.sheet1_data[round_name].sum()
            else:
                round_totals[round_name] = 0
        
        return pd.Series(round_totals)
    
    def calculate_average_price_by_round(self) -> pd.Series:
        """
        计算各轮次平均价格
        
        Returns:
            pd.Series: 各轮次平均价格
        """
        round_avg_prices = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                allocation = self.sheet1_data[round_name]
                total_allocation = allocation.sum()
                if total_allocation > 0:
                    total_sales = (allocation * self.sheet1_data['单箱价格']).sum()
                    round_avg_prices[round_name] = total_sales / total_allocation
                else:
                    round_avg_prices[round_name] = 0
            else:
                round_avg_prices[round_name] = 0
        
        return pd.Series(round_avg_prices)
    

    
    def calculate_round_sales(self, allocation_series: pd.Series) -> float:
        """
        计算轮次销售总额
        
        Args:
            allocation_series (pd.Series): 分配数量序列
            
        Returns:
            float: 销售总额
        """
        if allocation_series.empty:
            return 0.0
        
        # 确保索引对齐
        aligned_allocation = allocation_series.reindex(self.sheet1_data.index, fill_value=0)
        sales = (aligned_allocation * self.sheet1_data['单箱价格']).sum()
        return float(sales)
    
    def calculate_round_avg_price(self, allocation_series: pd.Series) -> float:
        """
        计算轮次单箱均价
        
        Args:
            allocation_series (pd.Series): 分配数量序列
            
        Returns:
            float: 单箱均价
        """
        if allocation_series.empty or allocation_series.sum() == 0:
            return 0.0
        
        # 确保索引对齐
        aligned_allocation = allocation_series.reindex(self.sheet1_data.index, fill_value=0)
        total_sales = (aligned_allocation * self.sheet1_data['单箱价格']).sum()
        total_quantity = aligned_allocation.sum()
        
        return float(total_sales / total_quantity) if total_quantity > 0 else 0.0
    
    def export_data_summary(self, output_path: str) -> None:
        """
        导出数据摘要
        
        Args:
            output_path (str): 输出文件路径
        """
        summary = {
            'data_info': {
                'total_products': len(self.sheet1_data),
                'total_rounds': len(self.rounds),
                'rounds': self.rounds
            },
            'allocation_summary': self.calculate_total_allocation_by_round().to_dict(),
            'price_summary': self.calculate_average_price_by_round().to_dict(),
            'constraint_summary': self.sheet2_data.to_dict()
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"数据摘要已导出到: {output_path}")


if __name__ == "__main__":
    # 测试代码
    loader = DataLoader(r"D:\2-code\huoyuan\data\huoyuanfenpei.xlsx")
    
    try:
        # 加载测试数据
        loader.load_data()
        
        # 获取约束条件
        for round_name in loader.get_rounds():
            constraints = loader.get_round_constraints(round_name)
            print(f"{round_name}约束条件:", constraints)
        
        # 导出数据摘要
        loader.export_data_summary(r"D:/2-code/huoyuan/output/data_summary.json")
        
        print("数据加载测试完成")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")