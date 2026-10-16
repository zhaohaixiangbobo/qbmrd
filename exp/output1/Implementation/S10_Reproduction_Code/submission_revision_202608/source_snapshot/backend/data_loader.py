"""Cigarette supply-allocation system: data loading.

Load supply data and allocation limits from Excel, then validate and preprocess the data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Load cigarette supply-allocation data from Excel.

Sheet1 contains supply preallocation data; Sheet2 contains per-round control requirements."""
    
    def __init__(self, file_path: str):
        """Initialize the data loader.

Args:
    file_path (str): Excel workbook path."""
        self.file_path = file_path
        self.sheet1_data = None
        self.sheet2_data = None
        self.rounds = []  # Detect rounds dynamically
        self.round_constraints = {}  # Cache round constraint data
        
        # Load data automatically
        self.load_data()
        
    def load_data(self) -> None:
        """Load all data from Excel and preprocess it."""
        try:
            logger.info(f"Loading data file: {self.file_path}")
            
            # Load Sheet1 data
            self.sheet1_data = pd.read_excel(self.file_path, sheet_name='Sheet1')
            logger.info(f"Sheet1 loaded: {len(self.sheet1_data)} data rows")
            
            # Load Sheet2 data
            self.sheet2_data = pd.read_excel(self.file_path, sheet_name='Sheet2', index_col=0)
            logger.info(f"Sheet2 loaded: {len(self.sheet2_data)} constraint rows")
            
            # Detect rounds dynamically
            self._detect_rounds()
            
            # Validate data
            self._validate_data()
            
            # Preprocess data
            self._preprocess_data()
            
            # Initialise the constraint cache
            self._initialize_constraints_cache()
            
            logger.info('Data loading and preprocessing complete')
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _detect_rounds(self) -> None:
        """Detect allocation rounds dynamically."""
        # Detect rounds from Sheet1 as a fallback
        sheet1_rounds = [col for col in self.sheet1_data.columns 
                        if 'Round_' in col and col.startswith('Round_')]
        
        # Take the intersection of both sets to ensure data consistency
        cn_num_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        self.rounds = sorted(list(set(sheet1_rounds)),key=lambda x: cn_num_map.get(x[6], 0))
        
        if not self.rounds:
            # Use the first 5 rounds by default if no rounds are detected
            self.rounds = ['Round_1', 'Round_2', 'Round_3', 'Round_4', 'Round_5']
            logger.warning('No rounds detected; using default rounds: ' + str(self.rounds))
        else:
            logger.info(f"Detected rounds: {self.rounds}")
    
    def _validate_data(self) -> None:
        """Validate data format and completeness.

Raises:
    ValueError: Invalid data format."""
        # Validate required Sheet1 columns
        required_sheet1_cols = ['Code', 'Wholesale_Price', 'Cigarette_Name', 'Class', 'Demand', 'Available_Supply'] + self.rounds
        missing_cols = [col for col in required_sheet1_cols if col not in self.sheet1_data.columns]
        if missing_cols:
            raise ValueError(f"Sheet1 is missing required columns: {missing_cols}")
        
        # Validate required Sheet2 rows
        required_sheet2_rows = ['Average_Case_Price_Upper', 'Average_Case_Price_Lower', 'Total_Quantity']
        missing_rows = [row for row in required_sheet2_rows if row not in self.sheet2_data.index]
        if missing_rows:
            raise ValueError(f"Sheet2 is missing required rows: {missing_rows}")
        
        # Validate required Sheet2 columns (rounds)
        missing_round_cols = [col for col in self.rounds if col not in self.sheet2_data.columns]
        if missing_round_cols:
            raise ValueError(f"Sheet2 is missing round columns: {missing_round_cols}")
        
        logger.info('Data validation passed')
    
    def _preprocess_data(self) -> None:
        """Preprocess data: clean values, create auxiliary columns, and handle missing values."""
        # Process Sheet1 data
        # Remove line breaks from column names
        # self.sheet1_data.columns = [col.replace('\n', '') for col in self.sheet1_data.columns]
        
        # Detect rounds again because column names may have changed
        sheet1_rounds = [col for col in self.sheet1_data.columns 
                        if 'Round_' in col and col.startswith('Round_')]
        cn_num_map = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        self.rounds = sorted(list(set(self.rounds) & set(sheet1_rounds)),key=lambda x: cn_num_map.get(x[6], 0))

        
        # Handle missing values
        self.sheet1_data['Demand'] = pd.to_numeric(self.sheet1_data['Demand'], errors='coerce').fillna(0)
        self.sheet1_data['Available_Supply'] = pd.to_numeric(self.sheet1_data['Available_Supply'], errors='coerce').fillna(0)
        self.sheet1_data['Wholesale_Price'] = pd.to_numeric(self.sheet1_data['Wholesale_Price'], errors='coerce').fillna(0)
        
        # Process round allocation data
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                self.sheet1_data[round_name] = pd.to_numeric(self.sheet1_data[round_name], errors='coerce').fillna(0)
        
        # Create helper columns
        self._create_auxiliary_columns()
        
        logger.info('Data preprocessing complete')
    
    def _create_auxiliary_columns(self) -> None:
        """Create auxiliary columns, including case price and total allocation."""
        # Calculate the price per case (assuming 50 cartons per case and 20 sticks per carton)
        if 'Carton_Stick_Ratio' in self.sheet1_data.columns:
            # Use the actual carton-to-stick ratio if that column exists
            self.sheet1_data['Carton_Stick_Ratio'] = pd.to_numeric(self.sheet1_data['Carton_Stick_Ratio'], errors='coerce').fillna(200)
            self.sheet1_data['Case_Price'] = self.sheet1_data['Wholesale_Price'] * 50000 / self.sheet1_data['Carton_Stick_Ratio']
        else:
            # Default to 20 sticks per carton
            self.sheet1_data['Case_Price'] = self.sheet1_data['Wholesale_Price'] * 2500  # 50 cartons * 50 yuan/carton
        
        # Calculate total allocation
        allocation_cols = [col for col in self.rounds if col in self.sheet1_data.columns]
        if allocation_cols:
            self.sheet1_data['Total_Allocation'] = self.sheet1_data[allocation_cols].sum(axis=1)
        else:
            self.sheet1_data['Total_Allocation'] = 0
        
        # Calculate the allocation rate
        self.sheet1_data['Allocation_Rate'] = np.where(
            self.sheet1_data['Demand'] > 0,
            self.sheet1_data['Total_Allocation'] / self.sheet1_data['Demand'],
            1
        )
        
        logger.info('Auxiliary columns created')
    
    def _initialize_constraints_cache(self) -> None:
        """Initialize the constraint cache with the constraints for every round from Sheet2."""
        logger.info('Initializing constraint cache')
        self.round_constraints = {}
        for round_name in self.rounds:
            if round_name in self.sheet2_data.columns:
                try:
                    self.round_constraints[round_name] = {
                        'upper_price_limit': float(self.sheet2_data.loc['Average_Case_Price_Upper', round_name]),
                        'lower_price_limit': float(self.sheet2_data.loc['Average_Case_Price_Lower', round_name]),
                        'total_quantity': float(self.sheet2_data.loc['Total_Quantity', round_name])
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"Cannot load constraints for round {round_name}: {e}")
                    continue
        logger.info(f"Constraint cache initialized for {len(self.round_constraints)} rounds")
    
    def update_round_constraints(self, round_name: str, constraint_config: Dict[str, Any]) -> None:
        """Update constraints for the specified round from constraint_config.

Args:
    round_name (str): Round name.
    constraint_config (Dict[str, Any]): Configuration, including optional price_upper_limits, price_lower_limits, and volume_limits."""
        if round_name not in self.round_constraints:
            logger.warning(f"Round {round_name} is absent from the constraint cache")
            return
        
        # Update upper price limits (support both new and legacy field names)
        price_upper_limits = constraint_config.get('price_upper_limits')
        if price_upper_limits and round_name in price_upper_limits:
            self.round_constraints[round_name]['upper_price_limit'] = float(price_upper_limits[round_name])
            logger.debug(f"Updating round {round_name} price upper bound: {price_upper_limits[round_name]}")
        elif constraint_config.get('price_upper_limit') is not None:
            # Support the legacy single-value format
            self.round_constraints[round_name]['upper_price_limit'] = float(constraint_config['price_upper_limit'])
            logger.debug(f"Updating round {round_name} price upper bound: {constraint_config['price_upper_limit']}")
        
        # Update lower price limits (support both new and legacy field names)
        price_lower_limits = constraint_config.get('price_lower_limits')
        if price_lower_limits and round_name in price_lower_limits:
            self.round_constraints[round_name]['lower_price_limit'] = float(price_lower_limits[round_name])
            logger.debug(f"Updating round {round_name} price lower bound: {price_lower_limits[round_name]}")
        elif constraint_config.get('price_lower_limit') is not None:
            # Support the legacy single-value format
            self.round_constraints[round_name]['lower_price_limit'] = float(constraint_config['price_lower_limit'])
            logger.debug(f"Updating round {round_name} price lower bound: {constraint_config['price_lower_limit']}")
        
        # Update total allocation limits
        volume_limits = constraint_config.get('volume_limits')
        if volume_limits and round_name in volume_limits:
            self.round_constraints[round_name]['total_quantity'] = float(volume_limits[round_name])
            logger.debug(f"Updating round {round_name} volume target: {volume_limits[round_name]}")
    
    def update_all_round_constraints(self, constraint_config: Dict[str, Any]) -> None:
        """Update constraints for every round from constraint_config.

Args:
    constraint_config (Dict[str, Any]): Constraint configuration."""
        for round_name in self.rounds:
            self.update_round_constraints(round_name, constraint_config)
    
    def get_existing_allocations(self) -> Dict[str, pd.Series]:
        """Return existing allocations by round.

Only positive values count because missing entries have already been filled with zero.

Returns:
    Dict[str, pd.Series]: Existing allocation data for each round."""
        existing_allocations = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                # Retrieve nonzero allocations
                allocation = self.sheet1_data[round_name]
                non_zero_allocation = allocation[allocation > 0]
                if len(non_zero_allocation) > 0:
                    existing_allocations[round_name] = non_zero_allocation
        
        return existing_allocations
    
    def get_round_constraints(self, round_name: str) -> Dict[str, float]:
        """Return constraints for the specified round, preferring the cache and falling back to Sheet2.

Args:
    round_name (str): Round name.

Returns:
    Dict[str, float]: Constraint dictionary."""
        # Retrieve from the cache first
        if round_name in self.round_constraints:
            return self.round_constraints[round_name].copy()
        
        # Read from Sheet2 if no cached value exists
        if round_name not in self.sheet2_data.columns:
            raise ValueError(f"Round {round_name} does not exist")
        
        constraints = {
            'upper_price_limit': float(self.sheet2_data.loc['Average_Case_Price_Upper', round_name]),
            'lower_price_limit': float(self.sheet2_data.loc['Average_Case_Price_Lower', round_name]),
            'total_quantity': float(self.sheet2_data.loc['Total_Quantity', round_name])
        }
        
        # Update the cache
        self.round_constraints[round_name] = constraints.copy()
        
        return constraints
    
    def get_product_data(self) -> pd.DataFrame:
        """Return basic product data.

Returns:
    pd.DataFrame: Basic product information."""
        return self.sheet1_data.copy()
    
    def get_constraint_data(self) -> pd.DataFrame:
        """Return constraint data.

Returns:
    pd.DataFrame: Constraint information."""
        return self.sheet2_data.copy()
    
    def get_rounds(self) -> List[str]:
        """Return the round-name list.

Returns:
    List[str]: Round names."""
        return self.rounds.copy()
    
    def get_allocation_matrix(self) -> pd.DataFrame:
        """Return the allocation matrix.

Returns:
    pd.DataFrame: Product-by-round allocation matrix."""
        allocation_cols = [col for col in self.rounds if col in self.sheet1_data.columns]
        return self.sheet1_data[allocation_cols].copy()
    
    def get_demand_vector(self) -> pd.Series:
        """Return the demand vector.

Returns:
    pd.Series: Demand data."""
        return self.sheet1_data['Demand'].copy()
    
    def get_supply_vector(self) -> pd.Series:
        """Return the supply vector.

Returns:
    pd.Series: Available supply data."""
        return self.sheet1_data['Available_Supply'].copy()
    
    def get_price_vector(self) -> pd.Series:
        """Return the price vector.

Returns:
    pd.Series: Case-price data."""
        return self.sheet1_data['Case_Price'].copy()
    
    def get_product_info(self) -> pd.DataFrame:
        """Return basic product information.

Returns:
    pd.DataFrame: Basic product information."""
        info_cols = ['Code', 'Cigarette_Name', 'Brand', 'Class', 'Case_Price']
        available_cols = [col for col in info_cols if col in self.sheet1_data.columns]
        return self.sheet1_data[available_cols].copy()
    
    def calculate_total_allocation_by_round(self) -> pd.Series:
        """Calculate total allocation for each round.

Returns:
    pd.Series: Total allocations by round."""
        round_totals = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                round_totals[round_name] = self.sheet1_data[round_name].sum()
            else:
                round_totals[round_name] = 0
        
        return pd.Series(round_totals)
    
    def calculate_average_price_by_round(self) -> pd.Series:
        """Calculate average price for each round.

Returns:
    pd.Series: Average prices by round."""
        round_avg_prices = {}
        for round_name in self.rounds:
            if round_name in self.sheet1_data.columns:
                allocation = self.sheet1_data[round_name]
                total_allocation = allocation.sum()
                if total_allocation > 0:
                    total_sales = (allocation * self.sheet1_data['Case_Price']).sum()
                    round_avg_prices[round_name] = total_sales / total_allocation
                else:
                    round_avg_prices[round_name] = 0
            else:
                round_avg_prices[round_name] = 0
        
        return pd.Series(round_avg_prices)
    

    
    def calculate_round_sales(self, allocation_series: pd.Series) -> float:
        """Calculate total sales for a round.

Args:
    allocation_series (pd.Series): Allocation quantities.

Returns:
    float: Total sales."""
        if allocation_series.empty:
            return 0.0
        
        # Ensure indices align
        aligned_allocation = allocation_series.reindex(self.sheet1_data.index, fill_value=0)
        sales = (aligned_allocation * self.sheet1_data['Case_Price']).sum()
        return float(sales)
    
    def calculate_round_avg_price(self, allocation_series: pd.Series) -> float:
        """Calculate average case price for a round.

Args:
    allocation_series (pd.Series): Allocation quantities.

Returns:
    float: Average case price."""
        if allocation_series.empty or allocation_series.sum() == 0:
            return 0.0
        
        # Ensure indices align
        aligned_allocation = allocation_series.reindex(self.sheet1_data.index, fill_value=0)
        total_sales = (aligned_allocation * self.sheet1_data['Case_Price']).sum()
        total_quantity = aligned_allocation.sum()
        
        return float(total_sales / total_quantity) if total_quantity > 0 else 0.0
    
    def export_data_summary(self, output_path: str) -> None:
        """Export a data summary.

Args:
    output_path (str): Output file path."""
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
        
        logger.info(f"Data summary exported to: {output_path}")


