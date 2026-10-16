"""Cigarette supply-allocation system: constraint management.

Manage and validate demand fulfillment, average case price, round volume, special C-category rules, and allocation-strategy constraints."""
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from data_loader import DataLoader


class ConstraintManager:
    """Manage and validate all constraints for cigarette supply allocation."""
    
    def __init__(self, data_loader, constraint_config=None):
        """Initialize the constraint manager.

Args:
    data_loader: Data-loader instance.
    constraint_config: Configuration dictionary, including volume_tolerance."""
        self.data_loader = data_loader
        self.product_data = data_loader.get_product_data()  # Retrieve complete product data
        self.constraint_config = constraint_config or {}
        
        # Initialise constraint parameters (support dynamic configuration)
        self._init_constraint_parameters()
        
        # Retrieve rounds dynamically, consistently with data_loader
        self.rounds = data_loader.get_rounds()
        self.round_constraints = {}
        for round_name in self.rounds:
            try:
                # Retrieve default constraint values from data_loader
                constraints = data_loader.get_round_constraints(round_name)
                
                # Prefer round-level constraint parameters in constraint_config
                price_upper = constraints['upper_price_limit']
                price_lower = constraints['lower_price_limit']
                total_quantity = constraints['total_quantity']
                
                # Override defaults with round-level constraint parameters in constraint_config, if provided
                if self.price_upper_limits and round_name in self.price_upper_limits:
                    price_upper = self.price_upper_limits[round_name]
                    
                if self.price_lower_limits and round_name in self.price_lower_limits:
                    price_lower = self.price_lower_limits[round_name]
                    
                if self.volume_limits and round_name in self.volume_limits:
                    total_quantity = self.volume_limits[round_name]
                
                # Correct constraint field names to match the format returned by data_loader
                self.round_constraints[round_name] = {
                    'price_upper': price_upper,
                    'price_lower': price_lower,
                    'total_target': total_quantity,
                    'total_upper': total_quantity * (1 + self.volume_tolerance),  # Dynamic tolerance
                    'total_lower': total_quantity * (1 - self.volume_tolerance)   # Dynamic tolerance
                }
                
                logger.debug(f"Initializing round {round_name} constraints: price_upper={price_upper}, "
                           f"price_lower={price_lower}, total_quantity={total_quantity}")
                           
            except (ValueError, KeyError) as e:
                logger.warning(f"Cannot obtain round {round_name}: {e}")
                continue
        
        self.existing_allocations = data_loader.get_existing_allocations()
        
        # Create indicators for C-category cigarettes and price-based SKUs
        self._create_auxiliary_flags()
    
    def _init_constraint_parameters(self):
        """Initialize constraint parameters, allowing dynamic configuration from constraint_config."""
        # Total allocation tolerance, default 0.5%
        self.volume_tolerance = self.constraint_config.get('volume_tolerance', 0.005)
        
        # Required proportion of price-based SKUs, default 30%
        self.price_based_ratio = self.constraint_config.get('price_based_ratio', 0.3)
        
        # C-category cigarette constraint parameters
        self.c_type_ratio = self.constraint_config.get('c_type_ratio', 0.4)  # Default 40%
        self.c_type_volume_limit = self.constraint_config.get('c_type_volume_limit', 4900)  # Default 4900 cases
        
        # Long-subtype constraint parameters
        self.chang_type_ratio = self.constraint_config.get('chang_type_ratio', 0.2)  # Default 20%
        self.chang_type_volume_limit = self.constraint_config.get('chang_type_volume_limit', 1000)  # Default 1000 cases
        
        # Slim-subtype constraint parameters
        self.xi_type_ratio = self.constraint_config.get('xi_type_ratio', 0.6)  # Default 60%
        self.xi_type_volume_limit = self.constraint_config.get('xi_type_volume_limit', 3000)  # Default 3000 cases
        
        # Round-level constraint parameters (override dataloader defaults)
        self.price_upper_limits = self.constraint_config.get('price_upper_limits', None)  # Upper mean price per case in each round
        self.price_lower_limits = self.constraint_config.get('price_lower_limits', None)  # Lower mean price per case in each round
        self.volume_limits = self.constraint_config.get('volume_limits', None)  # Total allocation limits for each round
        
        logger.debug(f"Constraint parameters initialized: volume_tolerance={self.volume_tolerance}, "
                    f"price_based_ratio={self.price_based_ratio}, c_type_ratio={self.c_type_ratio}, "
                    f"c_type_volume_limit={self.c_type_volume_limit}, chang_type_ratio={self.chang_type_ratio}, "
                    f"chang_type_volume_limit={self.chang_type_volume_limit}, xi_type_ratio={self.xi_type_ratio}, "
                    f"xi_type_volume_limit={self.xi_type_volume_limit}, price_upper_limits={self.price_upper_limits}, "
                    f"price_lower_limits={self.price_lower_limits}, volume_limits={self.volume_limits}")
    
    def _create_auxiliary_flags(self):
        """Create is_c_type and is_price_based auxiliary flags.

Read C-category information from the C and Attribute_Shu columns of allocation_template.xlsx."""
        # Create a C-category cigarette indicator: a nonempty C column identifies C-category cigarettes
        if 'C' in self.product_data.columns:
            self.product_data['is_c_type'] = self.product_data['C'].notna() & (self.product_data['C'].str.strip() != '')
        else:
            logger.warning('C column missing; cannot identify C-category products')
            self.product_data['is_c_type'] = False
        
        # Create a price-based SKU indicator: the price-based column contains the price marker
        if 'Price_Based' in self.product_data.columns:
            self.product_data['is_price_based'] = self.product_data['Price_Based'].notna() & self.product_data['Price_Based'].str.contains('Price_Based', na=False)
        else:
            logger.warning('Price_Based column missing; cannot identify price-based SKUs')
            self.product_data['is_price_based'] = False
        
        # Create a demand-based SKU indicator: the demand-based column contains the demand marker
        if 'Demand_Based' in self.product_data.columns:
            self.product_data['is_demand_based'] = self.product_data['Demand_Based'].notna() & self.product_data['Demand_Based'].str.contains('Demand_Based', na=False)
        else:
            logger.warning('Demand_Based column missing; cannot identify demand-based SKUs')
            self.product_data['is_demand_based'] = False
        
        # Retrieve C-category cigarette subtypes from the subtype column
        if 'C_Category' in self.product_data.columns:
            self.product_data['c_subtype'] = self.product_data['C_Category'].fillna('')
        else:
            logger.warning('Attribute_Shu column missing; cannot identify C-category subtypes')
            self.product_data['c_subtype'] = ''
        
        logger.info(f"Identified {self.product_data['is_c_type'].sum()} C-category SKUs")
        logger.info(f"Identified {self.product_data['is_price_based'].sum()} price-based SKUs")
        logger.info(f"Identified {self.product_data['is_demand_based'].sum()} demand-based SKUs")
        
    def update_config(self, constraint_config):
        """TODO: requires revision; the original note states that this is not quite correct.

Update all relevant ConstraintManager parameters from constraint_config.

Args:
    constraint_config (Dict): Constraint-parameter configuration."""
        logger.info('Updating ConstraintManager configuration parameters')
        
        # Update constraint configuration
        if constraint_config:
            self.constraint_config.update(constraint_config)
            
            # Reinitialise constraint parameters
            self._init_constraint_parameters()
            
            # # Update round constraints in data_loader if round-level settings exist
            # if any(key in constraint_config for key in ['price_upper_limits', 'price_lower_limits', 'volume_limits']):
            #     self.data_loader.update_all_round_constraints(constraint_config)
            #     logger.debug("Update round constraint settings in data_loader")
            
            # Reset round constraints to apply the new configuration
            self.reset_round_constraints()
            
            logger.info('ConstraintManager configuration update complete')
    
    def reset_round_constraints(self, constraint_config=None):
        """TODO: requires revision; the original note states that this is not quite correct.

Reset round constraints from the latest data_loader data to avoid applying incorrect values.

Args:
    constraint_config: Configuration dictionary, including volume_tolerance."""
        logger.info('Resetting round constraints')
        
        # Update constraint configuration and tolerance
        if constraint_config:
            self.constraint_config = constraint_config
            self._init_constraint_parameters()
        
        self.round_constraints = {}
        for round_name in self.rounds:
            try:
                constraints = self.data_loader.get_round_constraints(round_name)
                # Correct constraint field names to match the format returned by data_loader
                self.round_constraints[round_name] = {
                    'price_upper': constraints['upper_price_limit'],
                    'price_lower': constraints['lower_price_limit'],
                    'total_target': constraints['total_quantity'],
                    'total_upper': constraints['total_quantity'] * (1 + self.volume_tolerance),  # Dynamic tolerance
                    'total_lower': constraints['total_quantity'] * (1 - self.volume_tolerance)   # Dynamic tolerance
                }
                logger.debug(f"Resetting round {round_name} constraints: {self.round_constraints[round_name]}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Cannot reset round {round_name}: {e}")
                continue
        
    def validate_demand_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate demand fulfillment.

Constraint 1: each SKU total across all rounds equals its Sheet1 Demand value.

Args:
    allocation_matrix (pd.DataFrame): SKU-by-round allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        # Calculate total allocation for each SKU
        total_allocations = allocation_matrix.sum(axis=1)
        demands = self.product_data['Demand']
        
        # Check demand fulfilment
        for idx, (total_alloc, demand) in enumerate(zip(total_allocations, demands)):
            product_name = self.product_data.iloc[idx]['Cigarette_Name']
            diff = abs(total_alloc - demand)
            
            # Allow a small numerical error (0.001)
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
        """Validate round-volume constraints.

Constraint 3: round volume may vary by +/-0.5% around its Sheet2 target.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
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
            
            # Retrieve constraints, preferring configured parameters over defaults
            target = None
            upper_limit = None
            lower_limit = None
            
            # Prefer round-level constraint parameters in constraint_config
            if self.volume_limits and round_name in self.volume_limits:
                target = self.volume_limits[round_name]
                upper_limit = target * (1 + self.volume_tolerance)
                lower_limit = target * (1 - self.volume_tolerance)
            elif round_name in self.round_constraints:
                target = self.round_constraints[round_name]['total_target']
                upper_limit = self.round_constraints[round_name]['total_upper']
                lower_limit = self.round_constraints[round_name]['total_lower']
            
            # Check for constraint violations
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
        """Validate average case-price constraints.

Constraint 2: average case price for each round must lie between the Sheet2 bounds.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
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
            
            # Calculate the mean price per case locally for this round
            total_revenue = 0
            for idx, quantity in allocation.items():
                if quantity > 0:
                    wholesale_price = self.product_data.loc[idx, 'Wholesale_Price']
                    stick_ratio = self.product_data.loc[idx, 'Carton_Stick_Ratio']
                    # Round revenue for a SKU = allocated quantity × wholesale price × 50000 / carton-to-stick ratio
                    revenue = quantity * wholesale_price * 50000 / stick_ratio
                    total_revenue += revenue
            
            # Mean price per case = round revenue / total allocated quantity
            avg_price = total_revenue / total_quantity
            
            # Retrieve constraints, preferring configured parameters over defaults
            price_upper = None
            price_lower = None
            
            # Prefer round-level constraint parameters in constraint_config
            if self.price_upper_limits and round_name in self.price_upper_limits:
                price_upper = self.price_upper_limits[round_name]
            elif round_name in self.round_constraints:
                price_upper = self.round_constraints[round_name]['price_upper']
            
            if self.price_lower_limits and round_name in self.price_lower_limits:
                price_lower = self.price_lower_limits[round_name]
            elif round_name in self.round_constraints:
                price_lower = self.round_constraints[round_name]['price_lower']
            
            # Check for constraint violations
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
        """Validate fixed allocations.

Constraint 4: allocations already specified numerically in Sheet1 must remain unchanged.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
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
            
            # Check whether existing allocations were changed
            for idx in existing.index:
                exist_val = existing[idx]
                curr_val = current[idx]
                if exist_val > 0 and abs(exist_val - curr_val) > eps:
                    product_name = self.product_data.loc[idx, 'Cigarette_Name']
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
        """Validate the first-round supply upper bound.

Constraint 7: first-round allocation must not exceed available supply.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }

        eps = 1e-6
        
        # Retrieve the first round
        first_round = min(self.rounds) if self.rounds else None
        if not first_round or first_round not in allocation_matrix.columns:
            return results
        
        first_round_allocation = allocation_matrix[first_round]
        
        # Check whether each SKU's first-round allocation exceeds available supply
        for idx, allocation in first_round_allocation.items():
            if allocation > 0:
                available_supply = self.product_data.loc[idx, 'Available_Supply']
                if allocation > available_supply + eps:
                    product_name = self.product_data.loc[idx, 'Cigarette_Name']
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
        """Validate demand-based SKU priority.

Constraint 8: SKUs whose Demand_Based field contains the demand marker should be allocated preferentially in the first two rounds.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
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
            product_name = self.product_data.loc[idx, 'Cigarette_Name']
            
            # Calculate allocations in the first two rounds and subsequent rounds
            first_two_allocation = 0
            later_allocation = 0
            
            for round_name in first_two_rounds:
                if round_name in allocation_matrix.columns:
                    first_two_allocation += allocation_matrix.loc[idx, round_name]
            
            for round_name in later_rounds:
                if round_name in allocation_matrix.columns:
                    later_allocation += allocation_matrix.loc[idx, round_name]
            
            total_allocation = first_two_allocation + later_allocation
            
            # If allocation exists, check whether the first two rounds are prioritised
            if total_allocation > 0:
                first_two_ratio = first_two_allocation / total_allocation
                
                # Require at least 100% of allocation in the first two rounds (this threshold is adjustable)
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
        """Validate C-category constraints.

Constraint 10: special C-category rules.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
        results = {
            'is_valid': True,
            'violations': [],
            'details': {}
        }
        
        # Retrieve C-category cigarette data
        c_type_mask = self.product_data['is_c_type']
        c_type_data = self.product_data[c_type_mask]
        
        if len(c_type_data) == 0:
            return results
        
        # Constraint 10.1: each round must not exceed the specified share of total allocation
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            round_allocation = allocation_matrix[round_name]
            total_round = round_allocation.sum()
            c_type_round = round_allocation[c_type_mask].sum()
            
            if total_round > 0 and c_type_round / total_round > self.c_type_ratio:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'C-category share per round',
                    'round': round_name,
                    'actual_ratio': c_type_round / total_round,
                    'limit': self.c_type_ratio
                })
        
        # Constraint 10.2: each round must not exceed the specified total quantity
        for round_name in self.rounds:
            if round_name not in allocation_matrix.columns:
                continue
                
            c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
            if c_type_round > self.c_type_volume_limit:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'C-category volume per round',
                    'round': round_name,
                    'actual_total': c_type_round,
                    'limit': self.c_type_volume_limit
                })
        
        # Constraint 10.3: concentrate the square-pack subtype in one round
        fang_mask = self.product_data['c_subtype'].str.contains('Square', na=False)
        if fang_mask.any():
            fang_allocation = allocation_matrix.loc[fang_mask]
            non_zero_rounds = (fang_allocation > 0).sum(axis=0)
            if (non_zero_rounds > 0).sum() > 1:
                results['is_valid'] = False
                results['violations'].append({
                    'constraint': 'Concentrated allocation of square C-category products',
                    'actual_rounds': (non_zero_rounds > 0).sum(),
                    'limit': 1
                })
        
        # Constraint 10.4: the long subtype must not exceed 20% of C-category allocation per round, with each round ≤1000
        chang_mask = self.product_data['c_subtype'].str.contains('Long', na=False)
        if chang_mask.any():
            chang_allocation = allocation_matrix.loc[chang_mask]
            
            for round_name in self.rounds:
                if round_name not in allocation_matrix.columns:
                    continue
                    
                c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
                chang_round = chang_allocation[round_name].sum()
                
                # Total quantity limit per round
                if chang_round > self.chang_type_volume_limit:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'Long C-category volume per round',
                        'round': round_name,
                        'actual_total': chang_round,
                        'limit': self.chang_type_volume_limit
                    })
                
                # Proportion limit
                if c_type_round > 0 and chang_round / c_type_round > self.chang_type_ratio:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'Long C-category share per round',
                        'round': round_name,
                        'actual_ratio': chang_round / c_type_round,
                        'limit': self.chang_type_ratio
                    })
        
        # Constraint 10.5: the slim subtype must not exceed 60% of C-category allocation per round, with each round ≤3000
        xi_mask = self.product_data['c_subtype'].str.contains('Slim', na=False)
        if xi_mask.any():
            xi_allocation = allocation_matrix.loc[xi_mask]
            
            for round_name in self.rounds:
                if round_name not in allocation_matrix.columns:
                    continue
                    
                c_type_round = allocation_matrix.loc[c_type_mask, round_name].sum()
                xi_round = xi_allocation[round_name].sum()
                
                # Total quantity limit per round
                if xi_round > self.xi_type_volume_limit:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'Slim C-category volume per round',
                        'round': round_name,
                        'actual_total': xi_round,
                        'limit': self.xi_type_volume_limit
                    })
                
                # Proportion limit
                if c_type_round > 0 and xi_round / c_type_round > self.xi_type_ratio:
                    results['is_valid'] = False
                    results['violations'].append({
                        'constraint': 'Slim C-category share per round',
                        'round': round_name,
                        'actual_ratio': xi_round / c_type_round,
                        'limit': self.xi_type_ratio
                    })
        
        results['details']['total_violations'] = len(results['violations'])
        
        return results
    
    def validate_price_based_constraints(self, allocation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Validate price-based SKU constraints.

Constraint 9: SKUs whose Price_Based field contains the price marker must represent at least 30% of SKUs in each round.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Validation result."""
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
            
            # SKUs allocated in this round
            allocated_products = round_allocation > 0
            total_allocated_products = allocated_products.sum()
            
            # Price-based SKUs allocated in this round
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
        """Validate all constraints.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.

Returns:
    Dict[str, Any]: Complete validation results."""
        logger.info('Starting validation of all constraints')
        
        all_results = {
            'overall_valid': True,
            'constraint_results': {},
            'summary': {}
        }
        
        # Select constraints to validate according to the configuration
        constraint_validators = {}
        
        # Demand fulfilment constraint - usually always enabled
        if self.constraint_config.get('enable_demand_constraints', True):
            constraint_validators['demand'] = self.validate_demand_constraints
            
        # Mean price per case constraint
        if self.constraint_config.get('enable_price_constraints', True):
            constraint_validators['price'] = self.validate_price_constraints
            
        # Total allocation constraint
        if self.constraint_config.get('enable_volume_constraints', True):
            constraint_validators['volume'] = self.validate_volume_constraints
            
        # Fixed allocation constraint - usually always enabled (protect existing allocations)
        constraint_validators['fixed_allocation'] = self.validate_fixed_allocation_constraints
        
        # First-round supply cap constraint - usually always enabled (protect supply limits)
        constraint_validators['first_round_supply'] = self.validate_first_round_supply_constraints
        
        # Demand-based priority constraint
        if self.constraint_config.get('enable_demand_based_constraints', True):
            constraint_validators['demand_based_priority'] = self.validate_demand_based_priority_constraints
            
        # Price-based proportion constraint
        if self.constraint_config.get('enable_price_based_constraints', True):
            constraint_validators['price_based'] = self.validate_price_based_constraints
            
        # C-category cigarette constraints
        if self.constraint_config.get('enable_c_type_constraints', True):
            constraint_validators['c_type'] = self.validate_c_type_constraints
        
        total_violations = 0
        
        # Record skipped constraints
        skipped_constraints = []
        all_constraint_names = [
            'demand', 'price', 'volume', 'fixed_allocation', 'first_round_supply',
            'demand_based_priority', 'price_based', 'c_type'
        ]
        
        for constraint_name in all_constraint_names:
            if constraint_name not in constraint_validators:
                skipped_constraints.append(constraint_name)
                logger.info(f"Skipping {constraint_name} constraint validation (disabled)")
        
        for constraint_name, validator in constraint_validators.items():
            try:
                result = validator(allocation_matrix)
                all_results['constraint_results'][constraint_name] = result
                
                if not result['is_valid']:
                    all_results['overall_valid'] = False
                    total_violations += result['details'].get('total_violations', 0)
                    
                logger.info(f"{constraint_name}Constraint validation complete: {'passed' if result['is_valid'] else 'violated'}")
                
            except Exception as e:
                logger.error(f"Error validating {constraint_name} constraint: {str(e)}")
                all_results['constraint_results'][constraint_name] = {
                    'is_valid': False,
                    'error': str(e)
                }
                all_results['overall_valid'] = False
        
        # Generate the summary
        all_results['summary'] = {
            'total_violations': total_violations,
            'violated_constraints': [name for name, result in all_results['constraint_results'].items() 
                                   if not result.get('is_valid', False)],
            'passed_constraints': [name for name, result in all_results['constraint_results'].items() 
                                 if result.get('is_valid', False)],
            'skipped_constraints': skipped_constraints,
            'enabled_constraints': list(constraint_validators.keys())
        }
        
        logger.info(f"Constraint validation complete: {'all passed' if all_results['overall_valid'] else f'Found {total_violations} violations'}")
        logger.info(f"Enabled constraints: {list(constraint_validators.keys())}")
        logger.info(f"Skipped constraints: {skipped_constraints}")
        
        return all_results
    
    def get_constraint_weights(self) -> Dict[str, float]:
        """Return constraint weights for the optimization objective.

Returns:
    Dict[str, float]: Constraint-weight dictionary."""
        return {
            'demand_satisfaction': 1000.0,      # Give demand fulfilment the highest weight
            'price_deviation': 100.0,           # Price deviation weight
            'volume_deviation': 50.0,           # Total quantity deviation weight
            'balance_penalty': 10.0,            # Allocation balance weight
            'c_type_penalty': 200.0,            # C-category cigarette constraint weight
            'priority_bonus': 20.0              # Priority reward weight
        }

