"""Cigarette supply-allocation system: linear programming.

Implement the allocation problem in PuLP with linear constraints and a linear objective."""

import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class LinearProgrammingAllocator:
    """Solve the cigarette supply-allocation problem using PuLP linear programming."""
    
    def __init__(self, data_loader, constraint_manager=None):
        """Initialize the linear-programming allocator.

Args:
    data_loader: Data-loader instance.
    constraint_manager: Optional constraint-manager instance."""
        self.data_loader = data_loader
        self.constraint_manager = constraint_manager
        self.product_data = data_loader.get_product_data()
        self.rounds = data_loader.get_rounds()
        
        # Retrieve existing allocation data
        self.existing_allocations = data_loader.get_existing_allocations()
        
        # Linear programming model
        self.model = None
        self.variables = {}
        self.solution = None
        
    def create_variables(self) -> Dict[str, Dict[int, pulp.LpVariable]]:
        """Create decision variables.

Returns:
    Dict[str, Dict[int, pulp.LpVariable]]: Decision-variable dictionary."""
        variables = {}
        
        for round_name in self.rounds:
            variables[round_name] = {}
            
            for idx in range(len(self.product_data)):
                product_name = self.product_data.iloc[idx]['Cigarette_Name']
                demand = self.product_data.iloc[idx]['Demand']
                supply = self.product_data.iloc[idx]['Available_Supply']
                
                # Create nonnegative continuous variables
                var_name = f"{round_name}_{idx}_{product_name}"
                
                # Constraint 4: already determined SKU allocations cannot be changed
                if round_name in self.existing_allocations and idx in self.existing_allocations[round_name].index:
                    fixed_value = self.existing_allocations[round_name][idx]
                    variables[round_name][idx] = pulp.LpVariable(
                        var_name,
                        lowBound=fixed_value,
                        upBound=fixed_value,  # Fixed value
                        cat='Continuous'
                    )
                else:
                    # Constraint 7: the first round must not exceed the available supply cap
                    upper_bound = min(demand, supply if round_name == 'Round_1' else demand)
                    variables[round_name][idx] = pulp.LpVariable(
                        var_name,
                        lowBound=0,
                        upBound=upper_bound,
                        cat='Continuous'
                    )
        
        logger.info(f"Created {len(self.rounds) * len(self.product_data)} decision variables")
        return variables
    
    def add_demand_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]],
                             constraint_config: Dict[str, Any] = None) -> None:
        """Add demand-fulfillment constraints.

Constraint 1: total allocation of each SKU across all rounds equals demand.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['Demand']
            
            if demand > 0:
                # The SKU's allocation summed over all rounds must not exceed demand
                total_allocation = pulp.lpSum([
                    variables[round_name][idx] 
                    for round_name in self.rounds 
                    if round_name in variables
                ])
                
                model += total_allocation == demand, f"Demand_Product_{idx}"
        
        logger.info('Demand-fulfillment constraints added')
    
    def add_volume_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]], 
                             constraint_config: Dict[str, Any] = None) -> None:
        """Add round-volume constraints.

Constraint 3: each round total must be within its bounds.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # Total allocation in this round
            total_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in range(len(self.product_data))
            ])
            
            # Retrieve constraints
            try:
                # Prefer round-level constraint parameters
                if (constraint_config and 
                    constraint_config.get('volume_limits') and 
                    round_name in constraint_config['volume_limits']):
                    total_quantity = constraint_config['volume_limits'][round_name]
                else:
                    constraints = self.data_loader.get_round_constraints(round_name)
                    total_quantity = constraints['total_quantity']
                
                # Retrieve tolerance settings
                tolerance = constraint_config.get('volume_tolerance', 0.005) if constraint_config else 0.005
                
                # Add constraints allowing the configured tolerance
                upper_limit = total_quantity * (1 + tolerance)
                lower_limit = total_quantity * (1 - tolerance)
                
                model += total_round_allocation <= upper_limit, f"Volume_Upper_{round_name}"
                model += total_round_allocation >= lower_limit, f"Volume_Lower_{round_name}"
                
                logger.info(f"{round_name}Round-volume constraint: {lower_limit:.2f} <= volume <= {upper_limit:.2f} (target: {total_quantity:.2f})")
                
            except Exception as e:
                logger.warning(f"Cannot obtain constraints for {round_name}: {e}")
                continue
        
        logger.info('Round-volume constraints added')
    
    def add_c_type_constraints(self, model: pulp.LpProblem, 
                             variables: Dict[str, Dict[int, pulp.LpVariable]],
                             constraint_config: Dict[str, Any] = None) -> None:
        """Add the linear part of the C-category constraints.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        # Ensure the 'C' column has string type and identify C-category cigarettes
        type_column = self.product_data['C'].astype(str)
        c_type_mask = type_column.str.contains('C', na=False)
        c_type_indices = c_type_mask[c_type_mask].index.tolist()
        
        if not c_type_indices:
            logger.info('No C-category data found; skipping C-category constraints')
            return
        
        # Retrieve C-category cigarette settings
        c_type_ratio = 0.4  # Default 40%
        c_type_volume_limit = 4900  # Default 4900 cases
        
        if constraint_config:
            c_type_ratio = constraint_config.get('c_type_ratio', 0.4)
            c_type_volume_limit = constraint_config.get('c_type_volume_limit', 4900)
        
        # Constraint 10.1: C-category cigarettes must not exceed the configured share of each round's total allocation
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # C-category allocation in this round
            c_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in c_type_indices
            ])
            
            # Total allocation in this round
            total_round_allocation = pulp.lpSum([
                variables[round_name][idx] 
                for idx in range(len(self.product_data))
            ])
            
            # Linearised constraint limiting the C-category share to the configured proportion
            model += (c_round_allocation <= c_type_ratio * total_round_allocation), f"C_Type_Ratio_{round_name}"
            
            # Total C-category cigarette quantity per round must not exceed the configured limit
            model += c_round_allocation <= c_type_volume_limit, f"C_Type_Round_Limit_{round_name}"
        
        logger.info(f"C-category constraints added (share: {c_type_ratio*100:.1f}%, volume cap: {c_type_volume_limit} cases)")
    
    def _add_c_subtype_constraints(self, model: pulp.LpProblem, 
                                 variables: Dict[str, Dict[int, pulp.LpVariable]],
                                 constraint_config: Dict[str, Any] = None) -> None:
        """Add C-category subtype constraints.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        # Retrieve configuration parameters
        chang_type_ratio = 0.2  # Default 20%
        chang_type_volume_limit = 1000  # Default 1000 cases
        xi_type_ratio = 0.6  # Default 60%
        xi_type_volume_limit = 3000  # Default 3000 cases
        
        if constraint_config:
            chang_type_ratio = constraint_config.get('chang_type_ratio', chang_type_ratio)
            chang_type_volume_limit = constraint_config.get('chang_type_volume_limit', chang_type_volume_limit)
            xi_type_ratio = constraint_config.get('xi_type_ratio', xi_type_ratio)
            xi_type_volume_limit = constraint_config.get('xi_type_volume_limit', xi_type_volume_limit)
        
        # Retrieve indices for each subtype
        fang_indices = self.product_data[self.product_data['C_Category'] == 'Square'].index.tolist()
        chang_indices = self.product_data[self.product_data['C_Category'] == 'Long'].index.tolist()
        xi_indices = self.product_data[self.product_data['C_Category'] == 'Slim'].index.tolist()
        
        # Constraint 10.4: long-subtype quantity per round ≤ configured limit and no more than the configured share of C-category allocation
        if chang_indices:
            # Long-subtype quantity per round must not exceed the configured limit
            # Long-subtype quantity per round must not exceed the configured share of C-category allocation
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

        
        # Constraint 10.5: slim-subtype total quantity ≤ configured limit and no more than the configured share of C-category allocation per round
        if xi_indices:
            # Slim-subtype quantity per round must not exceed the configured limit
            # Slim-subtype quantity per round must not exceed the configured share of C-category allocation
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

        # Constraint 10.3: concentrate square-pack allocation in one round (exclude existing_allocations data)
        # Square-pack SKUs are mostly cx and have already been allocated
        if fang_indices:
            # Select square-pack SKUs outside existing_allocations
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
            
            # Apply the concentration constraint only to square-pack SKUs outside existing_allocations
            if non_existing_fang_indices:
                # Create a binary variable for each round indicating whether square-pack SKUs are allocated in that round
                round_binary = {}
                for round_name in self.rounds:
                    if round_name not in variables:
                        continue
                    round_binary[round_name] = pulp.LpVariable(
                        f"Fang_Round_{round_name}",
                        cat='Binary'
                    )
            
                # Allocation is allowed in only one round
                model += pulp.lpSum(round_binary.values()) == 1, "Fang_Single_Round"
            
                # Set all square-pack variables to 0 if the round is not selected
                M = 1000000  # A sufficiently large number
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
        """Add price-based SKU constraints.

Constraint 9: price-based SKUs must represent at least the configured share of SKUs in each round.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        # Retrieve indices of price-based SKUs
        price_based_indices = self.product_data[self.product_data['Price_Based'].str.contains('Price_Based', na=False)].index.tolist()
        
        if not price_based_indices:
            logger.info('No price-based SKU data found; skipping price-based constraints')
            return
        
        # Retrieve the configured price-based proportion
        price_based_ratio = 0.3  # Default 30%
        if constraint_config and constraint_config.get('price_based_ratio') is not None:
            price_based_ratio = constraint_config['price_based_ratio']
        
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # Create a binary variable indicating whether each SKU is allocated in the round (allocated quantity >0)
            allocation_binary = {}
            for idx in range(len(self.product_data)):
                allocation_binary[idx] = pulp.LpVariable(
                    f"Allocation_Binary_{round_name}_{idx}",
                    cat='Binary'
                )
                # If allocated quantity >0, the binary variable =1
                M = 1000000  # A sufficiently large number
                model += variables[round_name][idx] <= M * allocation_binary[idx], f"Binary_Link_Upper_{round_name}_{idx}"
                model += variables[round_name][idx] >= 1.0 * allocation_binary[idx], f"Binary_Link_Lower_{round_name}_{idx}"
            
            # Number of price-based SKUs
            price_based_count = pulp.lpSum([allocation_binary[idx] for idx in price_based_indices])
            # Total number of allocated SKUs
            total_count = pulp.lpSum(allocation_binary.values())
            
            # Price-based SKU share ≥ configured proportion
            model += price_based_count >= price_based_ratio * total_count, f"Price_Based_Ratio_{round_name}"
        
        logger.info(f"Price-based constraints added (share: {price_based_ratio*100:.1f}%)")
    
    def add_demand_based_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """Add soft demand-based SKU constraints.

Constraint 8: prioritize demand-based SKUs in the first two rounds, with a larger penalty for later allocation.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables."""
        # Retrieve indices of demand-based SKUs
        demand_based_indices = self.product_data[self.product_data['Demand_Based'].str.contains('Demand_Based', na=False)].index.tolist()
        
        if not demand_based_indices:
            logger.info('No demand-based SKU data found; skipping demand-based constraints')
            return
        
        # Create penalty variables for each demand-based SKU in subsequent rounds
        self.demand_penalty_vars = {}
        for idx in demand_based_indices:
            self.demand_penalty_vars[idx] = {}
            for i, round_name in enumerate(self.rounds[2:], start=3):  # Start from the third round
                if round_name in variables:
                    # Create a penalty variable representing allocation in this round
                    penalty_var = pulp.LpVariable(
                        f"Demand_Penalty_{idx}_{round_name}",
                        lowBound=0,
                        cat='Continuous'
                    )
                    self.demand_penalty_vars[idx][round_name] = penalty_var
                    
                    # The penalty variable equals allocation in this round
                    model += penalty_var == variables[round_name][idx], f"Demand_Penalty_Link_{idx}_{round_name}"
        
        logger.info('Soft demand-based constraints added')
    
    def add_demand_split_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """Add demand-splitting constraints only for SKUs without existing_allocations.

Constraint 6: concentrate demand below 50 cases in one round where possible; use at most two rounds for 50-100 cases and at least two rounds for 100-250 cases.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables."""
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['Demand']
            if demand <= 0:
                continue
            
            # Check whether the SKU has an already determined allocation
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] >= 0):
                    is_existing = True
                    break
            
            # Skip this constraint for SKUs with already determined allocations
            if is_existing:
                continue
            
            # Create binary variables indicating whether allocation occurs in each round
            round_binary = {}
            for round_name in self.rounds:
                if round_name not in variables:
                    continue
                round_binary[round_name] = pulp.LpVariable(
                    f"Round_Binary_{idx}_{round_name}",
                    cat='Binary'
                )
                # If allocation occurs in the round, the binary variable =1
                M = 1000000  # A sufficiently large number
                model += variables[round_name][idx] <= M * round_binary[round_name], f"Round_Binary_Link_Upper_{idx}_{round_name}"
                model += variables[round_name][idx] >= 0.01 * round_binary[round_name], f"Round_Binary_Link_Lower_{idx}_{round_name}"
            
            # Calculate the number of rounds used
            rounds_used = pulp.lpSum(round_binary.values())
            
            if demand < 50:
                # Use a soft constraint to concentrate allocation in one round where possible
                model += rounds_used <= 2, f"Small_Demand_Split_{idx}"
            elif demand <= 100:
                # Allocate across at most 2 rounds
                model += rounds_used <= 2, f"Medium_Demand_Split_{idx}"
            elif demand <= 250:
                # Allocate across at least 2 rounds
                model += rounds_used >= 2, f"Large_Demand_Split_{idx}"
        
        logger.info('Demand-splitting constraints added for data without existing_allocations')
    
    def add_average_price_constraints(self, model: pulp.LpProblem,
                                     variables: Dict[str, Dict[int, pulp.LpVariable]], 
                                     constraint_config: Dict[str, Any] = None) -> None:
        """Add average case-price constraints.

Constraint 2: each round average must lie between its Sheet2 price bounds.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables.
    constraint_config (Dict): Configuration overriding defaults."""
        for round_name in self.rounds:
            if round_name not in variables:
                continue
            
            # Retrieve constraints for this round
            try:
                constraints = self.data_loader.get_round_constraints(round_name)
                
                # Override defaults with configured parameters
                # Prefer round-level constraint parameters
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
                
                # Calculate total revenue in this round
                total_sales = pulp.lpSum([
                    variables[round_name][idx] * self.product_data.iloc[idx]['Wholesale_Price'] * 50000 / self.product_data.iloc[idx]['Carton_Stick_Ratio']
                    for idx in range(len(self.product_data))
                ])
                
                # Calculate total allocated quantity in this round
                total_allocation = pulp.lpSum([
                    variables[round_name][idx]
                    for idx in range(len(self.product_data))
                ])
                # Add mean price per case constraints
                model += total_sales <= upper_price_limit * total_allocation, f"Average_Price_Upper_{round_name}"
                model += total_sales >= lower_price_limit * total_allocation, f"Average_Price_Lower_{round_name}"
                
                logger.info(f"{round_name}Average case-price constraint: {lower_price_limit:.2f} <= average price <= {upper_price_limit:.2f}")
                
            except Exception as e:
                logger.warning(f"Cannot obtain constraints for {round_name}: {e}")
                continue
    
    def add_first_round_constraints(self, model: pulp.LpProblem,
                                    variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """Document the first-round supply bound.

Constraint 7 is implemented by decision-variable upper bounds; this method is documentation only.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables."""
        pass
    
    def add_balance_constraints(self, model: pulp.LpProblem, 
                                variables: Dict[str, Dict[int, pulp.LpVariable]]) -> None:
        """Add soft allocation-balance constraints.

Constraint 11 avoids overly concentrated or sparse rounds; adjacent-round differences above 20% are penalized as a soft constraint.

Args:
    model (pulp.LpProblem): Linear-programming model.
    variables (Dict): Decision variables."""
        # 1. Create auxiliary variables for total allocation in each round
        round_totals = {}
        for round_name in self.rounds:
            if round_name in variables:
                round_totals[round_name] = pulp.lpSum([variables[round_name][idx] for idx in range(len(self.product_data))])
        
        # 2. Create soft-constraint penalty variables
        self.balance_penalty_vars = {}
        for i in range(len(self.rounds)-1):
            round_name = self.rounds[i]
            next_round = self.rounds[i+1]
            if round_name in round_totals and next_round in round_totals:
                # Create positive and negative deviation variables
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
                
                # Soft constraint: allocation differences between adjacent rounds
                # round_totals[round_name] - 1.2 * round_totals[next_round] <= pos_deviation
                model += round_totals[round_name] - 1.2 * round_totals[next_round] <= pos_deviation, f"Balance_Upper_Soft_{round_name}_{next_round}"
                
                # round_totals[round_name] - 0.8 * round_totals[next_round] >= -neg_deviation
                model += round_totals[round_name] - 0.8 * round_totals[next_round] >= -neg_deviation, f"Balance_Lower_Soft_{round_name}_{next_round}"
        
        logger.info('Soft allocation-balance constraints added')
    
    def create_objective_function(self, variables: Dict[str, Dict[int, pulp.LpVariable]], objective_config: Dict[str, Any] = None) -> pulp.LpAffineExpression:
        """Create a weighted objective supporting total allocation, round-volume balance, round variance, product balance, and adjacent-round smoothness.

Args:
    variables (Dict): Decision variables.
    objective_config (Dict): Objective-weight configuration.

Returns:
    pulp.LpAffineExpression: Objective expression."""
        objective = 0
        
        # Use the default configuration if none is supplied
        if objective_config is None:
            objective_config = {
                "maximize_allocation_weight": 1000.0,      # Weight for maximising total allocation
                "round_balance_weight": 800.0,             # Weight for balancing total allocation across rounds
                "round_variance_weight": 400.0,            # Weight for minimising variance across rounds
                "product_balance_weight": 100.0,           # Weight for SKU-level balance
                "smooth_transition_weight": 300.0          # Weight for smooth transitions across rounds
            }
        
        logger.info(f"Objective configuration: {objective_config}")
        
        # 1. Maximise total allocation to satisfy as much demand as possible
        if objective_config.get("maximize_allocation_weight", 0) > 0:
            total_allocation = pulp.lpSum([
                variables[round_name][idx]
                for round_name in self.rounds
                for idx in range(len(self.product_data))
                if round_name in variables
            ])
            weight = objective_config["maximize_allocation_weight"]
            objective -= weight * total_allocation  # A negative sign represents maximisation
            logger.info(f"Total-allocation objective added; weight: {weight}")
        
        # Calculate round totals for subsequent balance calculations
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
            # 2. Balance total allocation across rounds: minimise the difference between maximum and minimum round totals
            if objective_config.get("round_balance_weight", 0) > 0:
                max_round = pulp.LpVariable("max_round", cat='Continuous')
                min_round = pulp.LpVariable("min_round", cat='Continuous')
                
                for round_total in round_totals:
                    self.model += max_round >= round_total
                    self.model += min_round <= round_total
                
                weight = objective_config["round_balance_weight"]
                objective += weight * (max_round - min_round)
                logger.info(f"Round-volume balance objective added; weight: {weight}")
            
            # 3. Minimise variance across rounds: penalise each round's deviation from the mean
            if objective_config.get("round_variance_weight", 0) > 0:
                avg_round = pulp.lpSum(round_totals) / len(round_totals)
                weight = objective_config["round_variance_weight"]
                
                for i, round_total in enumerate(round_totals):
                    deviation_pos = pulp.LpVariable(f"round_deviation_pos_{i}", lowBound=0)
                    deviation_neg = pulp.LpVariable(f"round_deviation_neg_{i}", lowBound=0)
                
                    self.model += round_total - avg_round == deviation_pos - deviation_neg
                    objective += weight * (deviation_pos + deviation_neg)
                
                logger.info(f"Round-variance objective added; weight: {weight}")

            # 4. SKU-level balance: encourage allocation across multiple rounds and avoid excessive concentration
            if objective_config.get("product_balance_weight", 0) > 0:
                weight = objective_config["product_balance_weight"]
                self._add_product_level_balance(objective, variables, weight)
                logger.info(f"Product-balance objective added; weight: {weight}")

            # 5. Smooth transitions across rounds: avoid excessive allocation differences between adjacent rounds
            if objective_config.get("smooth_transition_weight", 0) > 0:
                weight = objective_config["smooth_transition_weight"]
                self._add_smooth_transition(objective, variables, weight)
                logger.info(f"Smooth-transition objective added; weight: {weight}")

        # Add soft-constraint penalties for demand-based SKUs, increasing in later rounds
        if hasattr(self, 'demand_penalty_vars'):
            for idx, round_penalties in self.demand_penalty_vars.items():
                for i, (round_name, penalty_var) in enumerate(round_penalties.items(), start=3):
                    # Penalty coefficients are 50 for the third round, 100 for the fourth, 200 for the fifth, and so on
                    penalty_weight = 50 * (2 ** (i - 3))
                    objective += penalty_weight * penalty_var
        
        # Add soft-constraint penalties for allocation balance
        if hasattr(self, 'balance_penalty_vars'):
            for round_pair, deviations in self.balance_penalty_vars.items():
                # Balance constraint penalty coefficient
                balance_penalty_weight = 500
                objective += balance_penalty_weight * (deviations['pos'] + deviations['neg'])
        
        logger.info('Objective function created with configurable weights')
        return objective
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a solution summary.

Returns:
    Dict[str, Any]: Solution summary."""
        if self.solution is None:
            return {}
        
        allocation_matrix = self.solution['allocation_matrix']
        
        # Calculate basic statistics
        total_demand = allocation_matrix['Demand'].sum()
        total_allocated = allocation_matrix['Total_Allocation'].sum()
        allocation_rate = total_allocated / total_demand if total_demand > 0 else 0
        
        # Calculate allocated quantity in each round
        round_allocations = {}
        for round_name in self.rounds:
            if round_name in allocation_matrix.columns:
                round_allocations[round_name] = allocation_matrix[round_name].sum()
        
        # Calculate the number of allocated SKUs
        allocated_products = (allocation_matrix['Total_Allocation'] > 0).sum()
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
        """Add product-level balance, encouraging allocation across multiple rounds rather than concentration in a few rounds. Applies only to SKUs without existing_allocations.

Args:
    objective: Objective expression.
    variables: Decision variables.
    weight: Weight coefficient."""
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['Demand']
            if demand <= 0:
                continue
            
            # Check whether the SKU has an already determined allocation
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] > 0):
                    is_existing = True
                    break
            
            # Skip balance constraints for SKUs with already determined allocations
            if is_existing:
                continue
            
            # Retrieve the SKU's allocated quantity in each round
            product_allocations = []
            for round_name in self.rounds:
                if round_name in variables:
                    product_allocations.append(variables[round_name][idx])
            
            if len(product_allocations) >= 2:
                # Method 1: minimise the difference between the SKU's maximum and minimum allocated quantities
                product_max = pulp.LpVariable(f"product_max_{idx}", cat='Continuous')
                product_min = pulp.LpVariable(f"product_min_{idx}", cat='Continuous')
                
                for allocation in product_allocations:
                    self.model += product_max >= allocation
                    self.model += product_min <= allocation
                
                # Penalise imbalanced SKU allocation using the supplied weight
                objective += weight * (product_max - product_min)
                
                # Method 2: penalise allocations exceeding demand in a single round more strongly
                for round_name in self.rounds:
                    if round_name in variables:
                        # Penalise concentrated allocation above 60%
                        concentration_penalty_60 = pulp.LpVariable(f"concentration_penalty_60_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.6 * demand <= concentration_penalty_60
                        objective += weight * 3 * concentration_penalty_60  # Penalty for 60% concentration
                        
                        # Penalise concentrated allocation above 80% more strongly
                        concentration_penalty_80 = pulp.LpVariable(f"concentration_penalty_80_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.8 * demand <= concentration_penalty_80
                        objective += weight * 5 * concentration_penalty_80  # Penalty for 80% concentration
                        
                        # Penalise concentrated allocation above 90% very strongly
                        concentration_penalty_90 = pulp.LpVariable(f"concentration_penalty_90_{idx}_{round_name}", lowBound=0)
                        self.model += variables[round_name][idx] - 0.9 * demand <= concentration_penalty_90
                        objective += weight * 10 * concentration_penalty_90  # Penalty for 90% concentration
                
                # Method 3: encourage allocation across more rounds using binary variables
                if demand >= 50:  # Apply only to SKUs with relatively large demand
                    round_binary_vars = []
                    for round_name in self.rounds:
                        if round_name in variables:
                            binary_var = pulp.LpVariable(f"product_round_binary_{idx}_{round_name}", cat='Binary')
                            round_binary_vars.append(binary_var)
                            
                            # If allocation occurs in the round, the binary variable =1
                            M = 10000  # A sufficiently large number
                            self.model += variables[round_name][idx] <= M * binary_var
                            self.model += variables[round_name][idx] >= 0.1 * binary_var  # Minimum allocation threshold
                    
                    # Encourage allocation across more rounds
                    rounds_used = pulp.lpSum(round_binary_vars)
                    if demand >= 100:
                        # Encourage SKUs with large demand to be allocated across at least 2 rounds
                        rounds_shortage = pulp.LpVariable(f"rounds_shortage_{idx}", lowBound=0)
                        self.model += 2 - rounds_used <= rounds_shortage
                        objective += weight * 1.5 * rounds_shortage
        
        logger.info(f"Product-level balance mechanism added; weight: {weight}")
    
    def _add_smooth_transition(self, objective: pulp.LpAffineExpression, 
                              variables: Dict[str, Dict[int, pulp.LpVariable]], 
                              weight: float) -> None:
        """Add smooth transitions between rounds to discourage large adjacent-round allocation differences.

Args:
    objective: Objective expression.
    variables: Decision variables.
    weight: Weight coefficient."""
        round_names = list(self.rounds)
        
        # Calculate total allocation in each round
        round_totals = {}
        for round_name in round_names:
            if round_name in variables:
                round_totals[round_name] = pulp.lpSum([
                    variables[round_name][idx] for idx in range(len(self.product_data))
                    if self.product_data.iloc[idx]['Demand'] > 0
                ])
        
        # Add smoothing constraints between adjacent rounds
        for i in range(len(round_names) - 1):
            current_round = round_names[i]
            next_round = round_names[i + 1]
            
            if current_round in round_totals and next_round in round_totals:
                # Create difference variables
                diff_var = pulp.LpVariable(f"round_diff_{i}_{i+1}", cat='Continuous')
                
                # Absolute-difference constraints
                self.model += round_totals[next_round] - round_totals[current_round] <= diff_var
                self.model += round_totals[current_round] - round_totals[next_round] <= diff_var
                
                # Penalise excessive differences between adjacent rounds using the supplied weight
                objective += weight * diff_var
        
        # Add SKU-level smoothing constraints between adjacent rounds
        for idx in range(len(self.product_data)):
            demand = self.product_data.iloc[idx]['Demand']
            if demand <= 0:
                continue
            
            # Check whether the SKU has an already determined allocation
            is_existing = False
            for round_name in self.rounds:
                if (round_name in self.existing_allocations and 
                    idx in self.existing_allocations[round_name].index and
                    self.existing_allocations[round_name][idx] > 0):
                    is_existing = True
                    break
            
            # Skip smoothing constraints for SKUs with already determined allocations
            if is_existing:
                continue
            
            # Apply SKU-level smoothing constraints to SKUs with relatively large demand
            if demand >= 30:
                for i in range(len(round_names) - 1):
                    current_round = round_names[i]
                    next_round = round_names[i + 1]
                    
                    if current_round in variables and next_round in variables:
                        # SKU-level differences between adjacent rounds
                        product_diff = pulp.LpVariable(f"product_diff_{idx}_{i}_{i+1}", cat='Continuous')
                        
                        self.model += variables[next_round][idx] - variables[current_round][idx] <= product_diff
                        self.model += variables[current_round][idx] - variables[next_round][idx] <= product_diff
                        
                        # Penalise excessive SKU allocation differences between adjacent rounds using the supplied weight
                        objective += weight * 0.17 * product_diff  # A relatively small weight
        
        logger.info(f"Smooth-transition mechanism added; weight: {weight}")
    
    def solve(self, constraint_config: Dict[str, Any] = None, objective_config: Dict[str, Any] = None, time_limit: int = 300) -> Dict[str, Any]:
        """Solve the linear-programming problem.

Args:
    constraint_config (Dict[str, Any]): Constraint configuration.
    objective_config (Dict[str, Any]): Objective configuration.
    time_limit (int): Solver time limit in seconds.

Returns:
    Dict[str, Any]: Solver result."""
        logger.info('Starting linear-programming model construction')
        
        # Set the default configuration
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
                "maximize_allocation_weight": 1000.0,      # Weight for maximising total allocation
                "round_balance_weight": 800.0,             # Weight for balancing total allocation across rounds
                "round_variance_weight": 400.0,            # Weight for minimising variance across rounds
                "product_balance_weight": 100.0,           # Weight for SKU-level balance
                "smooth_transition_weight": 300.0          # Weight for smooth transitions across rounds
            }
        
        # Create the model
        self.model = pulp.LpProblem("Cigarette_Allocation", pulp.LpMinimize)
        
        # Create variables
        self.variables = self.create_variables()
        
        # Add constraints according to the configuration
        if constraint_config.get("enable_demand_constraints", True):
            self.add_demand_constraints(self.model, self.variables, constraint_config)  # Constraint 1: demand fulfilment 
        
        if constraint_config.get("enable_volume_constraints", True):
            self.add_volume_constraints(self.model, self.variables, constraint_config)  # Constraint 3: total allocation
        
        if constraint_config.get("enable_price_constraints", True):
            self.add_average_price_constraints(self.model, self.variables, constraint_config)  # Constraint 2: mean price per case
        
        # Add allocation strategy constraints according to the configuration
        if constraint_config.get("enable_demand_split_constraints", True):
            self.add_demand_split_constraints(self.model, self.variables)  # Constraint 6: demand concentration 50 00 150 (only outside existing_allocations)
        # Constraint 7: the first-round supply cap is already enforced through variable upper bounds
        if constraint_config.get("enable_demand_based_constraints", True):
            self.add_demand_based_constraints(self.model, self.variables)  # Constraint 8: demand-based priority (soft constraint)
        if constraint_config.get("enable_price_based_constraints", True):
            self.add_price_based_constraints(self.model, self.variables, constraint_config)  # Constraint 9: price-based proportion
        
        # Add C-category cigarette constraints according to the configuration
        if constraint_config.get("enable_c_type_constraints", True):
            self.add_c_type_constraints(self.model, self.variables, constraint_config)  # Constraints 10.1-10.2
            self._add_c_subtype_constraints(self.model, self.variables, constraint_config)  # Constraints 10.3-10.5
        
        # Add balance constraints according to the configuration
        if constraint_config.get("enable_balance_constraints", True):
            self.add_balance_constraints(self.model, self.variables)  # Constraint 11: allocation balance (soft constraint)
        
        # Set the objective function
        objective = self.create_objective_function(self.variables, objective_config)
        self.model += objective
        
        logger.info(f"Model constructed; starting solver (time limit: {time_limit} seconds)")
        
        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
        self.model.solve(solver)
        
        # Process the solution
        status = pulp.LpStatus[self.model.status]
        logger.info(f"Solver complete; status: {status}")
        
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
            logger.error('Model is infeasible')
            return {
                'status': 'Infeasible',
                'message': 'Conflicting constraints; no feasible solution',
                'allocation_matrix': None
            }
        
        else:
            logger.warning(f"Solver did not complete; status: {status}")
            return {
                'status': status,
                'message': f'Solver status: {status}',
                'allocation_matrix': None
            }
    
    def _extract_solution(self) -> Dict[str, Any]:
        """Extract the solution.

Returns:
    Dict[str, Any]: Solution including the full allocation matrix and original product fields."""
        # Create a DataFrame containing the original data
        allocation_matrix = self.data_loader.get_product_data().copy()
        
        # Update allocation results
        for round_name in self.rounds:
            if round_name in self.variables:
                allocation_matrix[round_name] = 0.0  # Initialise round columns
                for idx in range(len(self.product_data)):
                    var_value = self.variables[round_name][idx].varValue
                    if var_value is not None:
                        allocation_matrix.loc[idx, round_name] = float(var_value)
        
        # Handle tiny allocations below 0.1; many values of 0.01 arise from minimisation and can be set to 0
        allocation_matrix = self._handle_small_allocations(allocation_matrix)
        
        # Postprocess unallocated SKUs whose shortfall from demand is very small
        allocation_matrix = self._handle_tiny_unallocated_demand(allocation_matrix)
        
        # Calculate total allocation
        allocation_cols = [col for col in self.rounds if col in allocation_matrix.columns]
        if allocation_cols:
            allocation_matrix['Total_Allocation'] = allocation_matrix[allocation_cols].sum(axis=1)
        else:
            allocation_matrix['Total_Allocation'] = 0
        
        # Calculate the allocation rate
        allocation_matrix['Allocation_Rate'] = np.where(
            allocation_matrix['Demand'] > 0,
            allocation_matrix['Total_Allocation'] / allocation_matrix['Demand'],
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
        """Postprocess small allocations by merging quantities below the threshold into another round of the same SKU or setting them to zero.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.
    min_threshold (float): Minimum allocation threshold.

Returns:
    pd.DataFrame: Processed allocation matrix."""
        processed_matrix = allocation_matrix.copy()
        
        for idx in range(len(processed_matrix)):
            # Retrieve the SKU's allocated quantity in each round
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
            
            # Find allocations below the threshold
            small_allocations = {
                k: v for k, v in round_allocations.items() if (k not in fixed_rounds) and 0 < v < min_threshold
            }
            large_allocations = {
                k: v for k, v in round_allocations.items() if (k not in fixed_rounds) and v >= min_threshold
            }
            
            if small_allocations and large_allocations:
                # Merge small allocations into the round with the largest allocation
                max_round = max(large_allocations.keys(), key=lambda k: large_allocations[k])
                total_small = sum(small_allocations.values())
                
                # Merge allocations
                processed_matrix.loc[idx, max_round] += total_small
                
                # Set small allocations to zero
                for round_name in small_allocations.keys():
                    processed_matrix.loc[idx, round_name] = 0
                    
            elif small_allocations and not large_allocations:
                # If only small allocations exist, merge them into the first round with allocation
                if small_allocations:
                    first_round = list(small_allocations.keys())[0]
                    total_small = sum(small_allocations.values())
                    
                    processed_matrix.loc[idx, first_round] = total_small
                    for round_name in list(small_allocations.keys())[1:]:
                        processed_matrix.loc[idx, round_name] = 0
        
        logger.info(f"Small-allocation processing complete; threshold: {min_threshold}")
        return processed_matrix
    
    def _handle_tiny_unallocated_demand(self, allocation_matrix: pd.DataFrame, tiny_threshold: float = 0.01) -> pd.DataFrame:
        """Handle very small demand shortfalls.

For a small unallocated remainder, such as 0.01, add the difference to any already allocated round.

Args:
    allocation_matrix (pd.DataFrame): Allocation matrix.
    tiny_threshold (float): Small-difference threshold, default 0.01.

Returns:
    pd.DataFrame: Processed allocation matrix."""
        processed_matrix = allocation_matrix.copy()
        
        for idx in range(len(processed_matrix)):
            demand = processed_matrix.loc[idx, 'Demand']
            if demand <= 0:
                continue
            
            # Calculate the current total allocation
            round_allocations = {}
            total_allocated = 0
            for round_name in self.rounds:
                if round_name in processed_matrix.columns:
                    allocation = processed_matrix.loc[idx, round_name]
                    round_allocations[round_name] = allocation
                    total_allocated += allocation
            
            # Calculate the unallocated quantity
            unallocated = demand - total_allocated
            
            # If the unallocated quantity is small and positive, assign it to a round with existing allocation
            if unallocated <= tiny_threshold:
                # Find rounds with existing allocation
                allocated_rounds = {k: v for k, v in round_allocations.items() if v > 0}
                
                if allocated_rounds:
                    # Assign it to the round with the largest allocation
                    max_round = max(allocated_rounds.keys(), key=lambda k: allocated_rounds[k])
                    processed_matrix.loc[idx, max_round] += unallocated
                    
                    logger.debug(f"SKU {idx} small unallocated remainder {unallocated:.3f} assigned to {max_round}")
                else:
                    # If no round has allocation, assign it to the first round
                    first_round = self.rounds[0] if self.rounds else None
                    if first_round and first_round in processed_matrix.columns:
                        processed_matrix.loc[idx, first_round] = unallocated
                        logger.debug(f"SKU {idx} small unallocated remainder {unallocated:.3f} assigned to {first_round}")
        
        logger.info(f"Small-demand-shortfall processing complete; threshold: {tiny_threshold}")
        return processed_matrix
    
    def post_process_solution(self, allocation_matrix: pd.DataFrame) -> pd.DataFrame:
        """Postprocess constraints that are difficult to handle in the linear program, such as concentrated square-product allocation. Currently unused.

Args:
    allocation_matrix (pd.DataFrame): Original allocation matrix.

Returns:
    pd.DataFrame: Postprocessed matrix."""
        processed_matrix = allocation_matrix.copy()
        
        # Handle concentrated allocation constraints for C-category cigarettes
        type_column = self.product_data['Class'].astype(str)
        c_type_mask = type_column.str.contains('C', na=False)
        c_type_indices = c_type_mask[c_type_mask].index.tolist()
        
        for idx in c_type_indices:
            # Find the round with the largest allocation
            round_allocations = processed_matrix.loc[idx, self.rounds]
            if round_allocations.sum() > 0:
                max_round = round_allocations.idxmax()
                total_allocation = round_allocations.sum()
                
                # Concentrate all allocation in the largest round
                for round_name in self.rounds:
                    if round_name == max_round:
                        processed_matrix.loc[idx, round_name] = total_allocation
                    else:
                        processed_matrix.loc[idx, round_name] = 0
        
        # Integerisation (retain three decimal places)
        processed_matrix = processed_matrix.round(3)
        
        logger.info('Solution postprocessing complete')
        return processed_matrix


