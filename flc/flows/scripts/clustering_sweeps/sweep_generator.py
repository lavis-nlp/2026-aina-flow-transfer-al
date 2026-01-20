import itertools
from typing import Dict, List, Any, NamedTuple
from dataclasses import dataclass

from .config import AlgorithmSweepConfig


@dataclass(frozen=True)
class SweepCombination:
    """Represents a single parameter combination for an algorithm"""
    algorithm: str
    hyperparameters: Dict[str, Any]
    sweep_id: str

    def __post_init__(self):
        # Ensure hyperparameters is immutable
        if not isinstance(self.hyperparameters, dict):
            raise ValueError("Hyperparameters must be a dictionary")


def generate_sweep_combinations(algorithms: Dict[str, AlgorithmSweepConfig]) -> List[SweepCombination]:
    """
    Generate all possible parameter combinations for enabled algorithms.
    
    Args:
        algorithms: Dictionary of algorithm configurations
    
    Returns:
        List of SweepCombination objects representing all parameter combinations
    """
    all_combinations = []
    
    for algorithm_name, config in algorithms.items():
        if not config.enabled:
            continue
        
        # Generate combinations for this algorithm
        algo_combinations = _generate_algorithm_combinations(algorithm_name, config)
        all_combinations.extend(algo_combinations)
    
    return all_combinations


def _generate_algorithm_combinations(algorithm_name: str, config: AlgorithmSweepConfig) -> List[SweepCombination]:
    """
    Generate all parameter combinations for a single algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        config: Algorithm sweep configuration
    
    Returns:
        List of SweepCombination objects for this algorithm
    """
    hyperparameters = config.hyperparameters
    
    if not hyperparameters:
        # No hyperparameters specified, create default combination
        sweep_id = _create_sweep_id(algorithm_name, {})
        return [SweepCombination(
            algorithm=algorithm_name,
            hyperparameters={},
            sweep_id=sweep_id
        )]
    
    # Get parameter names and their possible values
    param_names = list(hyperparameters.keys())
    param_values = [hyperparameters[name] for name in param_names]
    
    # Validate parameter values are lists
    for name, values in zip(param_names, param_values):
        if not isinstance(values, list):
            raise ValueError(f"Parameter '{name}' for algorithm '{algorithm_name}' must be a list of values")
        if not values:
            raise ValueError(f"Parameter '{name}' for algorithm '{algorithm_name}' cannot have empty values list")
    
    # Generate all combinations using cartesian product
    combinations = []
    for value_combination in itertools.product(*param_values):
        # Create parameter dictionary for this combination
        param_dict = dict(zip(param_names, value_combination))
        
        # Create sweep ID
        sweep_id = _create_sweep_id(algorithm_name, param_dict)
        
        combinations.append(SweepCombination(
            algorithm=algorithm_name,
            hyperparameters=param_dict,
            sweep_id=sweep_id
        ))
    
    return combinations


def _create_sweep_id(algorithm_name: str, hyperparameters: Dict[str, Any]) -> str:
    """
    Create a unique identifier for a parameter combination.
    
    Args:
        algorithm_name: Name of the algorithm
        hyperparameters: Parameter dictionary
    
    Returns:
        Unique sweep ID string
    """
    # Sort parameters by name for consistent ordering
    sorted_params = sorted(hyperparameters.items())
    
    # Create parameter string
    param_strs = []
    for param_name, param_value in sorted_params:
        param_strs.append(f"{param_name}={param_value}")
    
    if param_strs:
        param_string = "_".join(param_strs)
        return f"{algorithm_name}_{param_string}"
    else:
        return f"{algorithm_name}_default"


def filter_completed_sweeps(
    all_combinations: List[SweepCombination], 
    completed_sweep_ids: set
) -> List[SweepCombination]:
    """
    Filter out already completed sweep combinations.
    
    Args:
        all_combinations: All possible sweep combinations
        completed_sweep_ids: Set of sweep IDs that have already been completed
    
    Returns:
        List of sweep combinations that still need to be executed
    """
    remaining_combinations = []
    
    for combination in all_combinations:
        if combination.sweep_id not in completed_sweep_ids:
            remaining_combinations.append(combination)
    
    return remaining_combinations


def get_sweep_statistics(combinations: List[SweepCombination]) -> Dict[str, Any]:
    """
    Get statistics about sweep combinations.
    
    Args:
        combinations: List of sweep combinations
    
    Returns:
        Dictionary with sweep statistics
    """
    if not combinations:
        return {
            'total_combinations': 0,
            'algorithms': {},
            'total_by_algorithm': {}
        }
    
    # Count by algorithm
    algorithm_counts = {}
    algorithm_details = {}
    
    for combo in combinations:
        algo = combo.algorithm
        
        if algo not in algorithm_counts:
            algorithm_counts[algo] = 0
            algorithm_details[algo] = []
        
        algorithm_counts[algo] += 1
        algorithm_details[algo].append(combo.hyperparameters)
    
    return {
        'total_combinations': len(combinations),
        'algorithms': list(algorithm_counts.keys()),
        'total_by_algorithm': algorithm_counts,
        'algorithm_details': algorithm_details
    }