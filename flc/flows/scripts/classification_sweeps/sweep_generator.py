import hashlib
import itertools
from typing import Dict, List, Any, Set
from pydantic.dataclasses import dataclass

from .config import AlgorithmSweepConfig
from flc.shared.classification import ClassificationFactory


@dataclass(frozen=True)
class SweepCombination:
    """Represents a single parameter combination for a sweep"""
    algorithm: str
    hyperparameters: Dict[str, Any]
    sweep_id: str
    parameter_hash: str  # Hash for exact parameter matching
    
    @classmethod
    def create(cls, algorithm: str, hyperparameters: Dict[str, Any]) -> "SweepCombination":
        """Create a sweep combination with generated IDs"""
        # Get ALL hyperparameters (swept + defaults) for complete hash
        all_hyperparameters = get_all_hyperparameters_for_sweep(algorithm, hyperparameters)
        
        sweep_id = generate_sweep_id(algorithm, hyperparameters)
        parameter_hash = generate_parameter_hash(all_hyperparameters)
        return cls(
            algorithm=algorithm,
            hyperparameters=hyperparameters,  # Store only swept params in the combination
            sweep_id=sweep_id,
            parameter_hash=parameter_hash  # But hash includes all params
        )


def get_all_hyperparameters_for_sweep(algorithm_name: str, swept_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ALL hyperparameters by merging swept parameters with config defaults.
    
    Args:
        algorithm_name: Name of the algorithm
        swept_hyperparameters: Hyperparameters that were swept (potentially incomplete)
        
    Returns:
        Dictionary with all hyperparameters (swept + defaults)
    """
    try:
        # Get the config class for this algorithm
        config_class = ClassificationFactory.get_model_config_class(algorithm_name)
        
        # Create a config instance with swept parameters to get all defaults
        config = config_class(**swept_hyperparameters)
        
        # Convert config to dictionary to get all parameters
        all_hyperparameters = {}
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            # Convert to JSON-serializable types
            if value is not None:
                all_hyperparameters[field_name] = value
                
        return all_hyperparameters
        
    except Exception:
        # Fallback to swept parameters only if we can't get defaults
        return swept_hyperparameters


def generate_parameter_hash(hyperparameters: Dict[str, Any]) -> str:
    """Generate a stable hash for exact parameter matching"""
    # Sort parameters to ensure consistent hashing
    sorted_params = sorted(hyperparameters.items())
    param_str = "_".join(f"{k}={v}" for k, v in sorted_params)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def generate_sweep_id(algorithm: str, hyperparameters: Dict[str, Any]) -> str:
    """Generate a human-readable sweep ID"""
    param_hash = generate_parameter_hash(hyperparameters)
    return f"{algorithm}_{param_hash}"


def generate_sweep_combinations(
    enabled_algorithms: Dict[str, AlgorithmSweepConfig]
) -> List[SweepCombination]:
    """
    Generate all combinations of algorithm hyperparameters.
    
    Args:
        enabled_algorithms: Dictionary of enabled algorithm configurations
        
    Returns:
        List of all possible sweep combinations
    """
    all_combinations = []
    
    for algorithm_name, algorithm_config in enabled_algorithms.items():
        if not algorithm_config.hyperparameters:
            # Create a single combination with no hyperparameters
            combination = SweepCombination.create(algorithm_name, {})
            all_combinations.append(combination)
            continue
            
        # Get all parameter names and their possible values
        param_names = list(algorithm_config.hyperparameters.keys())
        param_values = [algorithm_config.hyperparameters[name] for name in param_names]
        
        # Generate cartesian product of all parameter combinations
        for value_combination in itertools.product(*param_values):
            hyperparameters = dict(zip(param_names, value_combination))
            combination = SweepCombination.create(algorithm_name, hyperparameters)
            all_combinations.append(combination)
    
    return all_combinations


def filter_completed_sweeps(
    combinations: List[SweepCombination], 
    completed_sweep_ids: Set[str]
) -> List[SweepCombination]:
    """
    Filter out combinations that have already been completed.
    
    Args:
        combinations: List of all sweep combinations
        completed_sweep_ids: Set of completed sweep IDs
        
    Returns:
        List of combinations that still need to be executed
    """
    return [
        combination for combination in combinations 
        if combination.sweep_id not in completed_sweep_ids
    ]


def filter_completed_sweeps_exact_match(
    combinations: List[SweepCombination],
    completed_combinations: Set[tuple]  # (algorithm, parameter_hash) tuples
) -> List[SweepCombination]:
    """
    Filter combinations using exact parameter matching.
    
    This is more robust than sweep_id matching when parameters are added/changed.
    
    Args:
        combinations: List of all sweep combinations  
        completed_combinations: Set of (algorithm, parameter_hash) tuples that were completed
        
    Returns:
        List of combinations that still need to be executed
    """
    return [
        combination for combination in combinations
        if (combination.algorithm, combination.parameter_hash) not in completed_combinations
    ]


def filter_completed_sweeps_with_datasets(
    combinations: List[SweepCombination],
    completed_combinations: Set[tuple],  # (train_path, test_path, algorithm, parameter_hash) tuples
    train_dataset_path: str,
    test_dataset_path: str
) -> List[SweepCombination]:
    """
    Filter combinations using exact parameter matching with dataset pair.
    
    Args:
        combinations: List of all sweep combinations  
        completed_combinations: Set of (train_path, test_path, algorithm, parameter_hash) tuples
        train_dataset_path: Train dataset path for this execution
        test_dataset_path: Test dataset path for this execution
        
    Returns:
        List of combinations that still need to be executed for this dataset pair
    """
    return [
        combination for combination in combinations
        if (train_dataset_path, test_dataset_path, combination.algorithm, combination.parameter_hash) not in completed_combinations
    ]


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
            "total_combinations": 0,
            "algorithms": [],
            "total_by_algorithm": {},
            "unique_algorithms": 0
        }
    
    algorithms = [combo.algorithm for combo in combinations]
    unique_algorithms = list(set(algorithms))
    
    total_by_algorithm = {}
    for algorithm in unique_algorithms:
        total_by_algorithm[algorithm] = sum(1 for combo in combinations if combo.algorithm == algorithm)
    
    return {
        "total_combinations": len(combinations),
        "algorithms": unique_algorithms,
        "total_by_algorithm": total_by_algorithm,
        "unique_algorithms": len(unique_algorithms)
    }


def generate_hyperparameter_summary(combinations: List[SweepCombination]) -> Dict[str, Dict[str, Set[Any]]]:
    """
    Generate a summary of all hyperparameter values being tested.
    
    Args:
        combinations: List of sweep combinations
        
    Returns:
        Dictionary mapping algorithm names to their hyperparameter value sets
    """
    summary = {}
    
    for combination in combinations:
        algorithm = combination.algorithm
        if algorithm not in summary:
            summary[algorithm] = {}
            
        for param_name, param_value in combination.hyperparameters.items():
            if param_name not in summary[algorithm]:
                summary[algorithm][param_name] = set()
            summary[algorithm][param_name].add(param_value)
    
    return summary


def validate_sweep_combinations(combinations: List[SweepCombination]) -> None:
    """
    Validate that sweep combinations are well-formed.
    
    Args:
        combinations: List of sweep combinations to validate
        
    Raises:
        ValueError: If any combination is invalid
    """
    if not combinations:
        raise ValueError("No sweep combinations generated")
    
    for i, combination in enumerate(combinations):
        if not combination.algorithm:
            raise ValueError(f"Combination {i} has empty algorithm name")
        
        if not combination.sweep_id:
            raise ValueError(f"Combination {i} has empty sweep_id")
        
        if not combination.parameter_hash:
            raise ValueError(f"Combination {i} has empty parameter_hash")
        
        # Check for duplicate sweep IDs
        sweep_ids = [combo.sweep_id for combo in combinations]
        if len(set(sweep_ids)) != len(sweep_ids):
            duplicates = [sweep_id for sweep_id in set(sweep_ids) if sweep_ids.count(sweep_id) > 1]
            raise ValueError(f"Duplicate sweep IDs found: {duplicates}")