from typing import List, Dict, Any, Optional, Iterator
from itertools import product

from pydantic.dataclasses import dataclass

from .config import ClusteringALSweepConfig


@dataclass
class ClusteringALCombination:
    """Configuration for a single clustering active learning combination"""
    
    learner_type: str
    query_strategy: str
    query_strategy_config: Dict[str, Any]
    weighting_strategy: str
    weighting_strategy_config: Dict[str, Any]
    clustering_algorithm: str
    clustering_config: Dict[str, Any]
    label_selection_strategy: str
    label_selection_config: Dict[str, Any]
    classifier_name: str
    classifier_config: Dict[str, Any]
    max_iterations: int
    evaluation_interval: int
    test_evaluation_interval: int
    evaluate_on_test: bool
    random_state: int
    combination_id: str

    max_total_samples: Optional[int] = None
    
    def __post_init__(self):
        if self.query_strategy_config is None:
            self.query_strategy_config = {}
        if self.weighting_strategy_config is None:
            self.weighting_strategy_config = {}
        if self.clustering_config is None:
            self.clustering_config = {}
        if self.label_selection_config is None:
            self.label_selection_config = {}
        if self.classifier_config is None:
            self.classifier_config = {}


def generate_clustering_combinations(
    clustering_al_config: ClusteringALSweepConfig,
    random_seeds: List[int],
) -> List[ClusteringALCombination]:
    """
    Generate all clustering AL parameter combinations with random seeds.
    
    Args:
        clustering_al_config: Clustering AL sweep configuration
        random_seeds: List of random seeds for reproducibility
        
    Returns:
        List of ClusteringALCombination objects with unique combination IDs
    """
    combinations = []
    combination_counter = 1
    
    # Generate base combinations (without seeds)
    base_combinations = list(_generate_base_combinations(clustering_al_config))
    
    # Add random seeds to each base combination
    for base_combo in base_combinations:
        for seed in random_seeds:
            combination_id = create_combination_id(combination_counter, seed)
            
            # Set random_state in sub-configurations that require it
            query_strategy_config = base_combo["query_strategy_config"].copy()
            query_strategy_config["random_state"] = seed
            
            clustering_config = base_combo["clustering_config"].copy()
            clustering_config["random_state"] = seed
            
            label_selection_config = base_combo["label_selection_config"].copy()
            label_selection_config["random_state"] = seed
            
            classifier_config = base_combo["classifier_config"].copy()
            classifier_config["random_state"] = seed
            
            combo = ClusteringALCombination(
                learner_type="clustering",
                query_strategy=base_combo["query_strategy"],
                query_strategy_config=query_strategy_config,
                weighting_strategy=base_combo["weighting_strategy"],
                weighting_strategy_config=base_combo["weighting_strategy_config"].copy(),
                clustering_algorithm=base_combo["clustering_algorithm"],
                clustering_config=clustering_config,
                label_selection_strategy=base_combo["label_selection_strategy"],
                label_selection_config=label_selection_config,
                classifier_name=base_combo["classifier_name"],
                classifier_config=classifier_config,
                max_iterations=base_combo["max_iterations"],
                max_total_samples=base_combo["max_total_samples"],
                evaluation_interval=base_combo["evaluation_interval"],
                test_evaluation_interval=base_combo["test_evaluation_interval"],
                evaluate_on_test=base_combo["evaluate_on_test"],
                random_state=seed,
                combination_id=combination_id,
            )
            
            combinations.append(combo)
        
        combination_counter += 1
    
    return combinations


def _generate_base_combinations(
    clustering_al_config: ClusteringALSweepConfig,
) -> Iterator[Dict[str, Any]]:
    """
    Generate base parameter combinations (without random seeds).
    
    Args:
        clustering_al_config: Clustering AL sweep configuration
        
    Yields:
        Dictionary containing parameter combination
    """
    # Expand query strategy configurations
    query_strategy_combinations = []
    for strategy_name, strategy_params in clustering_al_config.query_strategies.items():
        if not strategy_params:
            # Empty config - single combination
            query_strategy_combinations.append((strategy_name, {}))
        else:
            # Generate all parameter combinations for this strategy
            param_names = list(strategy_params.keys())
            param_values = list(strategy_params.values())
            
            for param_combination in product(*param_values):
                strategy_config = dict(zip(param_names, param_combination))
                query_strategy_combinations.append((strategy_name, strategy_config))

    # Expand weighting strategy configurations
    weighting_strategy_combinations = []
    for strategy_name, strategy_params in clustering_al_config.weighting_strategies.items():
        if not strategy_params:
            # Empty config - single combination
            weighting_strategy_combinations.append((strategy_name, {}))
        else:
            # Generate all parameter combinations for this strategy
            param_names = list(strategy_params.keys())
            param_values = list(strategy_params.values())
            
            for param_combination in product(*param_values):
                strategy_config = dict(zip(param_names, param_combination))
                weighting_strategy_combinations.append((strategy_name, strategy_config))

    # Expand clustering algorithm configurations
    clustering_algorithm_combinations = []
    for algorithm_name, algorithm_params in clustering_al_config.clustering_algorithms.items():
        if not algorithm_params:
            # Empty config - single combination
            clustering_algorithm_combinations.append((algorithm_name, {}))
        else:
            # Generate all parameter combinations for this algorithm
            param_names = list(algorithm_params.keys())
            param_values = list(algorithm_params.values())
            
            for param_combination in product(*param_values):
                algorithm_config = dict(zip(param_names, param_combination))
                clustering_algorithm_combinations.append((algorithm_name, algorithm_config))

    # Expand label selection strategy configurations
    label_selection_combinations = []
    for strategy_name, strategy_params in clustering_al_config.label_selection_strategies.items():
        if not strategy_params:
            # Empty config - single combination
            label_selection_combinations.append((strategy_name, {}))
        else:
            # Generate all parameter combinations for this strategy
            param_names = list(strategy_params.keys())
            param_values = list(strategy_params.values())
            
            for param_combination in product(*param_values):
                strategy_config = dict(zip(param_names, param_combination))
                label_selection_combinations.append((strategy_name, strategy_config))
    
    # Expand classifier configurations
    classifier_combinations = []
    for classifier_name, classifier_params in clustering_al_config.classifier_configs.items():
        if not classifier_params:
            # Empty config - single combination
            classifier_combinations.append((classifier_name, {}))
        else:
            # Generate all parameter combinations for this classifier
            param_names = list(classifier_params.keys())
            param_values = list(classifier_params.values())
            
            for param_combination in product(*param_values):
                classifier_config = dict(zip(param_names, param_combination))
                classifier_combinations.append((classifier_name, classifier_config))
    
    # Generate all combinations
    for (query_strategy, query_config) in query_strategy_combinations:
        for (weighting_strategy, weighting_config) in weighting_strategy_combinations:
            for (clustering_algorithm, clustering_config) in clustering_algorithm_combinations:
                for (label_selection_strategy, label_selection_config) in label_selection_combinations:
                    for (classifier_name, classifier_config) in classifier_combinations:
                        for max_iters in clustering_al_config.max_iterations:
                            for max_total in clustering_al_config.max_total_samples:
                                for eval_interval in clustering_al_config.evaluation_interval:
                                    for test_eval_interval in clustering_al_config.test_evaluation_interval:
                                        for eval_on_test in clustering_al_config.evaluate_on_test:
                                            yield {
                                                "query_strategy": query_strategy,
                                                "query_strategy_config": query_config,
                                                "weighting_strategy": weighting_strategy,
                                                "weighting_strategy_config": weighting_config,
                                                "clustering_algorithm": clustering_algorithm,
                                                "clustering_config": clustering_config,
                                                "label_selection_strategy": label_selection_strategy,
                                                "label_selection_config": label_selection_config,
                                                "classifier_name": classifier_name,
                                                "classifier_config": classifier_config,
                                                "max_iterations": max_iters,
                                                "max_total_samples": max_total,
                                                "evaluation_interval": eval_interval,
                                                "test_evaluation_interval": test_eval_interval,
                                                "evaluate_on_test": eval_on_test,
                                            }


def create_combination_id(combination_number: int, random_state: int) -> str:
    """
    Generate unique identifier for parameter combination.
    
    Args:
        combination_number: Sequential combination number
        random_state: Random seed for this combination
        
    Returns:
        Unique combination ID string
    """
    return f"combination_{combination_number:04d}_seed_{random_state}"


def get_sweep_statistics(
    clustering_al_config: ClusteringALSweepConfig,
    random_seeds: List[int],
) -> Dict[str, Any]:
    """
    Get summary statistics for the parameter sweep.
    
    Args:
        clustering_al_config: Clustering AL sweep configuration
        random_seeds: List of random seeds
        
    Returns:
        Dictionary with sweep statistics
    """
    # Count base combinations
    base_combinations = list(_generate_base_combinations(clustering_al_config))
    base_count = len(base_combinations)
    
    # Total combinations including seeds
    total_combinations = base_count * len(random_seeds)
    
    # Count strategies, algorithms, and classifiers
    num_query_strategies = len(clustering_al_config.query_strategies)
    num_weighting_strategies = len(clustering_al_config.weighting_strategies)
    num_clustering_algorithms = len(clustering_al_config.clustering_algorithms)
    num_label_selection_strategies = len(clustering_al_config.label_selection_strategies)
    num_classifiers = len(clustering_al_config.classifier_configs)
    
    # Count parameter values
    num_max_iterations = len(clustering_al_config.max_iterations)
    num_max_total_samples = len(clustering_al_config.max_total_samples)
    num_evaluation_intervals = len(clustering_al_config.evaluation_interval)
    num_test_evaluation_intervals = len(clustering_al_config.test_evaluation_interval)
    num_evaluate_on_test = len(clustering_al_config.evaluate_on_test)
    num_seeds = len(random_seeds)
    
    # Calculate query strategy parameter counts
    query_strategy_param_counts = {}
    for strategy_name, strategy_params in clustering_al_config.query_strategies.items():
        if not strategy_params:
            count = 1
        else:
            count = 1
            for param_values in strategy_params.values():
                count *= len(param_values)
        query_strategy_param_counts[strategy_name] = count

    # Calculate weighting strategy parameter counts
    weighting_strategy_param_counts = {}
    for strategy_name, strategy_params in clustering_al_config.weighting_strategies.items():
        if not strategy_params:
            count = 1
        else:
            count = 1
            for param_values in strategy_params.values():
                count *= len(param_values)
        weighting_strategy_param_counts[strategy_name] = count

    # Calculate clustering algorithm parameter counts
    clustering_algorithm_param_counts = {}
    for algorithm_name, algorithm_params in clustering_al_config.clustering_algorithms.items():
        if not algorithm_params:
            count = 1
        else:
            count = 1
            for param_values in algorithm_params.values():
                count *= len(param_values)
        clustering_algorithm_param_counts[algorithm_name] = count

    # Calculate label selection strategy parameter counts
    label_selection_param_counts = {}
    for strategy_name, strategy_params in clustering_al_config.label_selection_strategies.items():
        if not strategy_params:
            count = 1
        else:
            count = 1
            for param_values in strategy_params.values():
                count *= len(param_values)
        label_selection_param_counts[strategy_name] = count
    
    # Calculate classifier parameter counts
    classifier_param_counts = {}
    for classifier_name, classifier_params in clustering_al_config.classifier_configs.items():
        if not classifier_params:
            count = 1
        else:
            count = 1
            for param_values in classifier_params.values():
                count *= len(param_values)
        classifier_param_counts[classifier_name] = count
    
    return {
        "total_combinations": total_combinations,
        "base_combinations": base_count,
        "random_seeds": num_seeds,
        "query_strategies": {
            "count": num_query_strategies,
            "names": list(clustering_al_config.query_strategies.keys()),
            "parameter_counts": query_strategy_param_counts,
        },
        "weighting_strategies": {
            "count": num_weighting_strategies,
            "names": list(clustering_al_config.weighting_strategies.keys()),
            "parameter_counts": weighting_strategy_param_counts,
        },
        "clustering_algorithms": {
            "count": num_clustering_algorithms,
            "names": list(clustering_al_config.clustering_algorithms.keys()),
            "parameter_counts": clustering_algorithm_param_counts,
        },
        "label_selection_strategies": {
            "count": num_label_selection_strategies,
            "names": list(clustering_al_config.label_selection_strategies.keys()),
            "parameter_counts": label_selection_param_counts,
        },
        "classifiers": {
            "count": num_classifiers,
            "names": list(clustering_al_config.classifier_configs.keys()),
            "parameter_counts": classifier_param_counts,
        },
        "al_parameters": {
            "max_iterations": num_max_iterations,
            "max_total_samples": num_max_total_samples,
        },
        "evaluation_parameters": {
            "evaluation_interval": num_evaluation_intervals,
            "test_evaluation_interval": num_test_evaluation_intervals,
            "evaluate_on_test": num_evaluate_on_test,
        },
        "estimated_experiments": total_combinations,
    }