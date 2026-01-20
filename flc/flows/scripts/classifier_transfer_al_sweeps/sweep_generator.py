from typing import List, Dict, Any, Optional, Iterator
from itertools import product

from pydantic.dataclasses import dataclass

from .config import ClassifierALSweepConfig


@dataclass
class ClassifierALCombination:
    """Configuration for a single classifier active learning combination"""

    learner_type: str
    query_strategy: str
    query_strategy_config: Dict[str, Any]
    weighting_strategy: str
    weighting_strategy_config: Dict[str, Any]
    classifier_name: str
    classifier_config: Dict[str, Any]
    samples_per_iteration: int
    max_iterations: int
    evaluation_interval: int
    test_evaluation_interval: int
    evaluate_on_test: bool
    random_state: int
    combination_id: str

    max_total_samples: Optional[int] = None


def generate_classifier_combinations(
    classifier_al_config: ClassifierALSweepConfig,
    random_seeds: List[int],
) -> List[ClassifierALCombination]:
    """
    Generate all classifier AL parameter combinations with random seeds.

    Args:
        classifier_al_config: Classifier AL sweep configuration
        random_seeds: List of random seeds for reproducibility

    Returns:
        List of ClassifierALCombination objects with unique combination IDs
    """
    combinations = []
    combination_counter = 1

    # Generate base combinations (without seeds)
    base_combinations = list(_generate_base_combinations(classifier_al_config))

    # Add random seeds to each base combination
    for base_combo in base_combinations:
        for seed in random_seeds:
            combination_id = create_combination_id(combination_counter, seed)

            # Set random_state in sub-configurations that require it
            query_strategy_config = base_combo["query_strategy_config"].copy()
            query_strategy_config["random_state"] = seed

            classifier_config = base_combo["classifier_config"].copy()
            classifier_config["random_state"] = seed

            combo = ClassifierALCombination(
                learner_type="classifier",
                query_strategy=base_combo["query_strategy"],
                query_strategy_config=query_strategy_config,
                weighting_strategy=base_combo["weighting_strategy"],
                weighting_strategy_config=base_combo["weighting_strategy_config"].copy(),
                classifier_name=base_combo["classifier_name"],
                classifier_config=classifier_config,
                samples_per_iteration=base_combo["samples_per_iteration"],
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
    classifier_al_config: ClassifierALSweepConfig,
) -> Iterator[Dict[str, Any]]:
    """
    Generate base parameter combinations (without random seeds).

    Args:
        classifier_al_config: Classifier AL sweep configuration

    Yields:
        Dictionary containing parameter combination
    """
    # Expand query strategy configurations
    query_strategy_combinations = []
    for strategy_name, strategy_params in classifier_al_config.query_strategies.items():
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
    for strategy_name, strategy_params in classifier_al_config.weighting_strategies.items():
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

    # Expand classifier configurations
    classifier_combinations = []
    for classifier_name, classifier_params in classifier_al_config.classifier_configs.items():
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
    for query_strategy, query_config in query_strategy_combinations:
        for weighting_strategy, weighting_config in weighting_strategy_combinations:
            for classifier_name, classifier_config in classifier_combinations:
                for samples_per_iter in classifier_al_config.samples_per_iteration:
                    for max_iters in classifier_al_config.max_iterations:
                        for max_total in classifier_al_config.max_total_samples:
                            for eval_interval in classifier_al_config.evaluation_interval:
                                for test_eval_interval in classifier_al_config.test_evaluation_interval:
                                    for eval_on_test in classifier_al_config.evaluate_on_test:
                                        yield {
                                            "query_strategy": query_strategy,
                                            "query_strategy_config": query_config,
                                            "weighting_strategy": weighting_strategy,
                                            "weighting_strategy_config": weighting_config,
                                            "classifier_name": classifier_name,
                                            "classifier_config": classifier_config,
                                            "samples_per_iteration": samples_per_iter,
                                            "max_iterations": max_iters,
                                            "max_total_samples": max_total,
                                            "evaluation_interval": eval_interval,
                                            "test_evaluation_interval": test_eval_interval,
                                            "evaluate_on_test": eval_on_test,
                                        }


def create_combination_id(combination_number: int, random_seed: int) -> str:
    """
    Generate unique identifier for parameter combination.

    Args:
        combination_number: Sequential combination number
        random_seed: Random seed for this combination

    Returns:
        Unique combination ID string
    """
    return f"combination_{combination_number:04d}_seed_{random_seed}"


def get_sweep_statistics(
    classifier_al_config: ClassifierALSweepConfig,
    random_seeds: List[int],
) -> Dict[str, Any]:
    """
    Get summary statistics for the parameter sweep.

    Args:
        classifier_al_config: Classifier AL sweep configuration
        random_seeds: List of random seeds

    Returns:
        Dictionary with sweep statistics
    """
    # Count base combinations
    base_combinations = list(_generate_base_combinations(classifier_al_config))
    base_count = len(base_combinations)

    # Total combinations including seeds
    total_combinations = base_count * len(random_seeds)

    # Count strategies and classifiers
    num_query_strategies = len(classifier_al_config.query_strategies)
    num_weighting_strategies = len(classifier_al_config.weighting_strategies)
    num_classifiers = len(classifier_al_config.classifier_configs)

    # Count parameter values
    num_samples_per_iter = len(classifier_al_config.samples_per_iteration)
    num_max_iterations = len(classifier_al_config.max_iterations)
    num_max_total_samples = len(classifier_al_config.max_total_samples)
    num_evaluation_intervals = len(classifier_al_config.evaluation_interval)
    num_test_evaluation_intervals = len(classifier_al_config.test_evaluation_interval)
    num_evaluate_on_test = len(classifier_al_config.evaluate_on_test)
    num_seeds = len(random_seeds)

    # Calculate query strategy parameter counts
    query_strategy_param_counts = {}
    for strategy_name, strategy_params in classifier_al_config.query_strategies.items():
        if not strategy_params:
            count = 1
        else:
            count = 1
            for param_values in strategy_params.values():
                count *= len(param_values)
        query_strategy_param_counts[strategy_name] = count

    # Calculate weighting strategy parameter counts
    weighting_strategy_param_counts = {}
    for strategy_name, strategy_params in classifier_al_config.weighting_strategies.items():
        if not strategy_params:
            count = 1
        else:
            count = 1
            for param_values in strategy_params.values():
                count *= len(param_values)
        weighting_strategy_param_counts[strategy_name] = count

    # Calculate classifier parameter counts
    classifier_param_counts = {}
    for classifier_name, classifier_params in classifier_al_config.classifier_configs.items():
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
            "names": list(classifier_al_config.query_strategies.keys()),
            "parameter_counts": query_strategy_param_counts,
        },
        "weighting_strategies": {
            "count": num_weighting_strategies,
            "names": list(classifier_al_config.weighting_strategies.keys()),
            "parameter_counts": weighting_strategy_param_counts,
        },
        "classifiers": {
            "count": num_classifiers,
            "names": list(classifier_al_config.classifier_configs.keys()),
            "parameter_counts": classifier_param_counts,
        },
        "al_parameters": {
            "samples_per_iteration": num_samples_per_iter,
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
