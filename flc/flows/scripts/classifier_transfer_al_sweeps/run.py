#!/usr/bin/env python3
"""
Classifier Transfer Active Learning Sweeps

Main script for running systematic classifier-based transfer active learning experiments
across multiple dataset triplets and parameter combinations.
"""

import argparse
import logging
import multiprocessing
import time
from dataclasses import asdict
from typing import Dict, Any, Tuple

from tqdm import tqdm

from flc.flows.scripts.classifier_transfer_al_sweeps.config import ClassifierTransferALSweepConfig
from flc.flows.scripts.classifier_transfer_al_sweeps.sweep_generator import (
    generate_classifier_combinations,
    get_sweep_statistics,
)
from flc.flows.scripts.classifier_transfer_al_sweeps.transfer_al_executor import execute_classifier_al
from flc.flows.scripts.classifier_transfer_al_sweeps.output_manager import (
    create_base_output_directories,
    create_output_directory,
)
from flc.shared.transfer_active_learning.query_strategies.factory import QueryStrategyFactory
from flc.shared.transfer_active_learning.weighting_strategies.factory import WeightingStrategyFactory
from flc.shared.transfer_active_learning.base import ClassifierActiveLearningConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=format_str)


def process_experiment_task(
    task_data: Tuple[int, int, object, object, ClassifierTransferALSweepConfig],
) -> Dict[str, Any]:
    """
    Worker function to process a single experiment task (triplet + combination).

    Args:
        task_data: Tuple containing (triplet_idx, combo_idx, triplet_config, combination, config)

    Returns:
        Dictionary with execution results and metadata
    """
    triplet_idx, combo_idx, triplet_config, combination, config = task_data
    task_id = f"triplet_{triplet_idx}_combo_{combo_idx}_{combination.combination_id}"

    # Create output directory
    output_dir = create_output_directory(config.output_base_dir, triplet_idx, combination)

    # Execute active learning
    results = execute_classifier_al(
        triplet_config,
        combination,
        config.preprocessing,
        output_dir,
        config.max_flows,
    )

    # Add task metadata to results
    results["task_id"] = task_id
    results["triplet_idx"] = triplet_idx
    results["combo_idx"] = combo_idx

    return results


def process_experiment_with_raw(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to process a single experiment task from raw dict format.
    Converts the dict back to proper objects and calls process_experiment_task.

    Args:
        task_dict: Dictionary containing task data with keys:
            - triplet_idx, combo_idx, triplet_config_dict, combination_dict, config_dict

    Returns:
        Dictionary with execution results and metadata
    """
    from flc.flows.scripts.classifier_transfer_al_sweeps.sweep_generator import ClassifierALCombination
    from flc.flows.scripts.classifier_transfer_al_sweeps.config import (
        DatasetTripletConfig,
        ClassifierTransferALSweepConfig,
    )

    # Extract data from dict
    triplet_idx = task_dict["triplet_idx"]
    combo_idx = task_dict["combo_idx"]
    triplet_config_dict = task_dict["triplet_config_dict"]
    combination_dict = task_dict["combination_dict"]
    config_dict = task_dict["config_dict"]

    # Reconstruct objects from dicts
    triplet_config = DatasetTripletConfig(**triplet_config_dict)
    combination = ClassifierALCombination(**combination_dict)
    config = ClassifierTransferALSweepConfig(**config_dict)

    # Create task tuple and call original function
    task_data = (triplet_idx, combo_idx, triplet_config, combination, config)

    result = process_experiment_task(task_data)

    return result


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run classifier transfer active learning parameter sweeps")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dry-run", action="store_true", help="Show sweep statistics without running experiments")

    args = parser.parse_args()

    # Set multiprocessing start method if not already set (for compatibility)
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method already set, continue
        pass

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load and validate configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = ClassifierTransferALSweepConfig.from_yaml(args.config)

    # Apply parameter defaults
    config = config.merge_parameter_defaults()

    # Validate configuration
    config.validate()
    logger.info("Configuration validation successful")

    # Generate parameter combinations
    logger.info("Generating parameter combinations...")
    combinations = generate_classifier_combinations(config.classifier_al, config.random_states)

    # Get sweep statistics
    sweep_stats = get_sweep_statistics(config.classifier_al, config.random_states)

    # Display sweep information
    display_sweep_info(config, sweep_stats, logger)

    # Validate all configurations before starting sweep
    logger.info("Validating all parameter combinations...")
    validate_all_combinations(combinations, logger)
    logger.info("All parameter combinations validated successfully")

    if args.dry_run:
        logger.info("Dry run completed - no experiments executed")
        return

    # Create base output directories
    logger.info("Creating base output directories...")
    create_base_output_directories(config.output_base_dir, config.dataset_triplets)

    # Execute sweep
    execute_sweep(config, combinations, logger)

    logger.info("Classifier transfer AL sweep completed successfully")


def display_sweep_info(
    config: ClassifierTransferALSweepConfig,
    sweep_stats: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Display sweep configuration and statistics"""
    logger.info("=== Classifier Transfer AL Sweep Configuration ===")
    logger.info(f"Dataset triplets: {len(config.dataset_triplets)}")
    logger.info(f"Random seeds: {len(config.random_states)} ({config.random_states})")
    logger.info(f"Max flows per dataset: {config.max_flows}")
    logger.info(f"Output directory: {config.output_base_dir}")

    logger.info("\n=== Parameter Sweep Statistics ===")
    logger.info(f"Total combinations: {sweep_stats['total_combinations']}")
    logger.info(f"Base combinations: {sweep_stats['base_combinations']}")
    logger.info(f"Random seeds: {sweep_stats['random_seeds']}")

    logger.info(f"\nQuery strategies: {sweep_stats['query_strategies']['count']}")
    for strategy, count in sweep_stats["query_strategies"]["parameter_counts"].items():
        logger.info(f"  - {strategy}: {count} parameter combinations")

    logger.info(f"\nClassifiers: {sweep_stats['classifiers']['count']}")
    for classifier, count in sweep_stats["classifiers"]["parameter_counts"].items():
        logger.info(f"  - {classifier}: {count} parameter combinations")

    al_params = sweep_stats["al_parameters"]
    logger.info(f"\nAL parameters:")
    logger.info(f"  - Samples per iteration: {al_params['samples_per_iteration']} values")
    logger.info(f"  - Max iterations: {al_params['max_iterations']} values")
    logger.info(f"  - Max total samples: {al_params['max_total_samples']} values")

    total_experiments = len(config.dataset_triplets) * sweep_stats["total_combinations"]
    logger.info(f"\nEstimated total experiments: {total_experiments}")


def execute_sweep(
    config: ClassifierTransferALSweepConfig,
    combinations: list,
    logger: logging.Logger,
) -> None:
    """Execute the parameter sweep with parallel processing"""
    total_experiments = len(config.dataset_triplets) * len(combinations)
    logger.info(f"Starting parallel sweep execution with {total_experiments} total experiments")
    logger.info(f"Using {config.n_jobs} parallel processes")
    start_time = time.time()

    # Create all task combinations
    tasks = []
    for triplet_idx, triplet_config in enumerate(config.dataset_triplets):
        for combo_idx, combination in enumerate(combinations):
            tasks.append((triplet_idx, combo_idx, triplet_config, combination, config))

    logger.info(f"Created {len(tasks)} tasks to execute")

    # Execute tasks
    if config.n_jobs == 1:
        # Sequential execution for debugging or single-core systems
        logger.info("Running in sequential mode (n_jobs=1)")
        for task in tqdm(tasks, desc="Processing experiments"):
            result = process_experiment_task(task)
            logger.info(f"Completed: {result['task_id']}")
    else:
        # Parallel execution - convert to dicts for multiprocessing
        logger.info(f"Running in parallel mode with {config.n_jobs} processes")

        # Convert tasks to dict format for pickling
        task_dicts = []
        for triplet_idx, combo_idx, triplet_config, combination, task_config in tasks:
            task_dict = {
                "triplet_idx": triplet_idx,
                "combo_idx": combo_idx,
                "triplet_config_dict": asdict(triplet_config),
                "combination_dict": asdict(combination),
                "config_dict": asdict(task_config),
            }
            task_dicts.append(task_dict)

        with multiprocessing.Pool(config.n_jobs) as pool:
            with tqdm(total=len(task_dicts), desc="Processing experiments") as pbar:
                for result in pool.imap_unordered(process_experiment_with_raw, task_dicts):
                    # logger.info(f"Completed: {result['task_id']}")
                    pbar.update(1)

    # Display final summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n=== Sweep Execution Summary ===")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Total execution time: {elapsed_time:.1f} seconds")


def validate_all_combinations(combinations: list, logger: logging.Logger) -> None:
    """
    Validate all parameter combinations by attempting to create strategy instances.

    Args:
        combinations: List of ClassifierALCombination objects
        logger: Logger instance for reporting validation progress

    Raises:
        ValueError: If any combination has invalid configuration
    """
    logger.info(f"Validating {len(combinations)} parameter combinations...")

    for i, combination in enumerate(combinations):
        # Validate query strategy
        QueryStrategyFactory.create_classifier_strategy(combination.query_strategy, combination.query_strategy_config)

        # Validate weighting strategy
        WeightingStrategyFactory.create(combination.weighting_strategy, combination.weighting_strategy_config)

        # Validate classifier AL config
        ClassifierActiveLearningConfig(
            samples_per_iteration=combination.samples_per_iteration,
            classifier_name=combination.classifier_name,
            classifier_config=combination.classifier_config,
            random_state=combination.random_state,
        )

        if (i + 1) % 100 == 0:
            logger.info(f"Validated {i + 1}/{len(combinations)} combinations")


if __name__ == "__main__":
    main()
