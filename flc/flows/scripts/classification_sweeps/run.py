#!/usr/bin/env python3
"""
Classification Sweep Script for Flow Classification Datasets

This script performs systematic classification sweeps across different algorithms and hyperparameters
on flow classification datasets. It trains models on training data and evaluates on test data,
generating comprehensive evaluation reports.

The script supports:
- Multiple classification algorithms (DecisionTree, RandomForest, XGBoost)
- Configurable hyperparameter sweeps for each algorithm
- Train/test dataset split evaluation
- Flow sampling with max_flows parameter for subset experiments
- Resumable execution (skips already completed experiments)
- Extensible parameter addition without losing previous results
- Detailed CSV reporting with all metrics and metadata

Usage:
    python run.py --config path/to/config.yaml [--verbose]

Example config structure:
    dataset_pairs:
      - train_split_path: "/path/to/dataset1/flow_labels/splits/train.yaml"
        test_split_path: "/path/to/dataset1/flow_labels/splits/test.yaml"
      - train_split_path: "/path/to/dataset2/group_labels/splits/train.yaml"
        test_split_path: "/path/to/dataset2/group_labels/splits/test.yaml"

    preprocessing:
      scaler_type: "robust"
      clip_quantiles: [0.01, 0.99]
      log_transform: true

    algorithms:
      random_forest:
        enabled: true
        hyperparameters:
          n_estimators: [100, 200, 500]
          max_depth: [10, 20, None]
          max_features: ["sqrt", "log2"]

      xgboost:
        enabled: true
        hyperparameters:
          n_estimators: [100, 200]
          max_depth: [3, 5, 6]
          learning_rate: [0.1, 0.2, 0.3]

    output_report_path: "classification_sweep_results.csv"

    # Optional: Limit number of flows for faster experiments
    max_flows: 1000  # Use only first 1000 training flows

Output CSV columns:
    - Dataset info: train_dataset_path, test_dataset_path, n_train_flows, n_test_flows,
                   dataset_name, train_split_name, test_split_name, etc.
    - Algorithm info: algorithm, hyperparameters (JSON string), parameter_hash
    - Model info: n_train_samples, has_feature_importance, oob_score, etc.
    - Evaluation metrics: accuracy, precision, recall, f1_score, log_loss, auc, etc.
    - Execution info: sweep_id, execution_time, error_message, config_version
"""

import argparse
import logging
import sys
import traceback

from flc.flows.scripts.classification_sweeps.config import ClassificationSweepConfig
from flc.flows.scripts.classification_sweeps.dataset_loader import (
    load_and_preprocess_dataset_pair,
    validate_dataset_paths,
)
from flc.flows.scripts.classification_sweeps.classification_executor import (
    execute_classification,
    validate_classification_inputs,
)
from flc.flows.scripts.classification_sweeps.evaluation_computer import evaluate_classification_sweep
from flc.flows.scripts.classification_sweeps.sweep_generator import (
    generate_sweep_combinations,
    filter_completed_sweeps_with_datasets,
    get_sweep_statistics,
    validate_sweep_combinations,
)
from flc.flows.scripts.classification_sweeps.report_manager import ReportManager


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run classification sweeps on flow classification datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = ClassificationSweepConfig.from_yaml(args.config)

        # Apply parameter defaults for extensibility
        config = config.merge_parameter_defaults()

        config.validate()
        logger.info("Configuration loaded and validated successfully")

        # Validate dataset paths
        logger.info("Validating dataset paths...")
        dataset_pairs = config.get_dataset_pairs()
        for dataset_pair in dataset_pairs:
            validate_dataset_paths(dataset_pair)
        logger.info(f"Validated {len(dataset_pairs)} dataset pairs")

        # Generate sweep combinations
        logger.info("Generating sweep combinations...")
        enabled_algorithms = config.get_enabled_algorithms()
        all_combinations = generate_sweep_combinations(enabled_algorithms)
        validate_sweep_combinations(all_combinations)

        sweep_stats = get_sweep_statistics(all_combinations)
        logger.info(f"Generated {sweep_stats['total_combinations']} total combinations")
        logger.info(f"Algorithms: {sweep_stats['algorithms']}")
        for algo, count in sweep_stats["total_by_algorithm"].items():
            logger.info(f"  {algo}: {count} combinations")

        # Initialize report manager and get completed combinations
        logger.info(f"Initializing report manager: {config.output_report_path}")
        report_manager = ReportManager(config.output_report_path)

        # Get all completed combinations across all datasets
        completed_combinations = report_manager.get_completed_combinations()
        logger.info(f"Found {len(completed_combinations)} completed experiments")

        # Execute sweep loop - filter per dataset pair for efficiency
        total_experiments = len(all_combinations) * len(dataset_pairs)
        experiment_count = 0

        logger.info(f"Starting execution of up to {total_experiments} experiments...")

        for dataset_pair in dataset_pairs:
            logger.info(f"Processing dataset pair:")
            logger.info(f"  Train: {dataset_pair.train_split_path}")
            logger.info(f"  Test:  {dataset_pair.test_split_path}")

            # Filter combinations for this specific dataset pair
            remaining_combinations = filter_completed_sweeps_with_datasets(
                all_combinations, completed_combinations, dataset_pair.train_split_path, dataset_pair.test_split_path
            )
            logger.info(f"Remaining combinations for this dataset pair: {len(remaining_combinations)}")

            if not remaining_combinations:
                logger.info("All experiments for this dataset pair already completed!")
                continue

            # Load and preprocess dataset pair
            try:
                X_train, y_train, X_test, y_test, dataset_metadata = load_and_preprocess_dataset_pair(
                    dataset_pair, config.preprocessing, config.max_flows, config.random_state
                )

                dataset_info = (
                    f"{dataset_metadata['n_train_flows']} train / "
                    f"{dataset_metadata['n_test_flows']} test flows, "
                    f"{dataset_metadata['n_features']} features, "
                    f"{dataset_metadata['n_classes']} classes"
                )
                if dataset_metadata.get("train_flows_sampled", False):
                    dataset_info += f" (sampled from {dataset_metadata['original_n_train_flows']} train flows)"
                logger.info(f"Loaded dataset: {dataset_info}")

            except Exception as e:
                logger.error(f"Failed to load dataset pair: {str(e)}")
                # Log error for all combinations with this dataset pair
                for combination in remaining_combinations:
                    experiment_count += 1
                    report_manager.append_result(
                        dataset_pair=dataset_pair,
                        sweep_combination=combination,
                        dataset_metadata={
                            "n_train_flows": None,
                            "n_test_flows": None,
                            "n_features": None,
                            "n_classes": None,
                        },
                        model_summary={
                            "algorithm": combination.algorithm,
                            "hyperparameters": combination.hyperparameters,
                            "multi_label": False,
                        },
                        evaluation_metrics={},
                        execution_time=0.0,
                        config_version=config.config_version,
                        error_message=f"Dataset loading failed: {str(e)}",
                    )
                continue

            # Execute classification for each combination
            for combination in remaining_combinations:
                experiment_count += 1

                logger.info(
                    f"[{experiment_count}/{total_experiments}] Executing {combination.algorithm} "
                    f"with {combination.hyperparameters}"
                )

                try:
                    # Validate inputs
                    validate_classification_inputs(
                        X_train, y_train, X_test, y_test, combination.algorithm, combination.hyperparameters
                    )

                    # Execute classification
                    y_pred, y_pred_proba, model_summary, execution_time = execute_classification(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        algorithm_name=combination.algorithm,
                        hyperparameters=combination.hyperparameters,
                        random_state=config.random_state,
                    )

                    # Evaluate classification
                    evaluation_metrics = evaluate_classification_sweep(y_test, y_pred, y_pred_proba)

                    # Log results
                    logger.info(f"  Completed in {execution_time:.2f}s")

                    if "macro_f1" in evaluation_metrics and evaluation_metrics["macro_f1"] is not None:
                        logger.info(f"  F1 macro: {evaluation_metrics['macro_f1']:.3f}")
                    if "micro_f1" in evaluation_metrics and evaluation_metrics["micro_f1"] is not None:
                        logger.info(f"  F1 micro: {evaluation_metrics['micro_f1']:.3f}")

                    # Append result to report
                    report_manager.append_result(
                        dataset_pair=dataset_pair,
                        sweep_combination=combination,
                        dataset_metadata=dataset_metadata,
                        model_summary=model_summary,
                        evaluation_metrics=evaluation_metrics,
                        execution_time=execution_time,
                        config_version=config.config_version,
                        error_message=None,
                    )

                except Exception as e:
                    logger.error(f"  Failed: {str(e)}")
                    if args.verbose:
                        logger.error(f"  Traceback: {traceback.format_exc()}")

                    # Append error result to report
                    report_manager.append_result(
                        dataset_pair=dataset_pair,
                        sweep_combination=combination,
                        dataset_metadata=dataset_metadata,
                        model_summary={
                            "algorithm": combination.algorithm,
                            "hyperparameters": combination.hyperparameters,
                            "multi_label": False,
                        },
                        evaluation_metrics={},
                        execution_time=0.0,
                        config_version=config.config_version,
                        error_message=str(e),
                    )

        # Final summary
        report_summary = report_manager.get_report_summary()
        logger.info("=" * 60)
        logger.info("CLASSIFICATION SWEEP COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total experiments: {report_summary['total_experiments']}")
        logger.info(f"Successful: {report_summary['successful_experiments']}")
        logger.info(f"Failed: {report_summary['failed_experiments']}")
        logger.info(f"Algorithms tested: {report_summary['algorithms_tested']}")
        logger.info(f"Config versions: {report_summary['config_versions']}")
        logger.info(f"Results saved to: {config.output_report_path}")

        # Show best results if any successful experiments
        if report_summary["successful_experiments"] > 0:
            try:
                best_results = report_manager.get_best_results(metric="f1_macro", top_k=3)
                if best_results is not None and len(best_results) > 0:
                    logger.info("\nTop 3 results by F1 macro:")
                    for i, (_, row) in enumerate(best_results.iterrows(), 1):
                        logger.info(
                            f"  {i}. {row['algorithm']} - F1: {row['f1_macro']:.3f}, "
                            f"Accuracy: {row.get('accuracy', 'N/A'):.3f}"
                        )
            except Exception as e:
                logger.debug(f"Could not display best results: {e}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if args.verbose:
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
