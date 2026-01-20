#!/usr/bin/env python3
"""
Classification Evaluation Script

This script trains and evaluates classification models on flow classification datasets.
It supports multiple datasets, models, and hyperparameter configurations, producing
comprehensive evaluation reports in both CSV and JSON formats.

Usage:
    python run.py --config path/to/config.yaml [--verbose]

Example:
    cd flc/flows/scripts/classification_evaluation
    python run.py --config example_config.yaml --verbose
"""

import argparse
import sys
import traceback

from flc.flows.scripts.classification_evaluation.config import ClassificationEvaluationConfig
from flc.flows.scripts.classification_evaluation.evaluation_executor import EvaluationExecutor
from flc.flows.scripts.classification_evaluation.report_manager import ReportManager


def run_classification_evaluation(config_path: str, verbose: bool = False) -> None:
    """
    Run classification evaluation experiments based on configuration

    Args:
        config_path: Path to YAML configuration file
        verbose: Enable verbose output
    """
    # Load and validate configuration
    if verbose:
        print(f"Loading configuration from: {config_path}")
    config = ClassificationEvaluationConfig.from_yaml(config_path)
    config.validate()

    if verbose:
        print(f"Configuration loaded successfully")
        print(f"Datasets: {len(config.datasets)}")
        print(f"Models: {len(config.get_enabled_models())}")

    # Initialize executor and report manager
    executor = EvaluationExecutor(random_state=config.random_state)
    report_manager = ReportManager(config.output_report_path)

    # Get enabled models
    enabled_models = config.get_enabled_models()
    if not enabled_models:
        print("No enabled models found in configuration")
        return

    # Run evaluations for each dataset-model combination
    total_experiments = len(config.datasets) * len(enabled_models)
    experiment_count = 0

    print(f"\nStarting {total_experiments} classification experiments...")
    print("=" * 80)

    for dataset_config in config.datasets:
        for model_config in enabled_models:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}")
            print(f"Train Dataset: {dataset_config.name}")
            print(f"Test Datasets: {[td.name for td in dataset_config.test_datasets]}")
            print(f"Model: {model_config.name}")
            print(f"Hyperparameters: {model_config.hyperparameters}")

            # Run single evaluation (trains once, tests on multiple datasets)
            result = executor.run_single_evaluation(
                dataset_config=dataset_config,
                model_config=model_config,
                output_directory=config.output_directory,
                defaults={
                    "preprocessing": config.global_preprocessing,
                },
            )

            # Add to report
            report_manager.add_result(result)

            # Print key metrics for each test dataset
            for test_result in result["test_results"]:
                test_metrics = test_result["test_metrics"]
                print(f"Test Dataset '{test_result['test_dataset_name']}':")
                if "accuracy" in test_metrics:
                    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                else:
                    print(f"  Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
                    print(f"  Hamming Loss: {test_metrics['hamming_loss']:.4f}")

            print("✓ Experiment completed successfully")

    # Save final report
    print("\n" + "=" * 80)
    print("Saving final report...")
    report_manager.save_csv_report()

    # Print summary
    report_manager.print_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run classification evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", required=True, help="Path to YAML configuration file")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    run_classification_evaluation(args.config, args.verbose)


if __name__ == "__main__":
    main()
