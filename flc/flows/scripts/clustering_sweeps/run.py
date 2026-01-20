#!/usr/bin/env python3
"""
Clustering Sweep Script for Flow Classification Datasets

This script performs systematic clustering sweeps across different algorithms and hyperparameters
on flow classification datasets. It generates comprehensive evaluation reports that include both
internal clustering metrics and external evaluation against true flow labels.

The script supports:
- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
- Configurable hyperparameter sweeps for each algorithm
- Multiple dataset splits for comprehensive evaluation
- Flow sampling with max_flows parameter for subset experiments
- Resumable execution (skips already completed experiments)
- Detailed CSV reporting with all metrics and metadata

Usage:
    python run.py --config path/to/config.yaml [--verbose]

Example config structure:
    dataset_split_paths:
      - "/path/to/dataset1/flow_labels/splits/train.yaml"
      - "/path/to/dataset2/group_labels/splits/test.yaml"

    preprocessing:
      scaler_type: "robust"
      clip_quantiles: [0.01, 0.99]
      log_transform: true

    algorithms:
      kmeans:
        enabled: true
        hyperparameters:
          n_clusters: [2, 4, 8, 16]
          init: ["k-means++", "random"]

      dbscan:
        enabled: true
        hyperparameters:
          eps: [0.1, 0.5, 1.0]
          min_samples: [3, 5, 10]

    output_report_path: "clustering_sweep_results.csv"

    # Optional: Limit number of flows for faster experiments
    max_flows: 1000  # Use only first 1000 flows (with stratified sampling)

Output CSV columns:
    - Dataset info: dataset_split_path, n_flows, n_unique_flow_labels, dataset_name, split_name,
                   original_n_flows, max_flows_used, flows_sampled
    - Algorithm info: algorithm, hyperparameters (JSON string)
    - Clustering results: n_clusters, n_noise_samples, noise_ratio
    - Evaluation metrics: silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                         adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score,
                         completeness_score, v_measure_score, and additional custom metrics
    - Execution info: sweep_id, execution_time, error_message
"""

import argparse
import logging
import sys
import traceback
from typing import Optional

from flc.flows.scripts.clustering_sweeps.config import ClusteringSweepConfig
from flc.flows.scripts.clustering_sweeps.dataset_loader import load_and_preprocess_dataset
from flc.flows.scripts.clustering_sweeps.clustering_executor import (
    execute_clustering,
    validate_clustering_inputs,
    get_clustering_summary,
)
from flc.flows.scripts.clustering_sweeps.evaluation_computer import evaluate_clustering
from flc.flows.scripts.clustering_sweeps.sweep_generator import (
    generate_sweep_combinations,
    filter_completed_sweeps,
    get_sweep_statistics,
)
from flc.flows.scripts.clustering_sweeps.report_manager import ReportManager


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
        description="Run clustering sweeps on flow classification datasets",
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
        config = ClusteringSweepConfig.from_yaml(args.config)
        config.validate()
        logger.info("Configuration loaded and validated successfully")

        # Generate sweep combinations
        logger.info("Generating sweep combinations...")
        enabled_algorithms = config.get_enabled_algorithms()
        all_combinations = generate_sweep_combinations(enabled_algorithms)

        sweep_stats = get_sweep_statistics(all_combinations)
        logger.info(f"Generated {sweep_stats['total_combinations']} total combinations")
        logger.info(f"Algorithms: {sweep_stats['algorithms']}")
        for algo, count in sweep_stats["total_by_algorithm"].items():
            logger.info(f"  {algo}: {count} combinations")

        # Initialize report manager and filter completed sweeps
        logger.info(f"Initializing report manager: {config.output_report_path}")
        report_manager = ReportManager(config.output_report_path)

        completed_sweep_ids = report_manager.get_completed_sweep_ids()
        logger.info(f"Found {len(completed_sweep_ids)} completed experiments")

        remaining_combinations = filter_completed_sweeps(all_combinations, completed_sweep_ids)
        logger.info(f"Remaining combinations to execute: {len(remaining_combinations)}")

        if not remaining_combinations:
            logger.info("All experiments already completed!")
            return

        # Execute sweep loop
        total_experiments = len(remaining_combinations) * len(config.dataset_split_paths)
        experiment_count = 0

        logger.info(f"Starting execution of {total_experiments} experiments...")

        for dataset_split_path in config.dataset_split_paths:
            logger.info(f"Processing dataset: {dataset_split_path}")

            # Load and preprocess dataset
            try:
                features, true_labels, metadata = load_and_preprocess_dataset(
                    dataset_split_path, config.preprocessing, config.max_flows, config.random_state
                )
                flow_info = f"{metadata['n_flows']} flows, {metadata['n_features']} features"
                if metadata.get("flows_sampled", False):
                    flow_info += f" (sampled from {metadata['original_n_flows']} flows)"
                logger.info(f"Loaded dataset: {flow_info}")

            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_split_path}: {str(e)}")
                # Log error for all combinations with this dataset
                for combination in remaining_combinations:
                    experiment_count += 1
                    report_manager.append_result(
                        dataset_split_path=dataset_split_path,
                        sweep_combination=combination,
                        dataset_metadata={"n_flows": None, "n_features": None, "n_unique_flow_labels": None},
                        clustering_summary={"n_clusters": None, "n_noise_samples": None, "noise_ratio": None},
                        evaluation_metrics={},
                        execution_time=0.0,
                        error_message=f"Dataset loading failed: {str(e)}",
                    )
                continue

            # Execute clustering for each combination
            for combination in remaining_combinations:
                experiment_count += 1

                logger.info(
                    f"[{experiment_count}/{total_experiments}] Executing {combination.algorithm} "
                    f"with {combination.hyperparameters}"
                )

                try:
                    # Validate inputs
                    validate_clustering_inputs(features, combination.algorithm, combination.hyperparameters)

                    # Execute clustering
                    cluster_labels, execution_time = execute_clustering(
                        features=features,
                        algorithm_name=combination.algorithm,
                        hyperparameters=combination.hyperparameters,
                        random_state=config.random_state,
                    )

                    # Get clustering summary
                    clustering_summary = get_clustering_summary(cluster_labels)

                    # Evaluate clustering
                    evaluation_metrics = evaluate_clustering(features, cluster_labels, true_labels)

                    # Log results
                    logger.info(
                        f"  Completed in {execution_time:.2f}s: "
                        f"{clustering_summary['n_clusters']} clusters, "
                        f"{clustering_summary['n_noise_samples']} noise points"
                    )

                    if "silhouette_score" in evaluation_metrics and evaluation_metrics["silhouette_score"] is not None:
                        logger.info(f"  Silhouette score: {evaluation_metrics['silhouette_score']:.3f}")

                    # Append result to report
                    report_manager.append_result(
                        dataset_split_path=dataset_split_path,
                        sweep_combination=combination,
                        dataset_metadata=metadata,
                        clustering_summary=clustering_summary,
                        evaluation_metrics=evaluation_metrics,
                        execution_time=execution_time,
                        error_message=None,
                    )

                except Exception as e:
                    logger.error(f"  Failed: {str(e)}")
                    if args.verbose:
                        logger.error(f"  Traceback: {traceback.format_exc()}")

                    # Append error result to report
                    report_manager.append_result(
                        dataset_split_path=dataset_split_path,
                        sweep_combination=combination,
                        dataset_metadata=metadata,
                        clustering_summary={"n_clusters": None, "n_noise_samples": None, "noise_ratio": None},
                        evaluation_metrics={},
                        execution_time=0.0,
                        error_message=str(e),
                    )

        # Final summary
        report_summary = report_manager.get_report_summary()
        logger.info("=" * 60)
        logger.info("CLUSTERING SWEEP COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total experiments: {report_summary['total_experiments']}")
        logger.info(f"Successful: {report_summary['successful_experiments']}")
        logger.info(f"Failed: {report_summary['failed_experiments']}")
        logger.info(f"Algorithms tested: {report_summary['algorithms_tested']}")
        logger.info(f"Results saved to: {config.output_report_path}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if args.verbose:
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
