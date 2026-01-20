"""
Clustering Sweeps Module for Flow Classification Datasets

This module provides a comprehensive framework for performing systematic clustering sweeps
on flow classification datasets. It supports multiple clustering algorithms with 
configurable hyperparameter grids and generates detailed evaluation reports.

Main Components:
- config: Configuration system for sweep parameters and dataset settings
- dataset_loader: Integration with FlowClassificationDataset for data loading and preprocessing
- clustering_executor: Execution of clustering algorithms using the existing clustering framework
- evaluation_computer: Comprehensive evaluation using internal and external metrics
- sweep_generator: Generation of all parameter combinations for systematic exploration
- report_manager: CSV report management with resumable execution support
- run: Main execution script with command-line interface

Usage:
    python -m flc.flows.scripts.clustering_sweeps.run --config config.yaml --verbose

The framework integrates seamlessly with:
- flc.flows.dataset.dataset.FlowClassificationDataset for data loading
- flc.shared.clustering.factory.ClusteringFactory for algorithm creation
- flc.shared.clustering.evaluation.ClusteringEvaluator for metrics computation

Features:
- Supports K-means, DBSCAN, and Hierarchical clustering algorithms
- Configurable preprocessing with scaling, clipping, and log transforms
- Comprehensive evaluation with internal and external clustering metrics
- Resumable execution that skips completed experiments
- Detailed CSV reporting with all hyperparameters and results
- Error handling and logging for robust execution
"""

from .config import ClusteringSweepConfig, AlgorithmSweepConfig
from flc.shared.preprocessing.config import PreprocessingConfig
from .dataset_loader import load_and_preprocess_dataset, create_true_labels_for_evaluation
from .clustering_executor import execute_clustering, validate_clustering_inputs, get_clustering_summary
from .evaluation_computer import evaluate_clustering
from .sweep_generator import generate_sweep_combinations, filter_completed_sweeps, get_sweep_statistics, SweepCombination
from .report_manager import ReportManager

__all__ = [
    "ClusteringSweepConfig",
    "AlgorithmSweepConfig", 
    "PreprocessingConfig",
    "load_and_preprocess_dataset",
    "create_true_labels_for_evaluation",
    "execute_clustering",
    "validate_clustering_inputs",
    "get_clustering_summary",
    "evaluate_clustering",
    "generate_sweep_combinations",
    "filter_completed_sweeps",
    "get_sweep_statistics",
    "SweepCombination",
    "ReportManager"
]