import os
from typing import Optional, Dict, Any, Tuple
import numpy as np

from flc.flows.dataset.dataset import FlowClassificationDataset

from flc.shared.transfer_active_learning.base import ClusteringActiveLearningConfig
from flc.shared.transfer_active_learning.experiments import ClusteringExperimentRunner, ClusteringExperimentConfig
from flc.shared.transfer_active_learning.query_strategies.factory import QueryStrategyFactory
from flc.shared.transfer_active_learning.weighting_strategies.factory import WeightingStrategyFactory
from flc.shared.transfer_active_learning.label_selection.factory import LabelSelectionFactory
from flc.shared.preprocessing.config import PreprocessingConfig
from .sweep_generator import ClusteringALCombination
from .config import DatasetTripletConfig


def execute_clustering_al(
    triplet_config: DatasetTripletConfig,
    combination: ClusteringALCombination,
    clustering_preprocessing: PreprocessingConfig,
    classification_preprocessing: PreprocessingConfig,
    output_dir: str,
    max_flows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute clustering-based transfer active learning with specific output directory.

    Args:
        triplet_config: Dataset triplet configuration with source/target/test paths
        combination: Parameter combination for AL execution
        clustering_preprocessing: Preprocessing configuration for clustering
        classification_preprocessing: Preprocessing configuration for classification
        output_dir: Output directory for results
        max_flows: Optional maximum number of flows to use per dataset

    Returns:
        Dictionary with execution results and metadata

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If AL execution fails
    """
    # Create output directory
    create_output_directory(output_dir)

    # Load datasets from triplet configuration
    (
        source_features,
        source_labels,
        target_features,
        target_labels,
        test_features,
        test_labels,
        source_dataset_name,
        target_dataset_name,
        test_dataset_name,
        source_sample_ids,
        target_sample_ids,
    ) = load_and_prepare_datasets(triplet_config, max_flows, combination.random_state)

    # Validate loaded datasets
    validate_al_inputs(
        source_features, source_labels, target_features, target_labels, test_features, test_labels, combination
    )

    # Create clustering AL configuration
    clustering_al_config = ClusteringActiveLearningConfig(
        query_strategy=combination.query_strategy,
        query_strategy_config=combination.query_strategy_config,
        weighting_strategy=combination.weighting_strategy,
        weighting_strategy_config=combination.weighting_strategy_config,
        label_selection_strategy=combination.label_selection_strategy,
        label_selection_config=combination.label_selection_config,
        clustering_algorithm=combination.clustering_algorithm,
        clustering_config=combination.clustering_config,
        max_iterations=combination.max_iterations,
        max_total_samples=combination.max_total_samples,
        classifier_name=combination.classifier_name,
        classifier_config=combination.classifier_config,
        evaluation_interval=combination.evaluation_interval,
        test_evaluation_interval=combination.test_evaluation_interval,
        evaluate_on_test=combination.evaluate_on_test,
        output_dir=output_dir,
        save_models=False,  # Save space for large sweeps
        random_state=combination.random_state,
    )

    # Create clustering experiment configuration
    experiment_config = ClusteringExperimentConfig(
        clustering_al_config=clustering_al_config,
        clustering_preprocessing=clustering_preprocessing,
        classification_preprocessing=classification_preprocessing,
        experiment_name=combination.combination_id,
        output_dir=output_dir,
        max_iterations=combination.max_iterations,
        random_state=combination.random_state,
        source_dataset_name=source_dataset_name,
        target_dataset_name=target_dataset_name,
        test_dataset_name=test_dataset_name,
        source_dataset_path=triplet_config.source_split_path,
        target_dataset_path=triplet_config.target_split_path,
        test_dataset_path=triplet_config.test_split_path,
        save_iterations_csv=True,
        save_evaluations_csv=True,
        save_configurations=True,
        save_interval=1,
        evaluate_on_test=combination.evaluate_on_test,
        evaluation_interval=combination.evaluation_interval,
    )

    # Create strategy instances using factories
    query_strategy = QueryStrategyFactory.create_cluster_strategy(
        combination.query_strategy, combination.query_strategy_config
    )
    weighting_strategy = WeightingStrategyFactory.create(
        combination.weighting_strategy, combination.weighting_strategy_config
    )
    label_selection_strategy = LabelSelectionFactory.create(
        combination.label_selection_strategy, combination.label_selection_config
    )

    # Create feature mask based on dataset preprocessing exclusions
    source_dataset = FlowClassificationDataset(triplet_config.source_split_path)
    preprocess_feature_mask = source_dataset.get_preprocessing_mask()

    # Create clustering experiment runner
    runner = ClusteringExperimentRunner(
        source_sample_ids=source_sample_ids,
        source_features=source_features,
        source_labels=source_labels,
        target_sample_ids=target_sample_ids,
        target_features=target_features,
        target_labels=target_labels,
        test_features=test_features,
        test_labels=test_labels,
        query_strategy=query_strategy,
        weighting_strategy=weighting_strategy,
        label_selection_strategy=label_selection_strategy,
        experiment_config=experiment_config,
        preprocess_feature_mask=preprocess_feature_mask,
    )

    # Run clustering experiment
    results = runner.run()

    # Add combination metadata to results
    results["combination_metadata"] = {
        "combination_id": combination.combination_id,
        "learner_type": combination.learner_type,
        "query_strategy": combination.query_strategy,
        "query_strategy_config": combination.query_strategy_config,
        "weighting_strategy": combination.weighting_strategy,
        "weighting_strategy_config": combination.weighting_strategy_config,
        "clustering_algorithm": combination.clustering_algorithm,
        "clustering_config": combination.clustering_config,
        "label_selection_strategy": combination.label_selection_strategy,
        "label_selection_config": combination.label_selection_config,
        "classifier_name": combination.classifier_name,
        "classifier_config": combination.classifier_config,
        "max_iterations": combination.max_iterations,
        "max_total_samples": combination.max_total_samples,
        "evaluation_interval": combination.evaluation_interval,
        "test_evaluation_interval": combination.test_evaluation_interval,
        "evaluate_on_test": combination.evaluate_on_test,
        "random_state": combination.random_state,
        "output_dir": output_dir,
    }

    return results


def validate_al_inputs(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    test_features: Optional[np.ndarray],
    test_labels: Optional[np.ndarray],
    combination: ClusteringALCombination,
) -> None:
    """
    Validate inputs before active learning execution.

    Args:
        source_features: Source dataset features
        source_labels: Source dataset labels
        target_features: Target dataset features
        target_labels: Target dataset labels
        test_features: Optional test dataset features
        test_labels: Optional test dataset labels
        combination: Parameter combination for AL execution

    Raises:
        ValueError: If inputs are invalid
    """
    # Validate source data
    if source_features is None or source_labels is None:
        raise ValueError("Source features and labels are required")

    if len(source_features) == 0 or len(source_labels) == 0:
        raise ValueError("Source datasets cannot be empty")

    if len(source_features) != len(source_labels):
        raise ValueError(f"Source feature/label count mismatch: {len(source_features)} vs {len(source_labels)}")

    # Validate target data
    if target_features is None or target_labels is None:
        raise ValueError("Target features and labels are required")

    if len(target_features) == 0 or len(target_labels) == 0:
        raise ValueError("Target datasets cannot be empty")

    if len(target_features) != len(target_labels):
        raise ValueError(f"Target feature/label count mismatch: {len(target_features)} vs {len(target_labels)}")

    # Validate test data consistency
    if (test_features is None) != (test_labels is None):
        raise ValueError("Test features and labels must both be provided or both be None")

    if test_features is not None and test_labels is not None:
        if len(test_features) != len(test_labels):
            raise ValueError(f"Test feature/label count mismatch: {len(test_features)} vs {len(test_labels)}")

    # Validate feature dimensions match
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: source={source_features.shape[1]}, " f"target={target_features.shape[1]}"
        )

    if test_features is not None and source_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: source={source_features.shape[1]}, " f"test={test_features.shape[1]}"
        )

    # Validate label dimensions match
    if source_labels.shape[1] != target_labels.shape[1]:
        raise ValueError(
            f"Label dimension mismatch: source={source_labels.shape[1]}, " f"target={target_labels.shape[1]}"
        )

    if test_labels is not None and source_labels.shape[1] != test_labels.shape[1]:
        raise ValueError(f"Label dimension mismatch: source={source_labels.shape[1]}, " f"test={test_labels.shape[1]}")

    # Validate combination parameters
    if combination.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    if combination.evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive")

    if combination.test_evaluation_interval <= 0:
        raise ValueError("test_evaluation_interval must be positive")

    # Validate max_total_samples if specified
    if combination.max_total_samples is not None:
        if combination.max_total_samples <= 0:
            raise ValueError("max_total_samples must be positive")

        if combination.max_total_samples > len(target_features):
            raise ValueError(
                f"max_total_samples ({combination.max_total_samples}) "
                f"exceeds target dataset size ({len(target_features)})"
            )

    # Validate clustering algorithm is specified
    if not combination.clustering_algorithm:
        raise ValueError("clustering_algorithm must be specified")

    # Validate query strategy is specified
    if not combination.query_strategy:
        raise ValueError("query_strategy must be specified")

    # Validate label selection strategy is specified
    if not combination.label_selection_strategy:
        raise ValueError("label_selection_strategy must be specified")


def create_output_directory(output_dir: str) -> None:
    """
    Create output directory for AL results.

    Args:
        output_dir: Path to output directory

    Raises:
        ValueError: If directory cannot be created
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {output_dir}: {e}") from e

    # Verify directory is writable
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_dir}")


def load_and_prepare_datasets(
    triplet_config: DatasetTripletConfig,
    max_flows: Optional[int],
    random_state: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str, np.ndarray, np.ndarray
]:
    """
    Load datasets from triplet configuration and prepare for transfer AL (without preprocessing).

    Args:
        triplet_config: Dataset triplet configuration with paths
        max_flows: Maximum number of flows to use per dataset (for faster experiments)
        random_state: Random state for shuffling datasets

    Returns:
        Tuple of (source_features, source_labels, target_features, target_labels,
                 test_features, test_labels, source_name, target_name, test_name,
                 source_sample_ids, target_sample_ids)

    Raises:
        ValueError: If datasets are incompatible or validation fails
    """
    # Load datasets (without any preprocessing)
    source_dataset = FlowClassificationDataset(triplet_config.source_split_path)
    target_dataset = FlowClassificationDataset(triplet_config.target_split_path)
    test_dataset = FlowClassificationDataset(triplet_config.test_split_path)

    # Apply flow limit to source dataset
    if max_flows is not None and len(source_dataset) > max_flows:
        source_dataset = source_dataset.subsample(max_flows, random_seed=random_state)

    # Apply flow limit to target dataset
    if max_flows is not None and len(target_dataset) > max_flows:
        target_dataset = target_dataset.subsample(max_flows, random_seed=random_state)

    # No limit on test dataset, it should be validated on the full set

    # Shuffle datasets (except test dataset)
    source_dataset.shuffle(random_state=random_state)
    target_dataset.shuffle(random_state=random_state)

    # Validate dataset compatibility
    validate_dataset_triplet(source_dataset, target_dataset, test_dataset)

    # Extract dataset names from metadata
    source_dataset_name = source_dataset.get_metadata()["dataset_name"]
    target_dataset_name = target_dataset.get_metadata()["dataset_name"]
    test_dataset_name = test_dataset.get_metadata()["dataset_name"]

    # Extract flow IDs from datasets
    source_sample_ids = np.array(source_dataset.get_flow_ids())
    target_sample_ids = np.array(target_dataset.get_flow_ids())

    # Convert to numpy arrays for transfer AL (without preprocessing)
    source_features, source_labels = source_dataset.to_sklearn_format(preprocessed=False)
    target_features, target_labels = target_dataset.to_sklearn_format(preprocessed=False)
    test_features, test_labels = test_dataset.to_sklearn_format(preprocessed=False)

    return (
        source_features,
        source_labels,
        target_features,
        target_labels,
        test_features,
        test_labels,
        source_dataset_name,
        target_dataset_name,
        test_dataset_name,
        source_sample_ids,
        target_sample_ids,
    )


def validate_dataset_triplet(
    source_dataset: FlowClassificationDataset,
    target_dataset: FlowClassificationDataset,
    test_dataset: FlowClassificationDataset,
) -> None:
    """
    Validate that datasets are compatible for transfer active learning.

    Args:
        source_dataset: Source dataset (fully labeled)
        target_dataset: Target dataset (for AL selection)
        test_dataset: Test dataset for evaluation

    Raises:
        ValueError: If datasets are incompatible
    """
    # Check label types match
    label_types = {source_dataset.label_type, target_dataset.label_type, test_dataset.label_type}
    if len(label_types) > 1:
        raise ValueError("Source, target, and test datasets must have the same label type")

    # Check feature alignment
    source_features = set(source_dataset.feature_names)
    target_features = set(target_dataset.feature_names)
    test_features = set(test_dataset.feature_names)

    if source_features != target_features or source_features != test_features:
        raise ValueError("Source, target, and test datasets must have the same feature names")

    # Validate datasets have data
    if len(source_dataset) == 0:
        raise ValueError("Source dataset is empty")
    if len(target_dataset) == 0:
        raise ValueError("Target dataset is empty")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")
