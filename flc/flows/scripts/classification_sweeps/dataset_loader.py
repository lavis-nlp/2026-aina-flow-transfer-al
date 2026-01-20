import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

from flc.flows.dataset.dataset import FlowClassificationDataset
from flc.shared.preprocessing.config import PreprocessingConfig
from .config import DatasetPairConfig


logger = logging.getLogger(__name__)


def load_and_preprocess_dataset_pair(
    dataset_pair: DatasetPairConfig,
    preprocessing_config: PreprocessingConfig,
    max_flows: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess a train/test dataset pair.

    Args:
        dataset_pair: Configuration for train/test dataset paths
        preprocessing_config: Preprocessing configuration
        max_flows: Maximum number of flows to use (applied to train set)
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, metadata)
        where metadata contains information about both datasets

    Raises:
        ValueError: If datasets are incompatible or loading fails
        FileNotFoundError: If dataset files don't exist
    """
    logger.debug(f"Loading train dataset: {dataset_pair.train_split_path}")
    logger.debug(f"Loading test dataset: {dataset_pair.test_split_path}")

    # Load train dataset
    train_dataset = FlowClassificationDataset(dataset_pair.train_split_path)

    # Load test dataset
    test_dataset = FlowClassificationDataset(dataset_pair.test_split_path)

    # Validate dataset compatibility
    _validate_dataset_compatibility(train_dataset, test_dataset)

    # Apply flow sampling to train dataset if requested
    original_n_train_flows = len(train_dataset)
    if max_flows is not None and len(train_dataset) > max_flows:
        logger.debug(f"Sampling {max_flows} flows from {len(train_dataset)} train flows")
        train_dataset = train_dataset.subsample(max_flows, random_seed=random_state)
        flows_sampled = True
    else:
        flows_sampled = False

    # Setup preprocessing on train dataset using config
    train_dataset.set_preprocessor(config=preprocessing_config)

    # Fit preprocessor on train data and get processed training data
    X_train, y_train = train_dataset.to_sklearn_format(preprocessed=True)

    # Setup separate preprocessor for test dataset with same config
    test_dataset.set_preprocessor(config=preprocessing_config)

    # Get processed test data
    X_test, y_test = test_dataset.to_sklearn_format(preprocessed=True)

    # Collect metadata
    train_metadata = train_dataset.get_metadata()
    test_metadata = test_dataset.get_metadata()
    train_statistics = train_dataset.get_statistics()
    test_statistics = test_dataset.get_statistics()

    metadata = {
        # Train dataset info
        "n_train_flows": len(train_dataset),
        "original_n_train_flows": original_n_train_flows,
        "train_flows_sampled": flows_sampled,
        "max_flows_used": max_flows,
        # Test dataset info
        "n_test_flows": len(test_dataset),
        # Shared info (from train dataset)
        "n_features": X_train.shape[1],
        "n_classes": y_train.shape[1],
        "dataset_name": train_metadata["dataset_name"],
        "label_type": train_metadata["label_type"],
        # Split info
        "train_split_name": train_metadata["split_name"],
        "test_split_name": test_metadata["split_name"],
        # File paths
        "train_split_path": dataset_pair.train_split_path,
        "test_split_path": dataset_pair.test_split_path,
        # Label distribution
        "train_label_distribution": train_dataset.get_label_distribution(),
        "test_label_distribution": test_dataset.get_label_distribution(),
        # Statistics
        "train_statistics": train_statistics,
        "test_statistics": test_statistics,
        # Preprocessing info
        "preprocessing_enabled": preprocessing_config.enabled,
        "scaler_type": preprocessing_config.scaler_type,
        "clip_quantiles": preprocessing_config.clip_quantiles,
        "log_transform": preprocessing_config.log_transform,
    }

    logger.debug(f"Loaded dataset pair: {metadata['n_train_flows']} train / {metadata['n_test_flows']} test flows")
    logger.debug(f"Features: {metadata['n_features']}, Classes: {metadata['n_classes']}")

    return X_train, y_train, X_test, y_test, metadata


def _validate_dataset_compatibility(
    train_dataset: FlowClassificationDataset, test_dataset: FlowClassificationDataset
) -> None:
    """
    Validate that train and test datasets are compatible.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset

    Raises:
        ValueError: If datasets are incompatible
    """
    # Check label types match
    if train_dataset.label_type != test_dataset.label_type:
        raise ValueError(
            f"Label type mismatch: train={train_dataset.label_type.value}, " f"test={test_dataset.label_type.value}"
        )

    # Check feature columns match
    train_features = set(train_dataset.get_features().columns)
    test_features = set(test_dataset.get_features().columns)

    if train_features != test_features:
        missing_in_test = train_features - test_features
        missing_in_train = test_features - train_features

        error_msg = "Feature mismatch between train and test datasets"
        if missing_in_test:
            error_msg += f"\nMissing in test: {missing_in_test}"
        if missing_in_train:
            error_msg += f"\nMissing in train: {missing_in_train}"

        raise ValueError(error_msg)

    # Check that datasets have some flows
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")


def get_label_names_mapping(dataset: FlowClassificationDataset) -> Dict[int, str]:
    """
    Get mapping from label indices to label names.

    Args:
        dataset: Dataset to get label names from

    Returns:
        Dictionary mapping label index to label name
    """
    return dataset.get_label_names()


def validate_dataset_paths(dataset_pair: DatasetPairConfig) -> None:
    """
    Validate that dataset paths exist and are accessible.

    Args:
        dataset_pair: Dataset pair configuration to validate

    Raises:
        FileNotFoundError: If any dataset file doesn't exist
        ValueError: If paths are invalid
    """
    train_path = Path(dataset_pair.train_split_path)
    test_path = Path(dataset_pair.test_split_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset file does not exist: {dataset_pair.train_split_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset file does not exist: {dataset_pair.test_split_path}")
