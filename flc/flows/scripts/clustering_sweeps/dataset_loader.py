from typing import Tuple, Dict, Any, Optional
import numpy as np

from flc.flows.dataset.dataset import FlowClassificationDataset
from flc.shared.preprocessing.config import PreprocessingConfig


def load_and_preprocess_dataset(
    split_file_path: str,
    preprocessing_config: PreprocessingConfig,
    max_flows: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load FlowClassificationDataset and extract features/labels for clustering.

    Args:
        split_file_path: Path to dataset split YAML file
        preprocessing_config: Preprocessing configuration
        max_flows: Maximum number of flows to use (if None, use all flows)
        random_state: Random seed for shuffling and sampling flows

    Returns:
        - features: Preprocessed feature matrix (n_flows, n_features)
        - true_labels: Multi-hot encoded labels for external evaluation (n_flows, n_classes)
        - metadata: Dataset metadata including flow count, unique labels, etc.

    Raises:
        ValueError: If dataset loading or preprocessing fails
        FileNotFoundError: If split file or data files don't exist
    """
    try:
        # Load dataset using existing FlowClassificationDataset
        dataset = FlowClassificationDataset(split_file_path)

        # Apply shuffling to the data before subsampling
        dataset.shuffle(random_state=random_state)

        # Apply flow subsampling if max_flows is specified
        original_n_flows = len(dataset)
        if max_flows is not None and max_flows < original_n_flows:
            dataset = dataset.subsample(max_flows=max_flows, random_seed=random_state)
            sampled_n_flows = len(dataset)
        else:
            sampled_n_flows = original_n_flows

        # Set preprocessor configuration for on-demand processing
        dataset.set_preprocessor(config=preprocessing_config)

        # Get preprocessed features and labels using sklearn format
        features, labels_multihot = dataset.to_sklearn_format(preprocessed=True)

        # Extract metadata
        label_distribution = dataset.get_label_distribution()

        metadata = {
            "dataset_split_path": split_file_path,
            "n_flows": sampled_n_flows,
            "n_features": features.shape[1],
            "n_unique_flow_labels": len(label_distribution),
            "dataset_name": dataset.split.dataset_name,
            "split_name": dataset.split.split_name,
            "label_type": dataset.label_type.value,
            "label_distribution": label_distribution,
            "preprocessing_config": {
                "enabled": preprocessing_config.enabled,
                "scaler_type": preprocessing_config.scaler_type,
                "clip_quantiles": preprocessing_config.clip_quantiles,
                "log_transform": preprocessing_config.log_transform,
                "excluded_features": dataset.get_excluded_from_preprocessing(),
            },
            "original_n_flows": original_n_flows,
            "max_flows_used": max_flows,
            "flows_sampled": max_flows is not None and max_flows < original_n_flows,
        }

        return features, labels_multihot, metadata

    except Exception as e:
        raise ValueError(f"Failed to load and preprocess dataset {split_file_path}: {str(e)}") from e


def create_true_labels_for_evaluation(labels_multihot: np.ndarray) -> np.ndarray:
    """
    Convert multi-hot encoded labels to single label array for external evaluation.

    For flows with multiple labels, uses the first label (lowest index).

    Args:
        labels_multihot: Multi-hot encoded labels (n_flows, n_classes)

    Returns:
        Single label array (n_flows,) where each flow has one label
    """
    # Use argmax to get the first (lowest index) label for each flow
    true_labels_single = np.argmax(labels_multihot, axis=1)

    # Handle flows with no labels (should not happen due to dataset validation)
    # but include as safety check
    no_labels_mask = np.sum(labels_multihot, axis=1) == 0
    if np.any(no_labels_mask):
        # Assign a default label (0) to flows without labels
        true_labels_single[no_labels_mask] = 0

    return true_labels_single
