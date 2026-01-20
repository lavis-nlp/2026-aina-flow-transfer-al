import random
from typing import Optional, List

from flc.flows.dataset.dataset import FlowClassificationDataset


def subsample_dataset(
    dataset: FlowClassificationDataset,
    max_flows: int,
    random_seed: Optional[int] = None,
) -> FlowClassificationDataset:
    """
    Create a subsampled FlowClassificationDataset from an existing one.

    Args:
        dataset: The original FlowClassificationDataset to subsample from
        max_flows: Maximum number of flows to include in the subsampled dataset
        random_seed: Random seed for reproducible subsampling. If None, uses random sampling.

    Returns:
        New FlowClassificationDataset with at most max_flows flows

    Raises:
        ValueError: If max_flows is less than 1 or greater than dataset size
    """
    if max_flows < 1:
        raise ValueError("max_flows must be at least 1")

    original_size = len(dataset)
    if max_flows >= original_size:
        # Return a copy of the original dataset if max_flows exceeds current size
        return _create_copy_dataset(dataset, dataset.get_flow_ids())

    # Set random seed if provided
    if random_seed is None:
        random_seed = random.randint(0, 2**31 - 1)

    # Perform random sampling
    all_flow_ids = dataset.get_flow_ids()
    selected_flow_ids = random.Random(random_seed).sample(all_flow_ids, max_flows)

    return _create_copy_dataset(dataset, selected_flow_ids)


def _create_copy_dataset(
    original_dataset: FlowClassificationDataset,
    selected_flow_ids: List[int],
) -> FlowClassificationDataset:
    """
    Create a new FlowClassificationDataset with only the selected flow IDs.

    This creates a lightweight copy that shares the same file paths and configuration
    but filters the data to only include the selected flows.
    """
    # Create a new instance using the same split file path
    # We'll need to temporarily modify the split to contain only selected flow IDs
    new_dataset = FlowClassificationDataset.__new__(FlowClassificationDataset)

    # Copy configuration from original
    new_dataset.split = original_dataset.split
    new_dataset.label_type = original_dataset.label_type
    new_dataset.exclude_features = original_dataset.exclude_features
    new_dataset.feature_mappings = original_dataset.feature_mappings
    new_dataset.dataset_path = original_dataset.dataset_path
    new_dataset.features_file = original_dataset.features_file
    new_dataset.labels_file = original_dataset.labels_file
    new_dataset.id2pcap_file = original_dataset.id2pcap_file

    # Filter features and labels to selected flow IDs
    selected_flow_ids_set = set(selected_flow_ids)
    new_dataset.features = original_dataset.features.loc[
        original_dataset.features.index.isin(selected_flow_ids_set)
    ].copy()
    new_dataset.labels = original_dataset.labels.loc[original_dataset.labels.index.isin(selected_flow_ids_set)].copy()

    # Ensure consistent ordering
    common_flow_ids = sorted(set(new_dataset.features.index).intersection(set(new_dataset.labels.index)))
    new_dataset.features = new_dataset.features.loc[common_flow_ids]
    new_dataset.labels = new_dataset.labels.loc[common_flow_ids]

    # Copy preprocessing configuration
    new_dataset._preprocessor = original_dataset._preprocessor
    new_dataset._exclude_from_preprocessing = original_dataset._exclude_from_preprocessing

    # Copy id2pcap mapping
    new_dataset._id2pcap_mapping = original_dataset._id2pcap_mapping

    return new_dataset
