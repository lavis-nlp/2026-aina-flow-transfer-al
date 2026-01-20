#!/usr/bin/env python3
"""
Flow Classification Dataset Creation Script

Creates standardized Flow Classification Datasets from extracted flow features and labels.
Transforms raw flow data into machine learning-ready format with proper filtering and ID mapping.

The script processes datasets through 3 phases:
1. Load and validate data (features + labels)
2. Apply filtering pipeline (remove invalid/rare labels)
3. Create output structure with integer IDs and write files

Output structure:
    dataset/
    ├── config.yaml      # DatasetSplit configuration
    ├── features.csv     # flow_id (integer) + features
    ├── labels.csv       # flow_id (integer) + labels
    └── id2pcap.csv      # flow_id (integer) -> pcap filename mapping
"""

import logging
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple

import click
import pandas as pd

from flc.flows.dataset.split import (
    DatasetSplit,
    SplitType,
    SplitMethod,
    SplitParameters,
    SplitStatistics,
    LabelStatistic,
)
from flc.flows.labels import FlowLabelType
from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.group_labels import FlowGroupLabel, label2group
from flc.flows.labels.label_source import Label, load_labels
from flc.flows.scripts.create_flow_classification_dataset.config import Config, DatasetSpec


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_flow_features(features_file: str) -> pd.DataFrame:
    """
    Load flow features from CSV file.

    Args:
        features_file: Path to CSV file with flow features

    Returns:
        DataFrame with flow_id as index and features as columns

    Raises:
        ValueError: If file doesn't exist or doesn't have flow_id column
    """
    if not os.path.exists(features_file):
        raise ValueError(f"Features file does not exist: {features_file}")

    logging.info(f"Loading flow features from: {features_file}")

    try:
        features_df = pd.read_csv(features_file)

        if "flow_id" not in features_df.columns:
            raise ValueError("Features file must have 'flow_id' column")

        # Set flow_id as index for easier processing
        features_df = features_df.set_index("flow_id")

        logging.info(f"Loaded {len(features_df)} flow features with {len(features_df.columns)} features")
        return features_df

    except Exception as e:
        raise ValueError(f"Error loading features file: {e}")


def load_flow_labels_data(labels_file: str) -> Dict[str, List[Label]]:
    """
    Load flow labels from pickle file.

    Args:
        labels_file: Path to pickle file with flow labels

    Returns:
        Dictionary mapping flow_id to list of Label objects

    Raises:
        ValueError: If file doesn't exist or has invalid format
    """
    if not os.path.exists(labels_file):
        raise ValueError(f"Labels file does not exist: {labels_file}")

    logging.info(f"Loading flow labels from: {labels_file}")

    try:
        labels_dict = load_labels(labels_file)

        # Convert defaultdict to regular dict and validate structure
        labels_dict = dict(labels_dict)

        # Validate that values are lists of Label objects
        for flow_id, labels in labels_dict.items():
            if not isinstance(labels, list):
                raise ValueError(f"Labels for flow_id {flow_id} must be a list")
            for label in labels:
                if not isinstance(label, Label):
                    raise ValueError(f"All labels must be Label objects, got {type(label)}")

        logging.info(f"Loaded labels for {len(labels_dict)} flows")
        return labels_dict

    except Exception as e:
        raise ValueError(f"Error loading labels file: {e}")


def _remove_flows_from_dataset(
    features_df: pd.DataFrame, labels_dict: Dict[str, List[Label]], flows_to_remove: Set[str], step_name: str
) -> Tuple[pd.DataFrame, Dict[str, List[Label]]]:
    """
    Helper function to remove flows from both features and labels.

    Args:
        features_df: DataFrame with flow features
        labels_dict: Dictionary with flow labels
        flows_to_remove: Set of flow_ids to remove
        step_name: Name of the filtering step for logging

    Returns:
        Tuple of (filtered_features, filtered_labels)
    """
    if flows_to_remove:
        features_df = features_df.loc[~features_df.index.isin(flows_to_remove)]
        labels_dict = {flow_id: labels for flow_id, labels in labels_dict.items() if flow_id not in flows_to_remove}
        logging.info(f"After {step_name}: {len(features_df)} flows remaining")

    return features_df, labels_dict


def create_id_mapping(flow_ids: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mapping between string flow_ids (pcap filenames) and integer IDs.

    Args:
        flow_ids: List of string flow_ids (typically pcap filenames)

    Returns:
        Tuple of (str_to_int_map, int_to_str_map)
    """
    # Sort flow_ids for reproducible integer assignment
    sorted_flow_ids = sorted(flow_ids)

    str_to_int = {flow_id: idx for idx, flow_id in enumerate(sorted_flow_ids)}
    int_to_str = {idx: flow_id for idx, flow_id in enumerate(sorted_flow_ids)}

    logging.info(f"Created ID mapping for {len(flow_ids)} flows (integers 0-{len(flow_ids)-1})")
    return str_to_int, int_to_str


def apply_label_filtering(
    features_df: pd.DataFrame, labels_dict: Dict[str, List[Label]], dataset_spec: DatasetSpec, config: Config
) -> Tuple[pd.DataFrame, Dict[str, List[Label]]]:
    """
    Apply filtering pipeline to features and labels.

    Filtering order is optimized for logical flow:
    1. Filter to common flow_ids (intersection)
    2. Remove flows with invalid labels (if enabled) - removes problematic flows first
    3. Remove specific label IDs - modifies labels, may create empty flows
    4. Apply minimum class occurrences - expensive counting operation
    5. Remove flows without labels (if enabled) - final cleanup of empty flows

    Args:
        features_df: DataFrame with flow features
        labels_dict: Dictionary with flow labels
        dataset_spec: Dataset-specific configuration
        config: Global configuration

    Returns:
        Tuple of (filtered_features, filtered_labels)
    """
    logging.info("Starting label filtering pipeline")

    # Get filtering parameters with dataset-specific overrides
    skip_invalid_labels = dataset_spec.get_skip_invalid_labels(config.skip_invalid_labels)
    discard_flows_without_labels = dataset_spec.get_discard_flows_without_labels(config.discard_flows_without_labels)
    discard_label_ids = dataset_spec.get_discard_label_ids(config.discard_label_ids)
    min_class_occurrences = dataset_spec.get_min_class_occurrences(config.min_class_occurrences)

    logging.info(
        f"Filtering parameters: skip_invalid_labels={skip_invalid_labels}, "
        f"discard_flows_without_labels={discard_flows_without_labels}, "
        f"discard_label_ids={discard_label_ids}, "
        f"min_class_occurrences={min_class_occurrences}"
    )

    # Start with intersection of flow_ids between features and labels
    feature_flow_ids = set(features_df.index)
    label_flow_ids = set(labels_dict.keys())
    common_flow_ids = feature_flow_ids.intersection(label_flow_ids)

    logging.info(
        f"Flow ID statistics: {len(feature_flow_ids)} features, "
        f"{len(label_flow_ids)} flows w labels, {len(common_flow_ids)} intersection"
    )

    # Filter to common flow_ids
    filtered_features = features_df.loc[features_df.index.isin(common_flow_ids)].copy()
    filtered_labels = {flow_id: labels for flow_id, labels in labels_dict.items() if flow_id in common_flow_ids}

    logging.info(f"After intersection: {len(filtered_features)} flows remaining")

    # Step 1: Remove flows with invalid labels (remove problematic flows early)
    if skip_invalid_labels:
        invalid_label_classes = FlowLabel.invalid_classes()
        flows_with_invalid_labels = set()
        for flow_id, labels in filtered_labels.items():
            if any(label.label in invalid_label_classes for label in labels):
                flows_with_invalid_labels.add(flow_id)
        logging.info(f"Removing {len(flows_with_invalid_labels)} flows with invalid label classes")
        filtered_features, filtered_labels = _remove_flows_from_dataset(
            filtered_features, filtered_labels, flows_with_invalid_labels, "invalid label removal"
        )

    # Step 2: Remove specific label IDs (modifies labels, may create empty flows)
    if discard_label_ids:
        discard_label_set = set(discard_label_ids)
        labels_removed_count = 0

        for flow_id, labels in filtered_labels.items():
            original_count = len(labels)
            # Filter out labels with discard IDs
            filtered_labels[flow_id] = [label for label in labels if label.flow_label_idx not in discard_label_set]
            labels_removed_count += original_count - len(filtered_labels[flow_id])

        logging.info(f"Removed {labels_removed_count} individual labels with discard label IDs")

    # Step 3: Apply minimum class occurrences filter (expensive counting operation)
    if min_class_occurrences > 1:
        # Count occurrences of each label
        label_counts = Counter()
        for labels in filtered_labels.values():
            for label in labels:
                label_counts[label.flow_label_idx] += 1

        # Find labels to remove
        rare_labels = {label_id for label_id, count in label_counts.items() if count < min_class_occurrences}

        if rare_labels:
            logging.info(f"Removing {len(rare_labels)} rare label classes with < {min_class_occurrences} occurrences")

            # Remove rare labels from flows
            for flow_id, labels in filtered_labels.items():
                valid_labels = [label for label in labels if label.flow_label_idx not in rare_labels]
                filtered_labels[flow_id] = valid_labels

    # Step 4: Final cleanup - remove flows without labels (only if enabled)
    if discard_flows_without_labels:
        flows_without_labels = {flow_id for flow_id, labels in filtered_labels.items() if not labels}
        if flows_without_labels:
            logging.info(f"Removing {len(flows_without_labels)} flows without labels")
            filtered_features, filtered_labels = _remove_flows_from_dataset(
                filtered_features, filtered_labels, flows_without_labels, "flows without labels removal"
            )

    logging.info(f"Final filtering result: {len(filtered_features)} flows remaining")

    return filtered_features, filtered_labels


def generate_group_labels(flow_labels_dict: Dict[str, List[Label]]) -> Dict[str, List[FlowGroupLabel]]:
    """
    Generate group labels from flow labels using label2group mapping.

    Args:
        flow_labels_dict: Dictionary mapping flow_id to list of Label objects

    Returns:
        Dictionary mapping flow_id to list of FlowGroupLabel objects
    """
    logging.info("Generating group labels from flow labels")

    group_labels_dict = {}

    for flow_id, labels in flow_labels_dict.items():
        # Convert each Label to its corresponding FlowGroupLabel
        group_labels = []
        for label in labels:
            try:
                group_label = label2group(label)
                group_labels.append(group_label)
            except Exception as e:
                logging.warning(f"Could not map label {label} to group label for flow {flow_id}: {e}")
                continue

        # Remove duplicates while preserving order
        unique_group_labels = []
        seen = set()
        for group_label in group_labels:
            if group_label not in seen:
                unique_group_labels.append(group_label)
                seen.add(group_label)

        group_labels_dict[flow_id] = unique_group_labels

    logging.info(f"Generated group labels for {len(group_labels_dict)} flows")
    return group_labels_dict


def create_output_directory(output_path: str) -> None:
    """
    Create the output directory for the dataset.

    Args:
        output_path: Root path for the dataset output
    """
    os.makedirs(output_path, exist_ok=True)
    logging.debug(f"Created directory: {output_path}")


def write_features_csv(features_df: pd.DataFrame, id_mapping: Dict[str, int], output_path: str) -> None:
    """
    Write filtered flow features to CSV file with integer flow_ids.

    Args:
        features_df: DataFrame with flow features (string flow_id as index)
        id_mapping: Mapping from string flow_id to integer flow_id
        output_path: Root path for dataset output
    """
    features_file = os.path.join(output_path, "features.csv")

    # Create a copy with integer flow_ids
    features_copy = features_df.copy()

    # Map string flow_ids to integer flow_ids
    features_copy.index = features_copy.index.map(id_mapping)
    features_copy.index.name = "flow_id"

    # Sort by integer flow_id for reproducibility
    features_copy = features_copy.sort_index()

    # Reset index to make flow_id a column
    features_copy = features_copy.reset_index()

    features_copy.to_csv(features_file, index=False)
    logging.info(f"Wrote {len(features_copy)} flow features to {features_file}")


def write_labels_csv(
    labels_dict: Dict[str, List], id_mapping: Dict[str, int], output_path: str, label_type: FlowLabelType
) -> None:
    """
    Write labels to CSV file with integer flow_ids.

    Args:
        labels_dict: Dictionary mapping string flow_id to list of label objects
        id_mapping: Mapping from string flow_id to integer flow_id
        output_path: Root path for dataset output
        label_type: FlowLabelType enum to determine which CSV method to use
    """
    labels_file = os.path.join(output_path, "labels.csv")

    # Convert string flow_ids to integer flow_ids
    int_labels_dict = {}
    for str_flow_id, labels in labels_dict.items():
        int_flow_id = id_mapping[str_flow_id]
        int_labels_dict[int_flow_id] = labels

    # Sort by integer flow_id for reproducibility
    sorted_int_flow_ids = sorted(int_labels_dict.keys())
    sorted_labels_dict = {flow_id: int_labels_dict[flow_id] for flow_id in sorted_int_flow_ids}

    if label_type == FlowLabelType.FLOW:
        FlowLabel.labels_to_csv(sorted_labels_dict, labels_file, zip=False)
    elif label_type == FlowLabelType.FLOW_GROUP:
        FlowGroupLabel.labels_to_csv(sorted_labels_dict, labels_file, zip=False)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    logging.info(f"Wrote {len(sorted_labels_dict)} {label_type.value} labels to {labels_file}")


def write_id2pcap_csv(id_mapping: Dict[int, str], output_path: str) -> None:
    """
    Write id2pcap mapping to CSV file.

    Args:
        id_mapping: Mapping from integer flow_id to string pcap filename
        output_path: Root path for dataset output
    """
    id2pcap_file = os.path.join(output_path, "id2pcap.csv")

    # Create DataFrame with sorted integer IDs
    sorted_ids = sorted(id_mapping.keys())
    id2pcap_df = pd.DataFrame({"flow_id": sorted_ids, "pcap_filename": [id_mapping[flow_id] for flow_id in sorted_ids]})

    id2pcap_df.to_csv(id2pcap_file, index=False)
    logging.info(f"Wrote {len(id2pcap_df)} ID mappings to {id2pcap_file}")


def generate_label_statistics(labels_dict: Dict[str, List], label_type: FlowLabelType) -> Dict:
    """
    Generate statistics for a set of labels.

    Args:
        labels_dict: Dictionary mapping flow_id to list of label objects
        label_type: Type of labels (FlowLabelType.FLOW or FlowLabelType.FLOW_GROUP)

    Returns:
        Dictionary with label statistics
    """

    # Count label occurrences
    label_counts = Counter()
    total_flows = len(labels_dict)
    flows_with_labels = 0

    for flow_id, labels in labels_dict.items():
        if labels:
            flows_with_labels += 1
            for label in labels:
                if hasattr(label, "value"):
                    # This is an enum (FlowGroupLabel)
                    label_counts[label.value] += 1
                elif hasattr(label, "flow_label_idx"):
                    # This is a Label object (for FlowLabel)
                    label_counts[label.flow_label_idx] += 1

    # Convert counts to statistics
    label_stats = {}
    for label_id, count in label_counts.items():
        percentage = (count / total_flows) * 100 if total_flows > 0 else 0

        # Get enum name based on label type parameter
        if label_type == FlowLabelType.FLOW:
            label_name = FlowLabel(label_id).name
        elif label_type == FlowLabelType.FLOW_GROUP:
            label_name = FlowGroupLabel(label_id).name
        else:
            raise ValueError(f"Unknown label_type parameter: {label_type}")

        label_stats[str(label_id)] = {"count": count, "percentage": round(percentage, 2), "name": label_name}

    return {
        "total_flows": total_flows,
        "flows_with_labels": flows_with_labels,
        "flows_without_labels": total_flows - flows_with_labels,
        "unique_labels": len(label_counts),
        "label_statistics": label_stats,
    }


def create_config_yaml(
    dataset_spec: DatasetSpec,
    config: Config,
    labels_dict: Dict[str, List],
    total_flows: int,
    output_path: str,
) -> None:
    """
    Create config.yaml file with DatasetSplit configuration.

    Args:
        dataset_spec: Dataset configuration
        config: Global configuration
        labels_dict: Dictionary with labels for statistics
        total_flows: Total number of flows in dataset
        output_path: Root output path
    """
    logging.info("Creating config.yaml file")

    # Generate statistics for the dataset
    stats_dict = generate_label_statistics(labels_dict, dataset_spec.label_type)
    statistics = convert_statistics_to_split_format(stats_dict, dataset_spec.label_type)

    # Create dataset split configuration
    split_parameters = SplitParameters(split_method=SplitMethod.MANUAL, parent_dataset_path=output_path)

    dataset_split = DatasetSplit.create(
        dataset_name=dataset_spec.name,
        split_name="full",
        split_type=SplitType.FULL,
        label_type=dataset_spec.label_type,
        flow_index_csv_path="",  # Not used in new structure
        statistics=statistics,
        split_parameters=split_parameters,
    )

    # Save config.yaml
    config_file = os.path.join(output_path, "config.yaml")
    dataset_split.to_yaml(config_file)
    logging.info(f"Created config.yaml: {config_file}")


def convert_statistics_to_split_format(stats_dict: Dict, label_type: FlowLabelType) -> SplitStatistics:
    """
    Convert existing statistics dictionary format to SplitStatistics format.

    Args:
        stats_dict: Statistics dictionary from generate_label_statistics()
        label_type: Type of labels for validation

    Returns:
        SplitStatistics object
    """
    # Convert label statistics to LabelStatistic objects
    label_statistics = {}
    for label_id, stats in stats_dict.get("label_statistics", {}).items():
        label_statistics[label_id] = LabelStatistic(
            count=stats["count"], percentage=stats["percentage"], name=stats["name"]
        )

    return SplitStatistics(
        total_flows=stats_dict.get("total_flows", 0),
        flows_with_labels=stats_dict.get("flows_with_labels", 0),
        flows_without_labels=stats_dict.get("flows_without_labels", 0),
        unique_labels=stats_dict.get("unique_labels", 0),
        label_statistics=label_statistics,
    )


def process_dataset(dataset_spec: DatasetSpec, config: Config) -> bool:
    """
    Process a single dataset according to the specification.

    Args:
        dataset_spec: Dataset configuration
        config: Global configuration

    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Processing dataset: {dataset_spec.name}")

    try:
        # Phase 1: Load and validate data
        features_df = load_flow_features(dataset_spec.features_file)
        labels_dict = load_flow_labels_data(dataset_spec.flow_labels_file)

        # Phase 2: Apply filtering pipeline
        filtered_features, filtered_labels = apply_label_filtering(features_df, labels_dict, dataset_spec, config)

        if len(filtered_features) == 0:
            logging.warning(f"No flows remaining after filtering for dataset {dataset_spec.name}")
            return False

        # Phase 3: Create output structure with integer IDs and write files
        create_output_directory(dataset_spec.output_path)

        # Create ID mapping for all flows
        flow_ids = list(filtered_features.index)
        str_to_int_mapping, int_to_str_mapping = create_id_mapping(flow_ids)

        # Generate labels based on label_type
        if dataset_spec.label_type == FlowLabelType.FLOW:
            # Use flow labels directly
            final_labels_dict = {
                flow_id: [label.label for label in labels] for flow_id, labels in filtered_labels.items()
            }
        elif dataset_spec.label_type == FlowLabelType.FLOW_GROUP:
            # Generate group labels from flow labels
            group_labels_dict = generate_group_labels(filtered_labels)
            final_labels_dict = group_labels_dict
        else:
            raise ValueError(f"Unknown label_type: {dataset_spec.label_type}")

        # Write all output files
        write_features_csv(filtered_features, str_to_int_mapping, dataset_spec.output_path)
        write_labels_csv(final_labels_dict, str_to_int_mapping, dataset_spec.output_path, dataset_spec.label_type)
        write_id2pcap_csv(int_to_str_mapping, dataset_spec.output_path)
        create_config_yaml(dataset_spec, config, final_labels_dict, len(filtered_features), dataset_spec.output_path)

        logging.info(f"Successfully processed dataset {dataset_spec.name} -> {dataset_spec.output_path}")
        return True

    except Exception as e:
        logging.error(f"Error processing dataset {dataset_spec.name}: {e}")
        return False


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(config: str, verbose: bool):
    """
    Create Flow Classification Datasets from extracted features and labels.

    This script transforms raw flow data into machine learning-ready format
    with proper filtering and metadata generation.

    Examples:

        python run.py --config config.yaml

        python run.py --config config.yaml --verbose
    """
    # Setup logging
    setup_logging(verbose)

    try:
        # Load configuration
        logging.info(f"Loading configuration from: {config}")
        config_obj = Config.from_yaml(config)

        # Process each enabled dataset
        successful_datasets = 0
        failed_datasets = 0

        for dataset_spec in config_obj.datasets:
            if not dataset_spec.enabled:
                logging.info(f"Skipping disabled dataset: {dataset_spec.name}")
                continue

            if process_dataset(dataset_spec, config_obj):
                successful_datasets += 1
            else:
                failed_datasets += 1

        # Summary
        logging.info(f"Processing complete: {successful_datasets} successful, {failed_datasets} failed")

        if failed_datasets > 0:
            sys.exit(1)

    except Exception as e:
        logging.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
