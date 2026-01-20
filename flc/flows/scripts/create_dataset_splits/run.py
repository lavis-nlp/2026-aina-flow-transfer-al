#!/usr/bin/env python3
"""
Dataset Splits Creation Script

Creates train/test/validation splits from existing flow classification datasets.
Loads an existing dataset and randomly splits flows according to specified percentages.
Each split becomes an independent complete dataset with the new structure.

The script processes datasets through the following phases:
1. Load existing dataset from new simplified structure (config.yaml, features.csv, labels.csv, id2pcap.csv)
2. Read complete dataset data from CSV files
3. Randomly partition flows according to split specifications
4. Create complete independent datasets for each split
5. Save each split as a complete dataset directory with all 4 files

Usage:
    python run.py --config config.yaml [--verbose]
"""

import logging
import os
import random
import sys
from collections import Counter
from typing import Dict, List

import click
import pandas as pd

from flc.flows.dataset.split import (
    DatasetSplit,
    SplitParameters,
    SplitMethod,
    SplitStatistics,
    LabelStatistic,
)
from flc.flows.labels import FlowLabelType
from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.group_labels import FlowGroupLabel
from flc.flows.scripts.create_dataset_splits.config import Config, DatasetSplitSpec


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_existing_dataset_split(dataset_path: str) -> DatasetSplit:
    """
    Load dataset split configuration from config.yaml.

    Args:
        dataset_path: Path to the dataset root directory

    Returns:
        DatasetSplit object with dataset metadata

    Raises:
        ValueError: If config file doesn't exist or is invalid
    """
    config_file = os.path.join(dataset_path, "config.yaml")

    if not os.path.exists(config_file):
        raise ValueError(f"Config file does not exist: {config_file}")

    try:
        dataset_split = DatasetSplit.from_yaml(config_file)
        logging.info(f"Loaded dataset configuration from: {config_file}")
        return dataset_split

    except Exception as e:
        raise ValueError(f"Error loading config file: {e}")


def load_dataset_data(dataset_path: str, label_type: FlowLabelType) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load complete dataset data from CSV files.

    Args:
        dataset_path: Path to the dataset root directory
        label_type: Type of labels (flow or group) to determine correct loading function

    Returns:
        Tuple of (features_df, labels_df, id2pcap_df)

    Raises:
        ValueError: If required files don't exist or are invalid
    """
    # Load features CSV
    features_file = os.path.join(dataset_path, "features.csv")
    if not os.path.exists(features_file):
        raise ValueError(f"Features file does not exist: {features_file}")

    # Load labels CSV
    labels_file = os.path.join(dataset_path, "labels.csv")
    if not os.path.exists(labels_file):
        raise ValueError(f"Labels file does not exist: {labels_file}")

    # Load id2pcap CSV
    id2pcap_file = os.path.join(dataset_path, "id2pcap.csv")
    if not os.path.exists(id2pcap_file):
        raise ValueError(f"id2pcap file does not exist: {id2pcap_file}")

    try:
        # Load features data
        features_df = pd.read_csv(features_file)

        # Load labels using appropriate function based on label type
        if label_type == FlowLabelType.FLOW:
            labels_df = FlowLabel.labels_from_csv(labels_file, add_enum_col=False)
        elif label_type == FlowLabelType.FLOW_GROUP:
            labels_df = FlowGroupLabel.labels_from_csv(labels_file, add_enum_col=False)
        else:
            raise ValueError(f"Unsupported label type: {label_type}")

        # Load id2pcap mapping
        id2pcap_df = pd.read_csv(id2pcap_file)

        # Verify flow_id consistency
        features_flow_ids = set(features_df["flow_id"])
        labels_flow_ids = set(labels_df["flow_id"])
        id2pcap_flow_ids = set(id2pcap_df["flow_id"])

        if features_flow_ids != labels_flow_ids:
            raise ValueError("Features and labels files have different flow_ids")

        if features_flow_ids != id2pcap_flow_ids:
            raise ValueError("Features and id2pcap files have different flow_ids")

        logging.info(f"Loaded dataset with {len(features_df)} flows")
        return features_df, labels_df, id2pcap_df

    except Exception as e:
        raise ValueError(f"Error loading dataset files: {e}")


def create_data_splits(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id2pcap_df: pd.DataFrame,
    dataset_spec: DatasetSplitSpec,
    global_config: Config,
) -> Dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Randomly split dataset data according to split specifications.

    Args:
        features_df: DataFrame with features data
        labels_df: DataFrame with labels data
        id2pcap_df: DataFrame with id2pcap mapping
        dataset_spec: Dataset specification with split details
        global_config: Global configuration with defaults

    Returns:
        Dictionary mapping split name to tuple of (features_df, labels_df, id2pcap_df) subsets
    """
    random_seed = dataset_spec.get_random_seed(global_config.random_seed)
    splits = dataset_spec.get_splits(global_config.splits)

    logging.info(f"Creating splits from {len(features_df)} flows (seed: {random_seed})")

    # Get flow_ids and shuffle them
    flow_ids = features_df["flow_id"].tolist()
    random.Random(random_seed).shuffle(flow_ids)

    # Calculate split sizes
    enabled_splits = [split for split in splits if split.enabled]
    split_data = {}
    current_idx = 0

    for i, split_spec in enumerate(enabled_splits):
        if i == len(enabled_splits) - 1:
            # Last split gets all remaining flows to handle rounding
            split_flow_ids = flow_ids[current_idx:]
        else:
            split_size = int(len(flow_ids) * split_spec.percentage)
            split_flow_ids = flow_ids[current_idx : current_idx + split_size]
            current_idx += split_size

        # Filter data for this split
        split_features = features_df[features_df["flow_id"].isin(split_flow_ids)].copy()
        split_labels = labels_df[labels_df["flow_id"].isin(split_flow_ids)].copy()
        split_id2pcap = id2pcap_df[id2pcap_df["flow_id"].isin(split_flow_ids)].copy()

        split_data[split_spec.name] = (split_features, split_labels, split_id2pcap)

        # Log split sizes
        percentage = (len(split_flow_ids) / len(flow_ids)) * 100
        logging.info(f"Split '{split_spec.name}': {len(split_flow_ids)} flows ({percentage:.1f}%)")

    return split_data


def generate_split_statistics(labels_df: pd.DataFrame, label_type: FlowLabelType) -> SplitStatistics:
    """
    Generate statistics for a specific split.

    Args:
        labels_df: DataFrame with flow_id and label_idxs for this split
        label_type: Type of labels (flow or group)

    Returns:
        SplitStatistics object with computed statistics
    """
    # Count label occurrences
    label_counts = Counter()
    total_flows = len(labels_df)
    flows_with_labels = 0

    for _, row in labels_df.iterrows():
        labels = row["label_idxs"]
        if labels and len(labels) > 0:
            flows_with_labels += 1
            for label_id in labels:
                label_counts[label_id] += 1

    # Convert counts to LabelStatistic objects
    label_statistics = {}
    for label_id, count in label_counts.items():
        percentage = (count / total_flows) * 100 if total_flows > 0 else 0

        # Get enum name based on label type
        if label_type == FlowLabelType.FLOW:
            label_name = FlowLabel(label_id).name
        elif label_type == FlowLabelType.FLOW_GROUP:
            label_name = FlowGroupLabel(label_id).name

        label_statistics[str(label_id)] = LabelStatistic(count=count, percentage=round(percentage, 2), name=label_name)

    return SplitStatistics(
        total_flows=total_flows,
        flows_with_labels=flows_with_labels,
        flows_without_labels=total_flows - flows_with_labels,
        unique_labels=len(label_counts),
        label_statistics=label_statistics,
    )


def create_and_save_splits(
    split_data_dict: Dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    dataset_split: DatasetSplit,
    dataset_spec: DatasetSplitSpec,
    global_config: Config,
) -> None:
    """
    Create complete independent dataset directories for each split.

    Args:
        split_data_dict: Dictionary mapping split name to (features_df, labels_df, id2pcap_df) tuples
        dataset_split: Original dataset split configuration
        dataset_spec: Dataset specification with output settings
        global_config: Global configuration with defaults
    """
    logging.info("Creating and saving independent dataset splits")

    # Get output directory and create it
    output_directory = dataset_spec.get_output_directory()
    os.makedirs(output_directory, exist_ok=True)

    # Get enabled splits with their specifications
    splits = dataset_spec.get_splits(global_config.splits)
    enabled_splits = [split for split in splits if split.enabled]
    split_specs_by_name = {split.name: split for split in enabled_splits}

    for split_name, (features_df, labels_df, id2pcap_df) in split_data_dict.items():
        split_spec = split_specs_by_name[split_name]

        # Create directory for this split
        split_directory = os.path.join(output_directory, split_name)
        os.makedirs(split_directory, exist_ok=True)

        # Save features.csv
        features_file = os.path.join(split_directory, "features.csv")
        features_df.to_csv(features_file, index=False)
        logging.info(f"Saved features for split '{split_name}' to {features_file} ({len(features_df)} flows)")

        # Save labels.csv
        labels_file = os.path.join(split_directory, "labels.csv")
        labels_df.to_csv(labels_file, index=False)
        logging.info(f"Saved labels for split '{split_name}' to {labels_file} ({len(labels_df)} flows)")

        # Save id2pcap.csv
        id2pcap_file = os.path.join(split_directory, "id2pcap.csv")
        id2pcap_df.to_csv(id2pcap_file, index=False)
        logging.info(f"Saved id2pcap mapping for split '{split_name}' to {id2pcap_file} ({len(id2pcap_df)} mappings)")

        # Generate statistics for this split
        split_statistics = generate_split_statistics(labels_df, dataset_spec.label_type)

        # Create split parameters
        random_seed = dataset_spec.get_random_seed(global_config.random_seed)
        split_parameters = SplitParameters(
            split_method=SplitMethod.RANDOM,
            split_ratio=split_spec.percentage,
            random_seed=random_seed,
            parent_dataset_path=dataset_spec.dataset_path,
        )

        # Create DatasetSplit object for this split
        split_dataset_split = DatasetSplit.create(
            dataset_name=dataset_split.dataset_name,
            split_name=split_name,
            split_type=split_spec.split_type,
            label_type=dataset_spec.label_type,
            flow_index_csv_path="",  # Not used in new structure
            statistics=split_statistics,
            split_parameters=split_parameters,
        )

        # Save config.yaml
        config_file = os.path.join(split_directory, "config.yaml")
        split_dataset_split.to_yaml(config_file)
        logging.info(f"Saved config for split '{split_name}' to {config_file}")


def process_single_dataset(dataset_spec: DatasetSplitSpec, global_config: Config) -> bool:
    """
    Process dataset splits for a single dataset.

    Args:
        dataset_spec: Dataset specification for split creation
        global_config: Global configuration with defaults

    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Processing dataset splits for: {dataset_spec.dataset_path}")

    try:
        # Load existing dataset configuration
        dataset_split = load_existing_dataset_split(dataset_spec.dataset_path)

        # Load complete dataset data
        features_df, labels_df, id2pcap_df = load_dataset_data(dataset_spec.dataset_path, dataset_spec.label_type)

        if len(features_df) == 0:
            logging.warning("No flows found in dataset")
            return False

        # Create splits
        split_data_dict = create_data_splits(features_df, labels_df, id2pcap_df, dataset_spec, global_config)

        # Create and save complete independent datasets for each split
        create_and_save_splits(split_data_dict, dataset_split, dataset_spec, global_config)

        logging.info(f"Successfully created {len(split_data_dict)} independent split datasets")
        return True

    except Exception as e:
        logging.error(f"Error processing dataset splits: {e}")
        return False


def process_multiple_datasets(config: Config) -> bool:
    """
    Process dataset splits for multiple datasets according to configuration.

    Args:
        config: Configuration with multiple datasets

    Returns:
        True if all datasets processed successfully, False otherwise
    """
    enabled_datasets = config.get_enabled_datasets()
    if not enabled_datasets:
        logging.error("No enabled datasets found")
        return False

    logging.info(f"Processing {len(enabled_datasets)} datasets")

    success_count = 0
    for dataset_spec in enabled_datasets:
        logging.info(f"Processing dataset '{dataset_spec.name}'")

        if process_single_dataset(dataset_spec, config):
            success_count += 1
            logging.info(f"Dataset '{dataset_spec.name}' processed successfully")
        else:
            logging.error(f"Failed to process dataset '{dataset_spec.name}'")

    logging.info(f"Completed processing: {success_count}/{len(enabled_datasets)} datasets successful")
    return success_count == len(enabled_datasets)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(config: str, verbose: bool):
    """
    Create Dataset Splits from existing Flow Classification Datasets.

    This script randomly splits flows from an existing dataset into
    train/test/validation splits according to specified percentages.

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

        # Validate configuration
        config_obj.validate()

        # Process dataset splits
        if process_multiple_datasets(config_obj):
            logging.info("Dataset splits creation completed successfully")
        else:
            logging.error("Dataset splits creation failed")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
