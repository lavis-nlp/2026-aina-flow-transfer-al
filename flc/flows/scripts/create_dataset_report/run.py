#!/usr/bin/env python3
"""
Dataset Report Creation Script

Creates CSV reports summarizing flow classification datasets from split YAML files.
Processes multiple dataset splits and generates comprehensive statistics including
flow counts, label distributions, and per-label occurrence counts.

The script processes split files through the following phases:
1. Load and validate configuration
2. Extract label schema from FlowLabel or FlowGroupLabel enum
3. Process each split file using FlowClassificationDataset
4. Extract metadata and statistics from each dataset
5. Generate comprehensive CSV report with per-label counts

Output CSV structure:
- Core columns: dataset_name, split, number_of_flows, etc.
- Dynamic columns: One column per label type with occurrence counts
"""

import logging
import sys
from typing import Dict, List, Any

import click
import pandas as pd

from flc.flows.dataset.dataset import FlowClassificationDataset
from flc.flows.labels import FlowLabelType
from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.group_labels import FlowGroupLabel
from flc.flows.scripts.create_dataset_report.config import Config


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def get_label_schema(label_type: FlowLabelType) -> List[str]:
    """
    Get all possible label names for the specified label type.
    
    Args:
        label_type: Type of labels (FlowLabelType.FLOW or FlowLabelType.FLOW_GROUP)
        
    Returns:
        List of label names sorted by their enum values
    """
    if label_type == FlowLabelType.FLOW:
        return [label.name for label in sorted(FlowLabel, key=lambda x: x.value)]
    elif label_type == FlowLabelType.FLOW_GROUP:
        return [label.name for label in sorted(FlowGroupLabel, key=lambda x: x.value)]
    else:
        raise ValueError(f"Unsupported label type: {label_type}")


def process_split_file(split_file_path: str, expected_label_type: FlowLabelType, all_label_names: List[str]) -> Dict[str, Any]:
    """
    Process a single split file and extract comprehensive statistics.
    
    Args:
        split_file_path: Path to the split YAML file
        expected_label_type: Expected label type for validation
        all_label_names: List of all possible label names for this label type
        
    Returns:
        Dictionary with dataset statistics and per-label counts
        
    Raises:
        ValueError: If split file has mismatched label type
    """
    logging.info(f"Processing split file: {split_file_path}")
    
    try:
        # Load dataset from split file
        dataset = FlowClassificationDataset(split_file_path)
        
        # Validate label type matches expected
        if dataset.label_type != expected_label_type:
            raise ValueError(
                f"Label type mismatch in {split_file_path}: expected {expected_label_type.value}, "
                f"got {dataset.label_type.value}"
            )
        
        # Extract metadata from dataset split
        dataset_name = dataset.split.dataset_name
        split_name = dataset.split.split_name
        split_type = dataset.split.split_type.value
        
        # Get dataset statistics
        stats = dataset.get_statistics()
        
        # Get label distribution (label_name -> count)
        label_distribution = dataset.get_label_distribution()
        
        # Create result dictionary with core statistics
        result = {
            "dataset_name": dataset_name,
            "split": split_name,  # Use split_name instead of split_type for more descriptive output
            "label_type": dataset.label_type.value,
            "number_of_flows": stats["total_flows"],
            "number_of_labeled_flows": stats["flows_with_labels"],
            "number_of_unlabeled_flows": stats["flows_without_labels"],
            "number_of_unique_labels": stats["unique_labels"],
        }
        
        # Add per-label counts with formatted strings
        total_flows = stats["total_flows"]
        for label_name in all_label_names:
            count = label_distribution.get(label_name, 0)
            if count == 0:
                result[label_name] = "-"
            else:
                percentage = (count / total_flows * 100) if total_flows > 0 else 0.0
                result[label_name] = f"{count} ({percentage:.3f}%)"
        
        logging.info(
            f"Processed {dataset_name}/{split_name}: {stats['total_flows']} flows, "
            f"{stats['unique_labels']} unique labels"
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error processing split file {split_file_path}: {e}")
        raise


def generate_report(config: Config) -> pd.DataFrame:
    """
    Generate comprehensive dataset report from configuration.
    
    Args:
        config: Configuration object with split files and settings
        
    Returns:
        DataFrame with dataset report
    """
    logging.info("Starting dataset report generation")
    
    # Get label schema for the specified label type
    label_type = config.flow_label_type
    all_label_names = get_label_schema(label_type)
    
    logging.info(f"Label type: {label_type.value}")
    logging.info(f"Total labels in schema: {len(all_label_names)}")
    
    # Process each split file
    report_data = []
    successful_files = 0
    failed_files = 0
    
    for split_file in config.split_files:
        try:
            result = process_split_file(split_file, label_type, all_label_names)
            report_data.append(result)
            successful_files += 1
        except Exception as e:
            logging.error(f"Failed to process {split_file}: {e}")
            failed_files += 1
            continue
    
    if not report_data:
        raise ValueError("No split files were successfully processed")
    
    logging.info(f"Processing complete: {successful_files} successful, {failed_files} failed")
    
    # Create DataFrame from report data
    df = pd.DataFrame(report_data)
    
    # Define column order: core columns first, then label columns
    core_columns = [
        "dataset_name", 
        "split", 
        "label_type",
        "number_of_flows", 
        "number_of_labeled_flows", 
        "number_of_unlabeled_flows", 
        "number_of_unique_labels"
    ]
    
    # Reorder columns: core columns + sorted label columns
    column_order = core_columns + all_label_names
    df = df[column_order]
    
    # Sort by dataset_name and split for consistent output
    df = df.sort_values(["dataset_name", "split"]).reset_index(drop=True)
    
    return df


def save_report(df: pd.DataFrame, output_file: str) -> None:
    """
    Save report DataFrame to CSV file.
    
    Args:
        df: Report DataFrame
        output_file: Path to output CSV file
    """
    logging.info(f"Saving report to: {output_file}")
    
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Report saved successfully: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        logging.error(f"Error saving report: {e}")
        raise


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
    Create comprehensive dataset reports from flow classification split files.
    
    This script processes multiple dataset split YAML files and generates
    a detailed CSV report with flow statistics and per-label occurrence counts.
    
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
        
        logging.info(f"Configuration loaded: {len(config_obj.split_files)} split files to process")
        
        # Generate report
        report_df = generate_report(config_obj)
        
        # Save report
        save_report(report_df, config_obj.output_file)
        
        # Summary
        logging.info("Dataset report generation completed successfully")
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()