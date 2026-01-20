import os
from typing import List

from flc.flows.scripts.clustering_transfer_al_sweeps.config import DatasetTripletConfig
from flc.flows.scripts.clustering_transfer_al_sweeps.sweep_generator import ClusteringALCombination


def create_output_directory(
    base_output_dir: str,
    dataset_triplet_idx: int,
    combination: ClusteringALCombination,
) -> str:
    """
    Create unique output directory for each parameter combination and dataset triplet.

    Args:
        base_output_dir: Base output directory
        dataset_triplet_idx: Index of dataset triplet
        combination: Parameter combination for AL execution

    Returns:
        Full path to created output directory

    Raises:
        ValueError: If directory cannot be created
    """
    # Generate directory structure
    triplet_dir = f"dataset_triplet_{dataset_triplet_idx}"
    combination_dir = generate_directory_name(combination)

    full_output_dir = os.path.join(base_output_dir, triplet_dir, combination_dir)

    # Create directory
    try:
        os.makedirs(full_output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output directory {full_output_dir}: {e}") from e

    # Validate directory is writable
    validate_output_path(full_output_dir)

    return full_output_dir


def generate_directory_name(combination: ClusteringALCombination) -> str:
    """
    Generate structured directory name for parameter combination.

    Args:
        combination: Parameter combination for AL execution

    Returns:
        Directory name following naming convention
    """
    return combination.combination_id


def validate_output_path(output_path: str) -> None:
    """
    Ensure output directory can be created and is writable.

    Args:
        output_path: Path to output directory

    Raises:
        ValueError: If directory cannot be used for output
    """
    # Check if path exists
    if not os.path.exists(output_path):
        raise ValueError(f"Output directory does not exist: {output_path}")

    # Check if path is a directory
    if not os.path.isdir(output_path):
        raise ValueError(f"Output path is not a directory: {output_path}")

    # Check if directory is writable
    if not os.access(output_path, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_path}")


def get_output_directory_structure(
    base_output_dir: str,
    dataset_triplets: List[DatasetTripletConfig],
    combinations: List[ClusteringALCombination],
) -> dict:
    """
    Get preview of output directory structure without creating directories.

    Args:
        base_output_dir: Base output directory
        dataset_triplets: List of dataset triplet configurations
        combinations: List of parameter combinations

    Returns:
        Dictionary describing the directory structure
    """
    structure = {
        "base_directory": base_output_dir,
        "dataset_triplets": {},
        "total_directories": 0,
    }

    for triplet_idx, triplet in enumerate(dataset_triplets):
        triplet_name = f"dataset_triplet_{triplet_idx}"
        triplet_info = {
            "source_split": triplet.source_split_path,
            "target_split": triplet.target_split_path,
            "test_split": triplet.test_split_path,
            "combinations": {},
        }

        for combination in combinations:
            combination_dir = generate_directory_name(combination)
            full_path = os.path.join(base_output_dir, triplet_name, combination_dir)

            triplet_info["combinations"][combination_dir] = {
                "full_path": full_path,
                "combination_id": combination.combination_id,
                "query_strategy": combination.query_strategy,
                "clustering_algorithm": combination.clustering_algorithm,
                "classifier_name": combination.classifier_name,
                "random_state": combination.random_state,
            }

            structure["total_directories"] += 1

        structure["dataset_triplets"][triplet_name] = triplet_info

    return structure


def create_base_output_directories(
    base_output_dir: str,
    dataset_triplets: List[DatasetTripletConfig],
) -> None:
    """
    Create base output directories for all dataset triplets.

    Args:
        base_output_dir: Base output directory
        dataset_triplets: List of dataset triplet configurations

    Raises:
        ValueError: If directories cannot be created
    """
    # Create base directory
    try:
        os.makedirs(base_output_dir, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create base output directory {base_output_dir}: {e}") from e

    # Create triplet directories
    for triplet_idx, _ in enumerate(dataset_triplets):
        triplet_dir = os.path.join(base_output_dir, f"dataset_triplet_{triplet_idx}")
        try:
            os.makedirs(triplet_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create triplet directory {triplet_dir}: {e}") from e

    # Validate base directory is writable
    validate_output_path(base_output_dir)


def cleanup_empty_directories(base_output_dir: str) -> None:
    """
    Remove empty directories from output structure.

    Args:
        base_output_dir: Base output directory to clean
    """
    if not os.path.exists(base_output_dir):
        return

    # Walk directory tree bottom-up to remove empty directories
    for root, dirs, files in os.walk(base_output_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Try to remove directory if empty
                os.rmdir(dir_path)
            except OSError:
                # Directory not empty, skip
                pass


def get_existing_results(
    base_output_dir: str,
    dataset_triplets: List[DatasetTripletConfig],
) -> dict:
    """
    Scan existing output directories for completed results.

    Args:
        base_output_dir: Base output directory
        dataset_triplets: List of dataset triplet configurations

    Returns:
        Dictionary mapping triplet_idx -> combination_id -> result_info
    """
    existing_results = {}

    if not os.path.exists(base_output_dir):
        return existing_results

    for triplet_idx, _ in enumerate(dataset_triplets):
        triplet_name = f"dataset_triplet_{triplet_idx}"
        triplet_dir = os.path.join(base_output_dir, triplet_name)

        if not os.path.exists(triplet_dir):
            continue

        triplet_results = {}

        # Scan combination directories
        for item in os.listdir(triplet_dir):
            combination_dir = os.path.join(triplet_dir, item)

            if not os.path.isdir(combination_dir):
                continue

            # Check for results files
            results_file = os.path.join(combination_dir, "results.json")
            config_file = os.path.join(combination_dir, "config.yaml")

            result_info = {
                "directory": combination_dir,
                "has_results": os.path.exists(results_file),
                "has_config": os.path.exists(config_file),
                "is_complete": os.path.exists(results_file) and os.path.exists(config_file),
            }

            if result_info["has_results"]:
                try:
                    # Get file modification time as completion indicator
                    result_info["completion_time"] = os.path.getmtime(results_file)
                except OSError:
                    result_info["completion_time"] = None

            triplet_results[item] = result_info

        if triplet_results:
            existing_results[triplet_idx] = triplet_results

    return existing_results
