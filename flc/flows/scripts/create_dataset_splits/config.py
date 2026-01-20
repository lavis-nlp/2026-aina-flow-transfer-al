import os
from typing import List, Optional
import yaml
from pydantic.dataclasses import dataclass

from flc.flows.dataset.split import SplitType
from flc.flows.labels import FlowLabelType


@dataclass(frozen=True)
class SplitSpec:
    """Specification for a single dataset split to be created."""

    name: str
    split_type: SplitType
    percentage: float
    enabled: bool = True

    def __post_init__(self):
        """Validate split specification parameters."""
        if not 0.0 < self.percentage <= 1.0:
            raise ValueError(f"Split percentage must be between 0.0 and 1.0, got {self.percentage}")


@dataclass(frozen=True)
class DatasetSplitSpec:
    """Specification for a single dataset to create splits for."""

    name: str
    dataset_path: str
    label_type: FlowLabelType
    output_directory: Optional[str] = None
    splits: Optional[List[SplitSpec]] = None
    random_seed: Optional[int] = None
    enabled: bool = True

    def get_output_directory(self) -> str:
        """Get output directory, defaulting to dataset_path parent with splits suffix if not specified."""
        if self.output_directory is not None:
            return self.output_directory

        # Default to creating splits directory alongside the dataset
        dataset_parent = os.path.dirname(self.dataset_path)
        dataset_name = os.path.basename(self.dataset_path)
        splits_dir = os.path.join(dataset_parent, f"{dataset_name}_splits")

        return str(splits_dir)

    def get_splits(self, global_default: List[SplitSpec]) -> List[SplitSpec]:
        """Get splits, using global default if not specified."""
        return self.splits if self.splits is not None else global_default

    def get_random_seed(self, global_default: int) -> int:
        """Get random_seed, using global default if not specified."""
        return self.random_seed if self.random_seed is not None else global_default


@dataclass(frozen=True)
class Config:
    """Configuration for creating dataset splits from multiple datasets."""

    datasets: List[DatasetSplitSpec]

    # Global defaults (can be overridden per dataset)
    splits: List[SplitSpec] = None
    random_seed: int = 42

    def __post_init__(self):
        if self.splits is None:
            object.__setattr__(self, "splits", [])

    def _validate_splits(self, splits: List[SplitSpec]) -> None:
        """Validate that splits are configured correctly."""
        enabled_splits = [split for split in splits if split.enabled]
        if not enabled_splits:
            return

        total_percentage = sum(split.percentage for split in enabled_splits)
        if abs(total_percentage - 1.0) > 1e-6:
            raise ValueError(f"Split percentages must sum to 1.0, got {total_percentage}")

        split_names = [split.name for split in enabled_splits]
        if len(split_names) != len(set(split_names)):
            raise ValueError("Split names must be unique")

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse global splits if specified
        global_splits = []
        if "splits" in config_dict:
            splits_dict = config_dict["splits"]
            for split_dict in splits_dict:
                if len(split_dict.keys()) != 1:
                    raise ValueError("Expected split entries to consist of exactly one key")

                split_name = list(split_dict.keys())[0]
                split_config = split_dict[split_name]

                required_fields = ["split_type", "percentage"]
                for field in required_fields:
                    if field not in split_config:
                        raise ValueError(f"Split '{split_name}' missing required '{field}'")

                split_spec = SplitSpec(
                    name=split_name,
                    split_type=SplitType(split_config["split_type"]),
                    percentage=split_config["percentage"],
                    enabled=split_config.get("enabled", True),
                )
                global_splits.append(split_spec)

        # Parse datasets
        datasets = []
        for dataset_dict in config_dict.get("datasets", []):
            if len(dataset_dict.keys()) != 1:
                raise ValueError("Expected dataset entries to consist of exactly one key")

            dataset_name = list(dataset_dict.keys())[0]
            dataset_config = dataset_dict[dataset_name]

            # Validate required fields
            required_fields = ["dataset_path", "label_type"]
            for field in required_fields:
                if field not in dataset_config:
                    raise ValueError(f"Dataset '{dataset_name}' missing required '{field}'")

            # Convert relative paths to absolute
            dataset_path = dataset_config["dataset_path"]
            if not os.path.isabs(dataset_path):
                dataset_path = os.path.abspath(dataset_path)

            output_directory = None
            if "output_directory" in dataset_config:
                output_directory = dataset_config["output_directory"]
                if not os.path.isabs(output_directory):
                    output_directory = os.path.abspath(output_directory)

            # Parse dataset-specific splits if specified
            dataset_splits = None
            if "splits" in dataset_config:
                dataset_splits = []
                splits_dict = dataset_config["splits"]
                for split_dict in splits_dict:
                    if len(split_dict.keys()) != 1:
                        raise ValueError("Expected split entries to consist of exactly one key")

                    split_name = list(split_dict.keys())[0]
                    split_config = split_dict[split_name]

                    required_fields = ["split_type", "percentage"]
                    for field in required_fields:
                        if field not in split_config:
                            raise ValueError(f"Split '{split_name}' missing required '{field}'")

                    split_spec = SplitSpec(
                        name=split_name,
                        split_type=SplitType(split_config["split_type"]),
                        percentage=split_config["percentage"],
                        enabled=split_config.get("enabled", True),
                    )
                    dataset_splits.append(split_spec)

            dataset_spec = DatasetSplitSpec(
                name=dataset_name,
                dataset_path=dataset_path,
                label_type=FlowLabelType(dataset_config["label_type"]),
                output_directory=output_directory,
                splits=dataset_splits,
                random_seed=dataset_config.get("random_seed"),
                enabled=dataset_config.get("enabled", True),
            )
            datasets.append(dataset_spec)

        config = cls(
            datasets=datasets,
            splits=global_splits,
            random_seed=config_dict.get("random_seed", 42),
        )

        # Validate splits for each dataset
        for dataset in datasets:
            if dataset.enabled:
                splits_to_validate = dataset.get_splits(config.splits)
                config._validate_splits(splits_to_validate)

        return config

    def get_enabled_datasets(self) -> List[DatasetSplitSpec]:
        """Get only enabled datasets."""
        return [dataset for dataset in self.datasets if dataset.enabled]

    def validate(self) -> None:
        """Validate configuration against file system."""
        enabled_datasets = self.get_enabled_datasets()
        if not enabled_datasets:
            raise ValueError("No datasets are enabled")

        for dataset in enabled_datasets:
            # Check dataset path exists
            if not os.path.exists(dataset.dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset.dataset_path}")

            # Check that the dataset has the expected new structure
            config_file = os.path.join(dataset.dataset_path, "config.yaml")
            if not os.path.exists(config_file):
                raise ValueError(f"Expected config.yaml file does not exist: {config_file}")

            features_file = os.path.join(dataset.dataset_path, "features.csv")
            if not os.path.exists(features_file):
                raise ValueError(f"Expected features.csv file does not exist: {features_file}")

            labels_file = os.path.join(dataset.dataset_path, "labels.csv")
            if not os.path.exists(labels_file):
                raise ValueError(f"Expected labels.csv file does not exist: {labels_file}")

            # Check output directory parent exists
            output_directory = dataset.get_output_directory()
            output_parent = os.path.dirname(output_directory)
            if output_parent and not os.path.exists(output_parent):
                raise ValueError(f"Output directory parent does not exist: {output_parent}")

            # Check at least one split is enabled for this dataset
            dataset_splits = dataset.get_splits(self.splits)
            enabled_splits = [split for split in dataset_splits if split.enabled]
            if not enabled_splits:
                raise ValueError(f"Dataset '{dataset.name}' has no enabled splits")
