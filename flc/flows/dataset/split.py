import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import yaml
from pydantic.dataclasses import dataclass

from flc.flows.labels import FlowLabelType


class SplitType(Enum):
    """Enumeration of supported dataset split types."""

    FULL = "full"
    DEV = "dev"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class SplitMethod(Enum):
    """Enumeration of supported split creation methods."""

    RANDOM = "random"
    MANUAL = "manual"
    # STRATIFIED = "stratified"  # maybe implement later
    # TEMPORAL = "temporal"


@dataclass(frozen=True)
class SplitParameters:
    """Parameters used to create a dataset split."""

    split_method: SplitMethod
    split_ratio: Optional[float] = None
    random_seed: Optional[int] = None
    parent_dataset_path: Optional[str] = None
    stratify_by: Optional[str] = None


@dataclass(frozen=True)
class LabelStatistic:
    """Statistics for a single label in a dataset split."""

    count: int
    percentage: float
    name: str


@dataclass(frozen=True)
class SplitStatistics:
    """Statistics for a dataset split."""

    total_flows: int
    flows_with_labels: int
    flows_without_labels: int
    unique_labels: int
    label_statistics: Dict[str, LabelStatistic]


@dataclass(frozen=True)
class DatasetSplit:
    """Configuration and data for a dataset split."""

    # Metadata
    dataset_name: str
    split_name: str
    split_type: SplitType
    created_at: str
    label_type: FlowLabelType

    # Split creation parameters
    split_parameters: SplitParameters

    # Statistics
    statistics: SplitStatistics

    # Path to CSV file containing flow_id,index_in_features,index_in_labels mapping
    flow_index_csv_path: str

    @classmethod
    def from_yaml(cls, file_path: str) -> "DatasetSplit":
        """
        Load a dataset split from a YAML file.

        Args:
            file_path: Path to the YAML split file

        Returns:
            DatasetSplit object loaded from the file

        Raises:
            ValueError: If file doesn't exist or has invalid format
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Split file does not exist: {file_path}")

        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            # Parse split parameters
            split_params_data = data.get("split_parameters", {})
            split_parameters = SplitParameters(
                split_method=SplitMethod(split_params_data.get("split_method", "manual")),
                split_ratio=split_params_data.get("split_ratio"),
                random_seed=split_params_data.get("random_seed"),
                parent_dataset_path=split_params_data.get("parent_dataset_path"),
                stratify_by=split_params_data.get("stratify_by"),
            )

            # Parse label statistics
            label_stats_data = data.get("statistics", {}).get("label_statistics", {})
            label_statistics = {}
            for label_id, stats in label_stats_data.items():
                label_statistics[label_id] = LabelStatistic(
                    count=stats["count"], percentage=stats["percentage"], name=stats["name"]
                )

            # Parse split statistics
            stats_data = data.get("statistics", {})
            statistics = SplitStatistics(
                total_flows=stats_data.get("total_flows", 0),
                flows_with_labels=stats_data.get("flows_with_labels", 0),
                flows_without_labels=stats_data.get("flows_without_labels", 0),
                unique_labels=stats_data.get("unique_labels", 0),
                label_statistics=label_statistics,
            )

            return cls(
                dataset_name=data["dataset_name"],
                split_name=data["split_name"],
                split_type=SplitType(data["split_type"]),
                created_at=data["created_at"],
                label_type=FlowLabelType(data["label_type"]),
                split_parameters=split_parameters,
                statistics=statistics,
                flow_index_csv_path=data["flow_index_csv_path"],
            )

        except Exception as e:
            raise ValueError(f"Error loading split file {file_path}: {e}")

    def to_yaml(self, file_path: str) -> None:
        """
        Save the dataset split to a YAML file.

        Args:
            file_path: Path where to save the YAML split file
        """
        # Convert label statistics to dict format
        label_stats_dict = {}
        for label_id, stats in self.statistics.label_statistics.items():
            label_stats_dict[label_id] = {"count": stats.count, "percentage": stats.percentage, "name": stats.name}

        # Build the complete data structure
        data = {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "split_type": self.split_type.value,
            "created_at": self.created_at,
            "label_type": self.label_type.value,
            "split_parameters": {
                "split_method": self.split_parameters.split_method.value,
                "split_ratio": self.split_parameters.split_ratio,
                "random_seed": self.split_parameters.random_seed,
                "parent_dataset_path": self.split_parameters.parent_dataset_path,
                "stratify_by": self.split_parameters.stratify_by,
            },
            "statistics": {
                "total_flows": self.statistics.total_flows,
                "flows_with_labels": self.statistics.flows_with_labels,
                "flows_without_labels": self.statistics.flows_without_labels,
                "unique_labels": self.statistics.unique_labels,
                "label_statistics": label_stats_dict,
            },
            "flow_index_csv_path": self.flow_index_csv_path,
        }

        # Remove None values from split_parameters for cleaner output
        data["split_parameters"] = {k: v for k, v in data["split_parameters"].items() if v is not None}

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write to YAML file
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True, indent=2)

    @classmethod
    def create(
        cls,
        dataset_name: str,
        split_name: str,
        split_type: SplitType,
        label_type: FlowLabelType,
        flow_index_csv_path: str,
        statistics: SplitStatistics,
        split_parameters: Optional[SplitParameters] = None,
        created_at: Optional[str] = None,
    ) -> "DatasetSplit":
        """
        Create a dataset split with the given parameters.

        Args:
            dataset_name: Name of the dataset
            split_name: Name of the split (e.g., "train", "test", "full")
            split_type: Type of the split (SplitType enum)
            label_type: Type of labels (flow or group)
            flow_index_csv_path: Path to CSV file containing flow_id,index mapping
            statistics: Statistics for this split
            split_parameters: Parameters used to create the split (defaults to MANUAL)
            created_at: Creation timestamp (defaults to current time)

        Returns:
            DatasetSplit object
        """
        if split_parameters is None:
            split_parameters = SplitParameters(split_method=SplitMethod.MANUAL)

        if created_at is None:
            created_at = datetime.now().isoformat()

        return cls(
            dataset_name=dataset_name,
            split_name=split_name,
            split_type=split_type,
            created_at=created_at,
            label_type=label_type,
            split_parameters=split_parameters,
            statistics=statistics,
            flow_index_csv_path=flow_index_csv_path,
        )

    def get_flow_index_data(self) -> pd.DataFrame:
        """
        Load flow index data from the CSV file.

        Returns:
            DataFrame with columns: flow_id, index_in_features, index_in_labels

        Raises:
            ValueError: If CSV file doesn't exist or has invalid format
        """
        if not os.path.exists(self.flow_index_csv_path):
            raise ValueError(f"Flow index CSV file does not exist: {self.flow_index_csv_path}")

        try:
            df = pd.read_csv(self.flow_index_csv_path)
            expected_columns = {"flow_id", "index_in_features", "index_in_labels"}
            if not expected_columns.issubset(df.columns):
                raise ValueError(f"CSV file must contain columns: {expected_columns}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading flow index CSV {self.flow_index_csv_path}: {e}")

    def get_flow_ids(self) -> List[str]:
        """
        Get list of flow IDs in this split.

        Returns:
            List of flow IDs
        """
        return self.get_flow_index_data()["flow_id"].tolist()

    def get_feature_indices(self) -> List[int]:
        """
        Get list of feature file row indices for flows in this split.

        Returns:
            List of row indices in features CSV file
        """
        return self.get_flow_index_data()["index_in_features"].tolist()

    def get_label_indices(self) -> List[int]:
        """
        Get list of label file row indices for flows in this split.

        Returns:
            List of row indices in labels CSV file
        """
        return self.get_flow_index_data()["index_in_labels"].tolist()
