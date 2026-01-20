import os
from typing import List, Optional

import yaml
from pydantic.dataclasses import dataclass

from flc.flows.labels import FlowLabelType


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a single dataset to be created."""

    name: str
    features_file: str
    flow_labels_file: str
    output_path: str
    label_type: FlowLabelType = FlowLabelType.FLOW
    has_tcp_control_packets: bool = True

    # Filtering parameters
    min_class_occurrences: Optional[int] = None
    skip_invalid_labels: Optional[bool] = None
    discard_label_ids: Optional[List[int]] = None
    discard_flows_without_labels: Optional[bool] = None

    enabled: bool = True

    def get_min_class_occurrences(self, global_default: int) -> int:
        """Get min_class_occurrences, using global default if not specified."""
        return self.min_class_occurrences if self.min_class_occurrences is not None else global_default

    def get_skip_invalid_labels(self, global_default: bool) -> bool:
        """Get skip_invalid_labels, using global default if not specified."""
        return self.skip_invalid_labels if self.skip_invalid_labels is not None else global_default

    def get_discard_label_ids(self, global_default: List[int]) -> List[int]:
        """Get discard_label_ids, using global default if not specified."""
        return self.discard_label_ids if self.discard_label_ids is not None else global_default

    def get_discard_flows_without_labels(self, global_default: bool) -> bool:
        """Get discard_flows_without_labels, using global default if not specified."""
        return self.discard_flows_without_labels if self.discard_flows_without_labels is not None else global_default


@dataclass(frozen=True)
class Config:
    """Configuration for creating multiple flow classification datasets."""

    datasets: List[DatasetSpec]

    # Global filtering defaults (can be overridden per dataset)
    min_class_occurrences: int = 1
    skip_invalid_labels: bool = False
    discard_label_ids: List[int] = None
    discard_flows_without_labels: bool = False

    def __post_init__(self):
        if self.discard_label_ids is None:
            object.__setattr__(self, "discard_label_ids", [])

    @classmethod
    def from_yaml(cls, config_file: str):
        """Load configuration from YAML file."""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse datasets
        datasets = []
        for dataset_dict in config_dict.get("datasets", []):
            if len(dataset_dict.keys()) != 1:
                raise ValueError("Expected dataset entries to consist of exactly one key")

            dataset_name = list(dataset_dict.keys())[0]
            dataset_config = dataset_dict[dataset_name]

            # Validate required fields
            required_fields = ["features_file", "flow_labels_file", "output_path"]
            for field in required_fields:
                if field not in dataset_config:
                    raise ValueError(f"Dataset '{dataset_name}' missing required '{field}'")

            # Convert relative paths to absolute
            features_file = dataset_config["features_file"]
            if not os.path.isabs(features_file):
                features_file = os.path.abspath(features_file)

            flow_labels_file = dataset_config["flow_labels_file"]
            if not os.path.isabs(flow_labels_file):
                flow_labels_file = os.path.abspath(flow_labels_file)

            output_path = dataset_config["output_path"]
            if not os.path.isabs(output_path):
                output_path = os.path.abspath(output_path)

            # Parse label_type
            label_type_str = dataset_config.get("label_type", "flow")
            if label_type_str == "flow":
                label_type = FlowLabelType.FLOW
            elif label_type_str == "group":
                label_type = FlowLabelType.FLOW_GROUP
            else:
                raise ValueError(f"Invalid label_type '{label_type_str}' for dataset '{dataset_name}'. Must be 'flow' or 'group'.")

            dataset_spec = DatasetSpec(
                name=dataset_name,
                features_file=features_file,
                flow_labels_file=flow_labels_file,
                output_path=output_path,
                label_type=label_type,
                has_tcp_control_packets=dataset_config.get("has_tcp_control_packets", True),
                min_class_occurrences=dataset_config.get("min_class_occurrences"),
                skip_invalid_labels=dataset_config.get("skip_invalid_labels"),
                discard_label_ids=dataset_config.get("discard_label_ids"),
                discard_flows_without_labels=dataset_config.get("discard_flows_without_labels"),
                enabled=dataset_config.get("enabled", True),
            )
            datasets.append(dataset_spec)

        return cls(
            datasets=datasets,
            min_class_occurrences=config_dict.get("min_class_occurrences", 1),
            skip_invalid_labels=config_dict.get("skip_invalid_labels", True),
            discard_label_ids=config_dict.get("discard_label_ids", []),
            discard_flows_without_labels=config_dict.get("discard_flows_without_labels", False),
        )
