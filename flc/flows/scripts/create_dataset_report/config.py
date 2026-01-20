import os
from typing import List

import yaml
from pydantic.dataclasses import dataclass

from flc.flows.labels import FlowLabelType


@dataclass(frozen=True)
class Config:
    """Configuration for creating dataset reports."""

    label_type: str
    output_file: str
    split_files: List[str]

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate label_type
        if self.label_type not in FlowLabelType.values():
            raise ValueError(f"Invalid label_type '{self.label_type}'. Must be one of: {FlowLabelType.values()}")

        # Validate split_files
        if not self.split_files:
            raise ValueError("split_files cannot be empty")

        # Validate that all split files exist
        for split_file in self.split_files:
            if not os.path.exists(split_file):
                raise ValueError(f"Split file does not exist: {split_file}")

        # Validate output_file directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

    @property
    def flow_label_type(self) -> FlowLabelType:
        """Get FlowLabelType enum from string."""
        return FlowLabelType.from_str(self.label_type)

    @classmethod
    def from_yaml(cls, config_file: str) -> "Config":
        """Load configuration from YAML file."""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["label_type", "output_file", "split_files"]
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required field '{field}' in config file")

        # Convert relative paths to absolute paths
        split_files = config_dict["split_files"]
        absolute_split_files = []
        for split_file in split_files:
            if not os.path.isabs(split_file):
                absolute_split_files.append(os.path.abspath(split_file))
            else:
                absolute_split_files.append(split_file)

        output_file = config_dict["output_file"]
        if not os.path.isabs(output_file):
            output_file = os.path.abspath(output_file)

        return cls(
            label_type=config_dict["label_type"],
            output_file=output_file,
            split_files=absolute_split_files,
        )