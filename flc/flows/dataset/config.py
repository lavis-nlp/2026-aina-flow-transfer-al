from dataclasses import asdict

from pydantic.dataclasses import dataclass
import yaml

from flc.flows.labels import FlowLabelType

DATASET_CONFIG_FILE = "config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str
    has_tcp_control_packets: bool

    features_file: str
    labels_file: str
    label_type: FlowLabelType

    config_file: str

    created_at: str  # timestamp

    _source_files: dict[str, str]  # list of source files used to create the dataset
    _meta: dict | None = None
