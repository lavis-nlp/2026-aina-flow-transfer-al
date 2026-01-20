import argparse
import logging
import os
from collections import defaultdict
from typing import List

import yaml


class Dataset:
    name: str = None
    enabled: bool = True
    filepaths_cache_file: str = None
    flow_pcaps_folder: str = None
    labels_file: str = None
    features_file: str = None

    def __init__(self, yaml_dataset: dict):
        if len(yaml_dataset.keys()) != 1:
            raise ValueError("Expected dataset entries to consist of exactly one key")

        self.name = list(yaml_dataset.keys())[0]
        yaml_dataset = yaml_dataset[self.name]

        if "enabled" in yaml_dataset:
            self.enabled = bool(yaml_dataset["enabled"])

        if "filepaths-cache-file" in yaml_dataset:
            self.filepaths_cache_file = os.path.abspath(yaml_dataset["filepaths-cache-file"])
        else:
            self.filepaths_cache_file = f"filepaths.{self.name}.pickle"

        if "labels-file" in yaml_dataset:
            self.labels_file = os.path.abspath(yaml_dataset["labels-file"])
        else:
            self.labels_file = f"labels.{self.name}.pickle"

        if "features-file" in yaml_dataset:
            self.features_file = os.path.abspath(yaml_dataset["features-file"])
        else:
            self.features_file = f"features.{self.name}.csv"

        if "flow-pcaps-folder" in yaml_dataset:
            self.flow_pcaps_folder = os.path.abspath(yaml_dataset["flow-pcaps-folder"])
            if not os.path.isdir(self.flow_pcaps_folder):
                raise ValueError(f"Flow pcaps folder {self.flow_pcaps_folder} does not exist or is not a directory")
        else:
            raise ValueError(f"Unable to find key 'flow-pcaps-folder' in dataset {self.name}")


class Config:
    parallel_processes: int = 16
    batch_size: int = 10000
    print_unlabeled: bool = False
    ignore_tcp_control_packets: bool = True
    write_summaries: bool = True
    write_features: bool = True
    datasets: List[Dataset] = []

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise ValueError(f"Could not open config file, file does not exist: {filepath}")

        with open(filepath, "r", encoding="utf-8") as config_file:
            yaml_config = defaultdict(lambda: {}, yaml.safe_load(config_file))

        if "parallel-processes" in yaml_config:
            self.parallel_processes = int(yaml_config["parallel-processes"])

        if "batch-size" in yaml_config:
            self.batch_size = int(yaml_config["batch-size"])

        if "print-unlabeled" in yaml_config:
            self.print_unlabeled = bool(yaml_config["print-unlabeled"])

        if "ignore-tcp-control-packets" in yaml_config:
            self.ignore_tcp_control_packets = bool(yaml_config["ignore-tcp-control-packets"])

        if "write-summaries" in yaml_config:
            self.write_summaries = bool(yaml_config["write-summaries"])

        if "write-features" in yaml_config:
            self.write_features = bool(yaml_config["write-features"])

        if "datasets" in yaml_config:
            for yaml_dataset in yaml_config["datasets"]:
                self.datasets.append(Dataset(yaml_dataset))


def load_config_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="path to config file", required=True, default="configs/example.config.yaml"
    )
    args, _ = parser.parse_known_args()

    if not hasattr(args, "config"):
        raise ValueError("missing parameter --config <filepath>")

    if len(args.config) == 0:
        raise ValueError("parameter --config cannot be empty")

    logging.info("Loading config from args: %s", args.config)
    return Config(filepath=args.config)
