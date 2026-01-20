import os
import hashlib
from typing import Dict, List, Optional, Any
import yaml
from pydantic.dataclasses import dataclass

from flc.shared.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True)
class AlgorithmSweepConfig:
    """Configuration for a single algorithm's parameter sweep"""

    enabled: bool = True
    hyperparameters: Dict[str, List[Any]] = None

    def __post_init__(self):
        if self.hyperparameters is None:
            object.__setattr__(self, "hyperparameters", {})


@dataclass(frozen=True)
class DatasetPairConfig:
    """Configuration for train/test dataset pair"""

    train_split_path: str
    test_split_path: str

    def validate(self) -> None:
        """Validate that both dataset files exist"""
        if not os.path.exists(self.train_split_path):
            raise ValueError(f"Train dataset split file does not exist: {self.train_split_path}")
        if not os.path.exists(self.test_split_path):
            raise ValueError(f"Test dataset split file does not exist: {self.test_split_path}")


@dataclass(frozen=True)
class ClassificationSweepConfig:
    """Configuration for classification sweep experiments"""

    # Dataset configuration
    dataset_pairs: List[DatasetPairConfig] = None

    # Preprocessing configuration
    preprocessing: PreprocessingConfig = None

    # Algorithm sweep configurations
    algorithms: Dict[str, AlgorithmSweepConfig] = None

    # Output configuration
    output_report_path: str = "classification_sweep_results.csv"

    # Execution settings
    random_state: int = 42
    n_jobs: int = 1
    max_flows: Optional[int] = None  # Maximum number of flows to use per dataset

    # Extensibility settings
    config_version: str = "1.0"
    parameter_defaults: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        # Initialize defaults
        if self.preprocessing is None:
            object.__setattr__(self, "preprocessing", PreprocessingConfig())
        if self.algorithms is None:
            object.__setattr__(self, "algorithms", {})
        if self.parameter_defaults is None:
            object.__setattr__(self, "parameter_defaults", {})
        if self.dataset_pairs is None:
            object.__setattr__(self, "dataset_pairs", [])

    @classmethod
    def from_yaml(cls, config_file: str) -> "ClassificationSweepConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse preprocessing config
        preprocessing_dict = config_dict.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            enabled=preprocessing_dict.get("enabled", True),
            scaler_type=preprocessing_dict.get("scaler_type", "robust"),
            clip_quantiles=(
                tuple(preprocessing_dict["clip_quantiles"]) if "clip_quantiles" in preprocessing_dict else (0.01, 0.99)
            ),
            log_transform=preprocessing_dict.get("log_transform", True),
        )

        # Parse algorithm configurations
        algorithms = {}
        algorithms_dict = config_dict.get("algorithms", {})
        for algo_name, algo_config in algorithms_dict.items():
            algorithms[algo_name] = AlgorithmSweepConfig(
                enabled=algo_config.get("enabled", True), hyperparameters=algo_config.get("hyperparameters", {})
            )

        # Parse dataset configuration
        dataset_pairs = []
        if "dataset_pairs" in config_dict:
            for pair_config in config_dict["dataset_pairs"]:
                train_path = pair_config["train_split_path"]
                test_path = pair_config["test_split_path"]
                if not os.path.isabs(train_path):
                    train_path = os.path.abspath(train_path)
                if not os.path.isabs(test_path):
                    test_path = os.path.abspath(test_path)
                dataset_pairs.append(DatasetPairConfig(train_path, test_path))

        # Convert relative paths to absolute for output
        output_report_path = config_dict.get("output_report_path", "classification_sweep_results.csv")
        if not os.path.isabs(output_report_path):
            output_report_path = os.path.abspath(output_report_path)

        # Parse parameter defaults for extensibility
        parameter_defaults = config_dict.get("parameter_defaults", {})

        return cls(
            dataset_pairs=dataset_pairs,
            preprocessing=preprocessing,
            algorithms=algorithms,
            output_report_path=output_report_path,
            random_state=config_dict.get("random_state", 42),
            n_jobs=config_dict.get("n_jobs", 1),
            max_flows=config_dict.get("max_flows"),
            config_version=config_dict.get("config_version", "1.0"),
            parameter_defaults=parameter_defaults,
        )

    def get_enabled_algorithms(self) -> Dict[str, AlgorithmSweepConfig]:
        """Get only enabled algorithms"""
        return {name: config for name, config in self.algorithms.items() if config.enabled}

    def merge_parameter_defaults(self) -> "ClassificationSweepConfig":
        """Apply parameter defaults to algorithms that don't specify new parameters"""
        if not self.parameter_defaults:
            return self

        updated_algorithms = {}
        for algo_name, algo_config in self.algorithms.items():
            if algo_name in self.parameter_defaults:
                # Merge defaults with existing hyperparameters
                defaults = self.parameter_defaults[algo_name]
                merged_hyperparams = dict(algo_config.hyperparameters)

                for param_name, default_value in defaults.items():
                    if param_name not in merged_hyperparams:
                        merged_hyperparams[param_name] = [default_value]

                updated_algorithms[algo_name] = AlgorithmSweepConfig(
                    enabled=algo_config.enabled, hyperparameters=merged_hyperparams
                )
            else:
                updated_algorithms[algo_name] = algo_config

        # Create new config with merged algorithms
        new_config_dict = {
            "dataset_pairs": self.dataset_pairs,
            "preprocessing": self.preprocessing,
            "algorithms": updated_algorithms,
            "output_report_path": self.output_report_path,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "max_flows": self.max_flows,
            "config_version": self.config_version,
            "parameter_defaults": self.parameter_defaults,
        }

        return self.__class__(**new_config_dict)

    def get_dataset_pairs(self) -> List[DatasetPairConfig]:
        """Get list of dataset pairs for iteration"""
        return self.dataset_pairs or []

    def validate(self) -> None:
        """Validate configuration"""
        # Check dataset pairs exist
        dataset_pairs = self.get_dataset_pairs()
        if not dataset_pairs:
            raise ValueError("No dataset pairs configured")

        for pair in dataset_pairs:
            pair.validate()

        # Check at least one algorithm is enabled
        enabled_algos = self.get_enabled_algorithms()
        if not enabled_algos:
            raise ValueError("No algorithms are enabled")

        # Check output directory exists
        output_dir = os.path.dirname(self.output_report_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

        # Validate max_flows parameter
        if self.max_flows is not None:
            if not isinstance(self.max_flows, int) or self.max_flows <= 0:
                raise ValueError("max_flows must be a positive integer")

    def get_config_hash(self) -> str:
        """Generate a hash of the configuration for tracking parameter evolution"""
        config_str = f"{self.config_version}_{self.preprocessing}_{self.algorithms}_{self.parameter_defaults}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
