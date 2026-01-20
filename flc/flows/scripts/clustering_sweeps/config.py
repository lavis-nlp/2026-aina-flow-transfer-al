import os
from typing import Dict, List, Optional, Tuple, Any
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
class ClusteringSweepConfig:
    """Configuration for clustering sweep experiments"""
    
    # Dataset configuration
    dataset_split_paths: List[str]
    
    # Preprocessing configuration
    preprocessing: PreprocessingConfig
    
    # Algorithm sweep configurations
    algorithms: Dict[str, AlgorithmSweepConfig]
    
    # Output configuration
    output_report_path: str
    
    # Execution settings
    random_state: int = 42
    n_jobs: int = 1
    max_flows: Optional[int] = None  # Maximum number of flows to use per sweep iteration

    @classmethod
    def from_yaml(cls, config_file: str) -> "ClusteringSweepConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse preprocessing config
        preprocessing_dict = config_dict.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            scaler_type=preprocessing_dict.get("scaler_type", "robust"),
            clip_quantiles=tuple(preprocessing_dict["clip_quantiles"]) if "clip_quantiles" in preprocessing_dict else (0.01, 0.99),
            log_transform=preprocessing_dict.get("log_transform", True)
        )

        # Parse algorithm configurations
        algorithms = {}
        algorithms_dict = config_dict.get("algorithms", {})
        for algo_name, algo_config in algorithms_dict.items():
            algorithms[algo_name] = AlgorithmSweepConfig(
                enabled=algo_config.get("enabled", True),
                hyperparameters=algo_config.get("hyperparameters", {})
            )

        # Convert relative paths to absolute
        output_report_path = config_dict["output_report_path"]
        if not os.path.isabs(output_report_path):
            output_report_path = os.path.abspath(output_report_path)

        dataset_split_paths = []
        for path in config_dict["dataset_split_paths"]:
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            dataset_split_paths.append(path)

        return cls(
            dataset_split_paths=dataset_split_paths,
            preprocessing=preprocessing,
            algorithms=algorithms,
            output_report_path=output_report_path,
            random_state=config_dict.get("random_state", 42),
            n_jobs=config_dict.get("n_jobs", 1),
            max_flows=config_dict.get("max_flows")
        )

    def get_enabled_algorithms(self) -> Dict[str, AlgorithmSweepConfig]:
        """Get only enabled algorithms"""
        return {name: config for name, config in self.algorithms.items() if config.enabled}

    def validate(self) -> None:
        """Validate configuration"""
        # Check dataset split paths exist
        for path in self.dataset_split_paths:
            if not os.path.exists(path):
                raise ValueError(f"Dataset split file does not exist: {path}")

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