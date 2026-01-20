import os
from typing import Dict, List, Optional, Any

import yaml
from pydantic.dataclasses import dataclass

from flc.shared.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True)
class DatasetTripletConfig:
    """Configuration for source/target/test dataset combination"""

    source_split_path: str
    target_split_path: str
    test_split_path: str

    def validate(self) -> None:
        """Validate that all dataset files exist"""
        if not os.path.exists(self.source_split_path):
            raise ValueError(f"Source dataset split file does not exist: {self.source_split_path}")
        if not os.path.exists(self.target_split_path):
            raise ValueError(f"Target dataset split file does not exist: {self.target_split_path}")
        if not os.path.exists(self.test_split_path):
            raise ValueError(f"Test dataset split file does not exist: {self.test_split_path}")


@dataclass(frozen=True)
class ClassifierALSweepConfig:
    """Configuration for classifier-based active learning sweeps"""

    # Query strategy configurations
    query_strategies: Dict[str, Dict[str, List[Any]]]

    # Weighting strategy configurations
    weighting_strategies: Dict[str, Dict[str, List[Any]]]

    # Classifier configurations
    classifier_configs: Dict[str, Dict[str, List[Any]]]

    # Active learning parameters
    samples_per_iteration: List[int]
    max_iterations: List[int]
    max_total_samples: List[Optional[int]]

    # Evaluation parameters
    evaluation_interval: List[int]
    test_evaluation_interval: List[int]
    evaluate_on_test: List[bool]


@dataclass(frozen=True)
class ClassifierTransferALSweepConfig:
    """Configuration for classifier transfer active learning sweep experiments"""

    # Dataset configuration
    dataset_triplets: List[DatasetTripletConfig]

    # Preprocessing configuration
    preprocessing: PreprocessingConfig

    # Classifier AL configuration
    classifier_al: ClassifierALSweepConfig

    # Output configuration
    output_base_dir: str

    # Execution settings
    random_states: List[int]
    n_jobs: int
    max_flows: Optional[int]

    # Extensibility
    parameter_defaults: Dict[str, Dict[str, Any]]

    @classmethod
    def from_yaml(cls, config_file: str):
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse preprocessing config
        preprocessing_dict = config_dict["preprocessing"]
        preprocessing = PreprocessingConfig(
            enabled=preprocessing_dict.get("enabled", None),
            scaler_type=preprocessing_dict.get("scaler_type", None),
            clip_quantiles=(
                tuple(preprocessing_dict["clip_quantiles"]) if "clip_quantiles" in preprocessing_dict else None
            ),
            log_transform=preprocessing_dict.get("log_transform", None),
        )

        # Parse dataset triplets
        dataset_triplets = []
        if "dataset_triplets" in config_dict:
            for triplet_config in config_dict["dataset_triplets"]:
                source_path = triplet_config["source_split_path"]
                target_path = triplet_config["target_split_path"]
                test_path = triplet_config["test_split_path"]

                # Convert to absolute paths
                if not os.path.isabs(source_path):
                    source_path = os.path.abspath(source_path)
                if not os.path.isabs(target_path):
                    target_path = os.path.abspath(target_path)
                if not os.path.isabs(test_path):
                    test_path = os.path.abspath(test_path)

                dataset_triplets.append(DatasetTripletConfig(source_path, target_path, test_path))

        # Parse classifier AL config
        classifier_al_dict = config_dict["classifier_al"]
        classifier_al = ClassifierALSweepConfig(
            query_strategies=classifier_al_dict["query_strategies"],
            weighting_strategies=classifier_al_dict["weighting_strategies"],
            classifier_configs=classifier_al_dict["classifier_configs"],
            samples_per_iteration=classifier_al_dict["samples_per_iteration"],
            max_iterations=classifier_al_dict["max_iterations"],
            max_total_samples=classifier_al_dict["max_total_samples"],
            evaluation_interval=classifier_al_dict["evaluation_interval"],
            test_evaluation_interval=classifier_al_dict["test_evaluation_interval"],
            evaluate_on_test=classifier_al_dict["evaluate_on_test"],
        )

        # Convert relative paths to absolute for output
        output_base_dir = config_dict["output_base_dir"]
        if not os.path.isabs(output_base_dir):
            output_base_dir = os.path.abspath(output_base_dir)

        # Parse parameter defaults for extensibility
        parameter_defaults = config_dict["parameter_defaults"]

        return cls(
            dataset_triplets=dataset_triplets,
            preprocessing=preprocessing,
            classifier_al=classifier_al,
            output_base_dir=output_base_dir,
            random_states=config_dict["random_states"],
            n_jobs=config_dict["n_jobs"],
            max_flows=config_dict["max_flows"],
            parameter_defaults=parameter_defaults,
        )

    def merge_parameter_defaults(self) -> "ClassifierTransferALSweepConfig":
        """Apply parameter defaults to configurations that don't specify parameters"""
        if not self.parameter_defaults:
            return self

        # Apply defaults to classifier AL configuration
        classifier_al_defaults = self.parameter_defaults.get("classifier_al", {})

        # Merge query strategy defaults
        updated_query_strategies = dict(self.classifier_al.query_strategies)
        for strategy_name, default_config in classifier_al_defaults.items():
            if strategy_name in updated_query_strategies:
                # Merge defaults with existing config
                strategy_config = dict(updated_query_strategies[strategy_name])
                for param_name, default_value in default_config.items():
                    if param_name not in strategy_config:
                        strategy_config[param_name] = [default_value]
                updated_query_strategies[strategy_name] = strategy_config

        # Merge weighting strategy defaults
        updated_weighting_strategies = dict(self.classifier_al.weighting_strategies)
        weighting_defaults = self.parameter_defaults.get("weighting_strategies", {})
        for strategy_name, default_config in weighting_defaults.items():
            if strategy_name in updated_weighting_strategies:
                # Merge defaults with existing config
                strategy_config = dict(updated_weighting_strategies[strategy_name])
                for param_name, default_value in default_config.items():
                    if param_name not in strategy_config:
                        strategy_config[param_name] = [default_value]
                updated_weighting_strategies[strategy_name] = strategy_config

        # Merge classifier defaults
        updated_classifier_configs = dict(self.classifier_al.classifier_configs)
        for classifier_name, default_config in self.parameter_defaults.items():
            if classifier_name in updated_classifier_configs:
                # Merge defaults with existing config
                classifier_config = dict(updated_classifier_configs[classifier_name])
                for param_name, default_value in default_config.items():
                    if param_name not in classifier_config:
                        classifier_config[param_name] = [default_value]
                updated_classifier_configs[classifier_name] = classifier_config

        # Create updated classifier AL config
        updated_classifier_al = ClassifierALSweepConfig(
            query_strategies=updated_query_strategies,
            weighting_strategies=updated_weighting_strategies,
            classifier_configs=updated_classifier_configs,
            samples_per_iteration=self.classifier_al.samples_per_iteration,
            max_iterations=self.classifier_al.max_iterations,
            max_total_samples=self.classifier_al.max_total_samples,
            evaluation_interval=self.classifier_al.evaluation_interval,
            test_evaluation_interval=self.classifier_al.test_evaluation_interval,
            evaluate_on_test=self.classifier_al.evaluate_on_test,
        )

        # Create new config with merged settings
        return self.__class__(
            dataset_triplets=self.dataset_triplets,
            preprocessing=self.preprocessing,
            classifier_al=updated_classifier_al,
            output_base_dir=self.output_base_dir,
            random_states=self.random_states,
            n_jobs=self.n_jobs,
            max_flows=self.max_flows,
            parameter_defaults=self.parameter_defaults,
        )

    def validate(self) -> None:
        """Validate configuration"""
        # Check dataset triplets exist
        if not self.dataset_triplets:
            raise ValueError("No dataset triplets configured")

        for triplet in self.dataset_triplets:
            triplet.validate()

        # Check at least one query strategy is configured
        if not self.classifier_al.query_strategies:
            raise ValueError("No query strategies are configured")

        # Check at least one weighting strategy is configured
        if not self.classifier_al.weighting_strategies:
            raise ValueError("No weighting strategies are configured")

        # Check at least one classifier is configured
        if not self.classifier_al.classifier_configs:
            raise ValueError("No classifier configurations are provided")

        # Check AL parameters
        if not self.classifier_al.samples_per_iteration:
            raise ValueError("samples_per_iteration must be specified")
        if not self.classifier_al.max_iterations:
            raise ValueError("max_iterations must be specified")

        # Check random seeds
        if not self.random_states:
            raise ValueError("At least one random seed must be specified")

        # Validate max_flows parameter
        if self.max_flows is not None:
            if not isinstance(self.max_flows, int) or self.max_flows <= 0:
                raise ValueError("max_flows must be a positive integer")

        # Check output directory can be created
        output_dir = os.path.dirname(self.output_base_dir)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create output directory: {e}")
