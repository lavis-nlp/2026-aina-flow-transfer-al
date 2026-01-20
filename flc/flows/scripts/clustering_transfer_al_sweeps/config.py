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
class ClusteringALSweepConfig:
    """Configuration for clustering-based active learning sweeps"""

    # Query strategy configurations (cluster-based)
    query_strategies: Dict[str, Dict[str, List[Any]]]

    # Weighting strategy configurations
    weighting_strategies: Dict[str, Dict[str, List[Any]]]

    # Clustering algorithm configurations
    clustering_algorithms: Dict[str, Dict[str, List[Any]]]

    # Label selection strategy configurations
    label_selection_strategies: Dict[str, Dict[str, List[Any]]]

    # Classifier configurations
    classifier_configs: Dict[str, Dict[str, List[Any]]]

    # Active learning parameters
    max_iterations: List[int]
    max_total_samples: List[Optional[int]]

    # Evaluation parameters
    evaluation_interval: List[int]
    test_evaluation_interval: List[int]
    evaluate_on_test: List[bool]


@dataclass(frozen=True)
class ClusteringTransferALSweepConfig:
    """Configuration for clustering transfer active learning sweep experiments"""

    # Dataset configuration
    dataset_triplets: List[DatasetTripletConfig]

    # Preprocessing configurations
    clustering_preprocessing: PreprocessingConfig
    classification_preprocessing: PreprocessingConfig

    # Clustering AL configuration
    clustering_al: ClusteringALSweepConfig

    # Output configuration
    output_base_dir: str

    # Execution settings
    random_states: List[int]
    n_jobs: int

    # Extensibility
    parameter_defaults: Dict[str, Dict[str, Any]]

    # Optional fields
    max_flows: Optional[int] = None

    @classmethod
    def from_yaml(cls, config_file: str) -> "ClusteringTransferALSweepConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse clustering preprocessing config
        clustering_preprocessing_dict = config_dict.get("clustering_preprocessing", {})
        clustering_preprocessing = PreprocessingConfig(
            enabled=clustering_preprocessing_dict.get("enabled", True),
            scaler_type=clustering_preprocessing_dict.get("scaler_type", None),
            clip_quantiles=(
                tuple(clustering_preprocessing_dict["clip_quantiles"])
                if "clip_quantiles" in clustering_preprocessing_dict
                else None
            ),
            log_transform=clustering_preprocessing_dict.get("log_transform", None),
        )

        # Parse classification preprocessing config
        classification_preprocessing_dict = config_dict.get("classification_preprocessing", {})
        classification_preprocessing = PreprocessingConfig(
            enabled=classification_preprocessing_dict.get("enabled", True),
            scaler_type=classification_preprocessing_dict.get("scaler_type", None),
            clip_quantiles=(
                tuple(classification_preprocessing_dict["clip_quantiles"])
                if "clip_quantiles" in classification_preprocessing_dict
                else None
            ),
            log_transform=classification_preprocessing_dict.get("log_transform", None),
        )

        # Parse dataset triplets
        dataset_triplets = []
        if "dataset_triplets" in config_dict:
            for triplet_config in config_dict["dataset_triplets"]:
                source_path = triplet_config["source_split_path"]
                target_path = triplet_config["target_split_path"]
                test_path = triplet_config.get("test_split_path")

                # Convert to absolute paths
                if not os.path.isabs(source_path):
                    source_path = os.path.abspath(source_path)
                if not os.path.isabs(target_path):
                    target_path = os.path.abspath(target_path)
                if not os.path.isabs(test_path):
                    test_path = os.path.abspath(test_path)

                dataset_triplets.append(DatasetTripletConfig(source_path, target_path, test_path))

        # Parse clustering AL config
        clustering_al_dict = config_dict["clustering_al"]
        clustering_al = ClusteringALSweepConfig(
            query_strategies=clustering_al_dict["query_strategies"],
            weighting_strategies=clustering_al_dict["weighting_strategies"],
            clustering_algorithms=clustering_al_dict["clustering_algorithms"],
            label_selection_strategies=clustering_al_dict["label_selection_strategies"],
            classifier_configs=clustering_al_dict["classifier_configs"],
            max_iterations=clustering_al_dict["max_iterations"],
            max_total_samples=clustering_al_dict["max_total_samples"],
            evaluation_interval=clustering_al_dict["evaluation_interval"],
            test_evaluation_interval=clustering_al_dict["test_evaluation_interval"],
            evaluate_on_test=clustering_al_dict["evaluate_on_test"],
        )

        # Convert relative paths to absolute for output
        output_base_dir = config_dict["output_base_dir"]
        if not os.path.isabs(output_base_dir):
            output_base_dir = os.path.abspath(output_base_dir)

        # Parse parameter defaults for extensibility
        parameter_defaults = config_dict["parameter_defaults"]

        return cls(
            dataset_triplets=dataset_triplets,
            clustering_preprocessing=clustering_preprocessing,
            classification_preprocessing=classification_preprocessing,
            clustering_al=clustering_al,
            output_base_dir=output_base_dir,
            random_states=config_dict["random_states"],
            n_jobs=config_dict["n_jobs"],
            parameter_defaults=parameter_defaults,
            max_flows=config_dict.get("max_flows"),
        )

    def merge_parameter_defaults(self) -> "ClusteringTransferALSweepConfig":
        """Apply parameter defaults to configurations that don't specify parameters"""
        if not self.parameter_defaults:
            return self

        # Merge query strategy defaults
        updated_query_strategies = dict(self.clustering_al.query_strategies)
        query_strategy_defaults = self.parameter_defaults.get("query_strategies", {})
        for strategy_name, default_config in query_strategy_defaults.items():
            if strategy_name in updated_query_strategies:
                # Merge defaults with existing config
                strategy_config = dict(updated_query_strategies[strategy_name])
                for param_name, default_value in default_config.items():
                    if param_name not in strategy_config:
                        strategy_config[param_name] = [default_value]
                updated_query_strategies[strategy_name] = strategy_config

        # Merge weighting strategy defaults
        updated_weighting_strategies = dict(self.clustering_al.weighting_strategies)
        weighting_defaults = self.parameter_defaults.get("weighting_strategies", {})
        for strategy_name, default_config in weighting_defaults.items():
            if strategy_name in updated_weighting_strategies:
                # Merge defaults with existing config
                strategy_config = dict(updated_weighting_strategies[strategy_name])
                for param_name, default_value in default_config.items():
                    if param_name not in strategy_config:
                        strategy_config[param_name] = [default_value]
                updated_weighting_strategies[strategy_name] = strategy_config

        # Merge clustering algorithm defaults
        updated_clustering_algorithms = dict(self.clustering_al.clustering_algorithms)
        clustering_defaults = self.parameter_defaults.get("clustering_algorithms", {})
        for algorithm_name, default_config in clustering_defaults.items():
            if algorithm_name in updated_clustering_algorithms:
                # Merge defaults with existing config
                algorithm_config = dict(updated_clustering_algorithms[algorithm_name])
                for param_name, default_value in default_config.items():
                    if param_name not in algorithm_config:
                        algorithm_config[param_name] = [default_value]
                updated_clustering_algorithms[algorithm_name] = algorithm_config

        # Merge label selection strategy defaults
        updated_label_selection_strategies = dict(self.clustering_al.label_selection_strategies)
        label_selection_defaults = self.parameter_defaults.get("label_selection_strategies", {})
        for strategy_name, default_config in label_selection_defaults.items():
            if strategy_name in updated_label_selection_strategies:
                # Merge defaults with existing config
                strategy_config = dict(updated_label_selection_strategies[strategy_name])
                for param_name, default_value in default_config.items():
                    if param_name not in strategy_config:
                        strategy_config[param_name] = [default_value]
                updated_label_selection_strategies[strategy_name] = strategy_config

        # Merge classifier defaults
        updated_classifier_configs = dict(self.clustering_al.classifier_configs)
        classifier_defaults = self.parameter_defaults.get("classifier_configs", {})
        for classifier_name, default_config in classifier_defaults.items():
            if classifier_name in updated_classifier_configs:
                # Merge defaults with existing config
                classifier_config = dict(updated_classifier_configs[classifier_name])
                for param_name, default_value in default_config.items():
                    if param_name not in classifier_config:
                        classifier_config[param_name] = [default_value]
                updated_classifier_configs[classifier_name] = classifier_config

        # Create updated clustering AL config
        updated_clustering_al = ClusteringALSweepConfig(
            query_strategies=updated_query_strategies,
            weighting_strategies=updated_weighting_strategies,
            clustering_algorithms=updated_clustering_algorithms,
            label_selection_strategies=updated_label_selection_strategies,
            classifier_configs=updated_classifier_configs,
            max_iterations=self.clustering_al.max_iterations,
            max_total_samples=self.clustering_al.max_total_samples,
            evaluation_interval=self.clustering_al.evaluation_interval,
            test_evaluation_interval=self.clustering_al.test_evaluation_interval,
            evaluate_on_test=self.clustering_al.evaluate_on_test,
        )

        # Create new config with merged settings
        return self.__class__(
            dataset_triplets=self.dataset_triplets,
            clustering_preprocessing=self.clustering_preprocessing,
            classification_preprocessing=self.classification_preprocessing,
            clustering_al=updated_clustering_al,
            output_base_dir=self.output_base_dir,
            random_states=self.random_states,
            n_jobs=self.n_jobs,
            parameter_defaults=self.parameter_defaults,
            max_flows=self.max_flows,
        )

    def validate(self) -> None:
        """Validate configuration"""
        # Check dataset triplets exist
        if not self.dataset_triplets:
            raise ValueError("No dataset triplets configured")

        for triplet in self.dataset_triplets:
            triplet.validate()

        # Check at least one query strategy is configured
        if not self.clustering_al.query_strategies:
            raise ValueError("No query strategies are configured")

        # Check at least one weighting strategy is configured
        if not self.clustering_al.weighting_strategies:
            raise ValueError("No weighting strategies are configured")

        # Check at least one clustering algorithm is configured
        if not self.clustering_al.clustering_algorithms:
            raise ValueError("No clustering algorithms are configured")

        # Check at least one label selection strategy is configured
        if not self.clustering_al.label_selection_strategies:
            raise ValueError("No label selection strategies are configured")

        # Check at least one classifier is configured
        if not self.clustering_al.classifier_configs:
            raise ValueError("No classifier configurations are provided")

        # Check AL parameters
        if not self.clustering_al.max_iterations:
            raise ValueError("max_iterations must be specified")

        # Check random seeds
        if not self.random_states:
            raise ValueError("At least one random state must be specified")

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
