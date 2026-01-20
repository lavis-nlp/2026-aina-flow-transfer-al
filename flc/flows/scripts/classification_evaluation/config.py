from dataclasses import asdict
import os
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pydantic.dataclasses import dataclass

from flc.shared.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True)
class TestDatasetConfig:
    """Configuration for a test dataset"""
    
    name: str
    path: str


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset with one train and multiple test datasets"""

    name: str
    train_split_path: str
    test_datasets: List[TestDatasetConfig]
    preprocessing: Optional[PreprocessingConfig] = None

    def get_preprocessing_config(self, global_config: Optional[PreprocessingConfig] = None) -> PreprocessingConfig:
        """
        Get the effective preprocessing configuration for this dataset.
        If a global preprocessing config is provided, it will override this dataset's config.
        """

        params = dict()
        global_defaults = asdict(global_config) if global_config else {}

        params.update(**global_defaults)

        if self.preprocessing:
            # Override global defaults with dataset-specific preprocessing
            params.update(asdict(self.preprocessing))

        return PreprocessingConfig(**params)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a classification model"""

    name: str  # "random_forest", "xgboost", etc.
    hyperparameters: Dict[str, Any]
    enabled: bool = True

    def __post_init__(self):
        if self.hyperparameters is None:
            object.__setattr__(self, "hyperparameters", {})


@dataclass(frozen=True)
class ClassificationEvaluationConfig:
    """Configuration for classification evaluation experiments"""

    # Dataset configurations
    datasets: List[DatasetConfig]

    # Model configurations
    models: List[ModelConfig]

    # Global preprocessing configuration (can be overridden per dataset)
    global_preprocessing: Optional[PreprocessingConfig] = None

    # Output configuration
    output_report_path: str = None
    output_directory: Optional[str] = None

    # Execution settings
    random_state: int = 42

    @classmethod
    def from_yaml(cls, config_file: str) -> "ClassificationEvaluationConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(config_file):
            raise ValueError(f"Config file does not exist: {config_file}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse global preprocessing config
        global_preprocessing = None
        if "global_preprocessing" in config_dict:
            global_preprocessing_dict = config_dict["global_preprocessing"]
            global_preprocessing = PreprocessingConfig(**global_preprocessing_dict)

        # Parse dataset configurations
        datasets = []
        for dataset_dict in config_dict.get("datasets", []):
            # Parse preprocessing config (can override global config)
            preprocessing = None
            if "preprocessing" in dataset_dict:
                preprocessing_dict = dataset_dict["preprocessing"]
                preprocessing = PreprocessingConfig(**preprocessing_dict)

            # Convert relative paths to absolute
            train_split_path = dataset_dict["train_split_path"]
            if not os.path.isabs(train_split_path):
                train_split_path = os.path.abspath(train_split_path)

            # Parse test datasets
            test_datasets = []
            for test_dict in dataset_dict.get("test_datasets", []):
                test_path = test_dict["path"]
                if not os.path.isabs(test_path):
                    test_path = os.path.abspath(test_path)
                
                test_dataset = TestDatasetConfig(
                    name=test_dict["name"],
                    path=test_path
                )
                test_datasets.append(test_dataset)

            dataset = DatasetConfig(
                name=dataset_dict["name"],
                train_split_path=train_split_path,
                test_datasets=test_datasets,
                preprocessing=preprocessing,
            )
            datasets.append(dataset)

        # Parse model configurations
        models = []
        for model_dict in config_dict.get("models", []):
            model = ModelConfig(
                name=model_dict["name"],
                hyperparameters=model_dict.get("hyperparameters", {}),
                enabled=model_dict.get("enabled", True),
            )
            models.append(model)

        # Convert relative paths to absolute
        output_report_path = config_dict["output_report_path"]
        if not os.path.isabs(output_report_path):
            output_report_path = os.path.abspath(output_report_path)

        output_directory = config_dict.get("output_directory")
        if output_directory and not os.path.isabs(output_directory):
            output_directory = os.path.abspath(output_directory)

        return cls(
            datasets=datasets,
            models=models,
            global_preprocessing=global_preprocessing,
            output_report_path=output_report_path,
            output_directory=output_directory,
            random_state=config_dict.get("random_state", 42),
        )

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get only enabled models"""
        return [model for model in self.models if model.enabled]

    def validate(self) -> None:
        """Validate configuration"""
        # Check dataset split paths exist
        for dataset in self.datasets:
            if not os.path.exists(dataset.train_split_path):
                raise ValueError(f"Train split file does not exist: {dataset.train_split_path}")
            
            # Check each test dataset path exists
            for test_dataset in dataset.test_datasets:
                if not os.path.exists(test_dataset.path):
                    raise ValueError(f"Test split file does not exist: {test_dataset.path}")
            
            # Check that at least one test dataset is provided
            if not dataset.test_datasets:
                raise ValueError(f"No test datasets provided for dataset: {dataset.name}")

        # Check at least one model is enabled
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            raise ValueError("No models are enabled")

        # Check each dataset has a preprocessing config
        if self.global_preprocessing is None:
            for dataset in self.datasets:
                if dataset.preprocessing is None:
                    raise ValueError(f"Dataset {dataset.name} must have a preprocessing config")
