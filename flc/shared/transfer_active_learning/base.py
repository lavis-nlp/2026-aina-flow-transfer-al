from typing import Any, Dict, Optional
from pydantic.dataclasses import dataclass
from pydantic import Field

# Type alias for sample IDs
SampleId = str


@dataclass
class ActiveLearningConfig:
    """Base configuration for active learning components"""

    random_state: Optional[int] = Field(description="Random seed for reproducibility")


@dataclass
class ClassifierActiveLearningConfig:
    """Configuration for classifier-based transfer active learning"""

    # Active learning parameters
    samples_per_iteration: int

    # Model configuration
    classifier_name: str
    classifier_config: Dict[str, Any]

    random_state: int


@dataclass
class ClusteringActiveLearningConfig:
    """Configuration for clustering-based transfer active learning"""

    # Strategy configurations
    query_strategy: str
    query_strategy_config: Dict[str, Any]
    weighting_strategy: str
    weighting_strategy_config: Dict[str, Any]
    label_selection_strategy: str
    label_selection_config: Dict[str, Any]

    clustering_algorithm: str
    clustering_config: Dict[str, Any]

    # Active learning parameters
    max_iterations: int
    max_total_samples: Optional[int]  # Budget constraint

    # Model configuration
    classifier_name: str
    classifier_config: Dict[str, Any]

    # Evaluation and reporting
    evaluation_interval: int
    test_evaluation_interval: int
    evaluate_on_test: bool
    output_dir: str
    save_models: bool
    random_state: int
