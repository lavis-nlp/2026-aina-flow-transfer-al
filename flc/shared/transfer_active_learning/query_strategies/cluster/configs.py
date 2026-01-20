"""Configuration classes for transfer active learning"""

from typing import Literal
from pydantic.dataclasses import dataclass as pydantic_dataclass

from flc.shared.transfer_active_learning.query_strategies.config import BaseQueryStrategyConfig


@pydantic_dataclass
class BaseClusterQueryStrategyConfig(BaseQueryStrategyConfig):
    """Base configuration for cluster-based query strategies"""

    min_cluster_size: int
    exclude_noise: bool


@pydantic_dataclass
class BiggestClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for biggest cluster query strategy"""

    cluster_selection_method: Literal["largest_first", "proportional"]


@pydantic_dataclass
class SmallestClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for smallest cluster query strategy"""

    cluster_selection_method: Literal["smallest_first", "inverse_proportional"]
    max_cluster_size_ratio: float = 0.1


@pydantic_dataclass
class MostDiverseClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for most diverse cluster query strategy"""

    cluster_selection_method: Literal["most_diverse_first", "diversity_proportional"]
    diversity_metric: Literal["variance", "silhouette", "centroid_distance"]
    min_samples_for_diversity: int = 3


@pydantic_dataclass
class RandomClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for random cluster query strategy"""

    pass


@pydantic_dataclass
class HighestUncertaintyClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for highest uncertainty cluster query strategy"""

    uncertainty_measure: Literal["entropy", "margin"]


@pydantic_dataclass
class NoveltyClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for novelty cluster query strategy"""

    distance_metric: Literal["euclidean", "cosine", "manhattan"]


@pydantic_dataclass
class TotalNoveltyClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for total novelty cluster query strategy"""

    distance_metric: Literal["euclidean", "cosine", "manhattan"]


@pydantic_dataclass
class UncertaintyNoveltyClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for uncertainty + novelty hybrid cluster query strategy"""

    uncertainty_measure: Literal["entropy", "margin"]
    distance_metric: Literal["euclidean", "cosine", "manhattan"]
    novelty_type: Literal["source_only", "total"]
    uncertainty_weight: float
    novelty_weight: float


@pydantic_dataclass
class NoveltyByMedoidClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for novelty by medoid cluster query strategy"""

    distance_metric: Literal["euclidean", "cosine", "manhattan"]


@pydantic_dataclass
class TotalNoveltyByMedoidClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for total novelty by medoid cluster query strategy"""

    distance_metric: Literal["euclidean", "cosine", "manhattan"]


@pydantic_dataclass
class UncertaintyNoveltyByMedoidClusterQueryStrategyConfig(BaseClusterQueryStrategyConfig):
    """Configuration for uncertainty + novelty by medoid hybrid cluster query strategy"""

    uncertainty_measure: Literal["entropy", "margin"]
    distance_metric: Literal["euclidean", "cosine", "manhattan"]
    novelty_type: Literal["source_only", "total"]
    uncertainty_weight: float
    novelty_weight: float
