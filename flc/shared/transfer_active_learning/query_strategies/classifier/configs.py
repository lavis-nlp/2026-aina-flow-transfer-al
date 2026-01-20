from typing import Literal

from pydantic.dataclasses import dataclass as pydantic_dataclass

from flc.shared.transfer_active_learning.query_strategies.config import BaseQueryStrategyConfig


@pydantic_dataclass
class BaseClassifierQueryStrategyConfig(BaseQueryStrategyConfig):
    """Base configuration for classifier-based query strategies"""

    pass


@pydantic_dataclass
class UncertaintyQueryStrategyConfig(BaseQueryStrategyConfig):
    """Configuration for uncertainty-based query strategy"""

    uncertainty_measure: Literal["entropy", "margin"] = "entropy"


@pydantic_dataclass
class RandomQueryStrategyConfig(BaseQueryStrategyConfig):
    """Configuration for random query strategy"""

    pass


@pydantic_dataclass
class TotalNoveltyQueryStrategyConfig(BaseQueryStrategyConfig):
    """Configuration for total novelty query strategy"""

    distance_metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"


@pydantic_dataclass
class UncertaintyNoveltyQueryStrategyConfig(BaseQueryStrategyConfig):
    """Configuration for uncertainty + novelty hybrid query strategy"""

    uncertainty_measure: Literal["entropy", "margin"] = "entropy"
    distance_metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"
    novelty_type: Literal["source_only", "total"] = "total"
    uncertainty_weight: float = 0.5
    novelty_weight: float = 0.5
