from typing import Dict, Type, Any
from .classifier.base import ClassifierQueryStrategy
from .cluster.base import ClusterQueryStrategy
from .classifier.uncertainty import UncertaintyQueryStrategy
from .classifier.random import RandomQueryStrategy
from .classifier.total_novelty import TotalNoveltyQueryStrategy
from .classifier.uncertainty_novelty import UncertaintyNoveltyQueryStrategy
from .cluster.biggest_cluster import BiggestClusterQueryStrategy
from .cluster.smallest_cluster import SmallestClusterQueryStrategy
from .cluster.most_diverse_cluster import MostDiverseClusterQueryStrategy
from .cluster.random_cluster import RandomClusterQueryStrategy
from .cluster.highest_uncertainty_cluster import HighestUncertaintyClusterQueryStrategy
from .cluster.novelty_cluster import NoveltyClusterQueryStrategy
from .cluster.novelty_by_medoid_cluster import NoveltyByMedoidClusterQueryStrategy
from .cluster.total_novelty_cluster import TotalNoveltyClusterQueryStrategy
from .cluster.total_novelty_by_medoid_cluster import TotalNoveltyByMedoidClusterQueryStrategy
from .cluster.uncertainty_novelty_cluster import UncertaintyNoveltyClusterQueryStrategy
from .cluster.uncertainty_novelty_by_medoid_cluster import UncertaintyNoveltyByMedoidClusterQueryStrategy
from .classifier.configs import (
    UncertaintyQueryStrategyConfig,
    RandomQueryStrategyConfig,
    TotalNoveltyQueryStrategyConfig,
    UncertaintyNoveltyQueryStrategyConfig,
)
from .cluster.configs import (
    BiggestClusterQueryStrategyConfig,
    SmallestClusterQueryStrategyConfig,
    MostDiverseClusterQueryStrategyConfig,
    RandomClusterQueryStrategyConfig,
    HighestUncertaintyClusterQueryStrategyConfig,
    NoveltyClusterQueryStrategyConfig,
    NoveltyByMedoidClusterQueryStrategyConfig,
    TotalNoveltyClusterQueryStrategyConfig,
    TotalNoveltyByMedoidClusterQueryStrategyConfig,
    UncertaintyNoveltyClusterQueryStrategyConfig,
    UncertaintyNoveltyByMedoidClusterQueryStrategyConfig,
)


class QueryStrategyFactory:
    """Factory for creating query strategies"""

    _classifier_strategies: Dict[str, Type[ClassifierQueryStrategy]] = {
        "uncertainty": UncertaintyQueryStrategy,
        "random": RandomQueryStrategy,
        "total_novelty": TotalNoveltyQueryStrategy,
        "uncertainty_novelty": UncertaintyNoveltyQueryStrategy,
    }

    _cluster_strategies: Dict[str, Type[ClusterQueryStrategy]] = {
        "biggest_cluster": BiggestClusterQueryStrategy,
        "smallest_cluster": SmallestClusterQueryStrategy,
        "most_diverse_cluster": MostDiverseClusterQueryStrategy,
        "random_cluster": RandomClusterQueryStrategy,
        "highest_uncertainty_cluster": HighestUncertaintyClusterQueryStrategy,
        "novelty_cluster": NoveltyClusterQueryStrategy,
        "novelty_by_medoid_cluster": NoveltyByMedoidClusterQueryStrategy,
        "total_novelty_cluster": TotalNoveltyClusterQueryStrategy,
        "total_novelty_by_medoid_cluster": TotalNoveltyByMedoidClusterQueryStrategy,
        "uncertainty_novelty_cluster": UncertaintyNoveltyClusterQueryStrategy,
        "uncertainty_novelty_by_medoid_cluster": UncertaintyNoveltyByMedoidClusterQueryStrategy,
    }

    @classmethod
    def create_classifier_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> ClassifierQueryStrategy:
        """
        Create classifier-based query strategy instance

        Args:
            strategy_name: Name of the query strategy
            config: Configuration dictionary

        Returns:
            Configured classifier query strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._classifier_strategies:
            available = list(cls._classifier_strategies.keys())
            raise ValueError(f"Unknown classifier query strategy: {strategy_name}. Available: {available}")

        # Create appropriate Pydantic config based on strategy
        if strategy_name == "uncertainty":
            pydantic_config = UncertaintyQueryStrategyConfig(**config)
        elif strategy_name == "random":
            pydantic_config = RandomQueryStrategyConfig(**config)
        elif strategy_name == "total_novelty":
            pydantic_config = TotalNoveltyQueryStrategyConfig(**config)
        elif strategy_name == "uncertainty_novelty":
            pydantic_config = UncertaintyNoveltyQueryStrategyConfig(**config)
        else:
            raise ValueError(f"Config mapping not implemented for strategy: {strategy_name}")

        strategy_class = cls._classifier_strategies[strategy_name]
        return strategy_class(pydantic_config)

    @classmethod
    def create_cluster_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> ClusterQueryStrategy:
        """
        Create cluster-based query strategy instance

        Args:
            strategy_name: Name of the query strategy
            config: Configuration dictionary

        Returns:
            Configured cluster query strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._cluster_strategies:
            available = list(cls._cluster_strategies.keys())
            raise ValueError(f"Unknown cluster query strategy: {strategy_name}. Available: {available}")

        # Create appropriate Pydantic config based on strategy
        if strategy_name == "biggest_cluster":
            pydantic_config = BiggestClusterQueryStrategyConfig(**config)
        elif strategy_name == "smallest_cluster":
            pydantic_config = SmallestClusterQueryStrategyConfig(**config)
        elif strategy_name == "most_diverse_cluster":
            pydantic_config = MostDiverseClusterQueryStrategyConfig(**config)
        elif strategy_name == "random_cluster":
            pydantic_config = RandomClusterQueryStrategyConfig(**config)
        elif strategy_name == "highest_uncertainty_cluster":
            pydantic_config = HighestUncertaintyClusterQueryStrategyConfig(**config)
        elif strategy_name == "novelty_cluster":
            pydantic_config = NoveltyClusterQueryStrategyConfig(**config)
        elif strategy_name == "novelty_by_medoid_cluster":
            pydantic_config = NoveltyByMedoidClusterQueryStrategyConfig(**config)
        elif strategy_name == "total_novelty_cluster":
            pydantic_config = TotalNoveltyClusterQueryStrategyConfig(**config)
        elif strategy_name == "total_novelty_by_medoid_cluster":
            pydantic_config = TotalNoveltyByMedoidClusterQueryStrategyConfig(**config)
        elif strategy_name == "uncertainty_novelty_cluster":
            pydantic_config = UncertaintyNoveltyClusterQueryStrategyConfig(**config)
        elif strategy_name == "uncertainty_novelty_by_medoid_cluster":
            pydantic_config = UncertaintyNoveltyByMedoidClusterQueryStrategyConfig(**config)
        else:
            raise ValueError(f"Config mapping not implemented for strategy: {strategy_name}")

        strategy_class = cls._cluster_strategies[strategy_name]
        return strategy_class(pydantic_config)

    @classmethod
    def register_classifier_strategy(cls, name: str, strategy_class: Type[ClassifierQueryStrategy]) -> None:
        """
        Register a new classifier query strategy

        Args:
            name: Name for the strategy
            strategy_class: Strategy implementation class
        """
        name = name.lower()
        cls._classifier_strategies[name] = strategy_class

    @classmethod
    def register_cluster_strategy(cls, name: str, strategy_class: Type[ClusterQueryStrategy]) -> None:
        """
        Register a new cluster query strategy

        Args:
            name: Name for the strategy
            strategy_class: Strategy implementation class
        """
        name = name.lower()
        cls._cluster_strategies[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> Dict[str, list]:
        """Get dictionary of all available query strategies"""
        return {
            "classifier_strategies": list(cls._classifier_strategies.keys()),
            "cluster_strategies": list(cls._cluster_strategies.keys()),
        }
