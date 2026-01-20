from typing import Dict, Type, Any
from .base import WeightingStrategy
from .uniform import UniformWeightingStrategy
from .balanced import BalancedWeightingStrategy
from .configs import UniformWeightingStrategyConfig, BalancedWeightingStrategyConfig


class WeightingStrategyFactory:
    """Factory for creating weighting strategies"""

    _strategies: Dict[str, Type[WeightingStrategy]] = {
        "uniform": UniformWeightingStrategy,
        "balanced": BalancedWeightingStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str, config: Dict[str, Any]) -> WeightingStrategy:
        """
        Create weighting strategy instance

        Args:
            strategy_name: Name of the weighting strategy
            config: Configuration dictionary

        Returns:
            Configured weighting strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown weighting strategy: {strategy_name}. Available: {available}")

        # Create appropriate Pydantic config based on strategy
        if strategy_name == "uniform":
            pydantic_config = UniformWeightingStrategyConfig(**config)
        elif strategy_name == "balanced":
            pydantic_config = BalancedWeightingStrategyConfig(**config)
        else:
            raise ValueError(f"Config mapping not implemented for strategy: {strategy_name}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(pydantic_config)

    @classmethod
    def register(cls, name: str, strategy_class: Type[WeightingStrategy]) -> None:
        """
        Register a new weighting strategy

        Args:
            name: Name for the strategy
            strategy_class: Strategy implementation class
        """
        name = name.lower()
        cls._strategies[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of all available weighting strategies"""
        return list(cls._strategies.keys())