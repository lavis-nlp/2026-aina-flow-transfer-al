from typing import Dict, Type, Any
from .base import LabelSelectionStrategy
from .random import RandomSelectionStrategy
from .simulated import SimulatedSelectionStrategy
from .medoid import MedoidSelectionStrategy
from .configs import (
    RandomSelectionStrategyConfig,
    MedoidSelectionStrategyConfig,
    SimulatedSelectionStrategyConfig,
)


class LabelSelectionFactory:
    """Factory for creating label selection strategies"""

    _strategies: Dict[str, Type[LabelSelectionStrategy]] = {
        "random": RandomSelectionStrategy,
        "simulated": SimulatedSelectionStrategy,
        "medoid": MedoidSelectionStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str, config: Dict[str, Any]) -> LabelSelectionStrategy:
        """
        Create label selection strategy instance

        Args:
            strategy_name: Name of the label selection strategy
            config: Configuration dictionary

        Returns:
            Configured label selection strategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown label selection strategy: {strategy_name}. Available: {available}")

        # Create appropriate Pydantic config based on strategy
        if strategy_name == "random":
            pydantic_config = RandomSelectionStrategyConfig(**config)
        elif strategy_name == "simulated":
            pydantic_config = SimulatedSelectionStrategyConfig(**config)
        elif strategy_name == "medoid":
            pydantic_config = MedoidSelectionStrategyConfig(**config)
        else:
            raise ValueError(f"Config mapping not implemented for strategy: {strategy_name}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(pydantic_config)

    @classmethod
    def register(cls, name: str, strategy_class: Type[LabelSelectionStrategy]) -> None:
        """
        Register a new label selection strategy

        Args:
            name: Name for the strategy
            strategy_class: Strategy implementation class
        """
        name = name.lower()
        cls._strategies[name] = strategy_class

    @classmethod
    def get_available_strategies(cls) -> list:
        """Get list of all available label selection strategies"""
        return list(cls._strategies.keys())