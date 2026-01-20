from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class BaseWeightingStrategyConfig:
    """Base configuration for all weighting strategies"""

    pass


@pydantic_dataclass
class UniformWeightingStrategyConfig(BaseWeightingStrategyConfig):
    """
    Configuration for the uniform weighting strategy.
    """

    pass


@pydantic_dataclass
class BalancedWeightingStrategyConfig(BaseWeightingStrategyConfig):
    """
    Configuration for the balanced weighting strategy.

    This strategy weights samples to balance the contribution of source and target datasets.
    Samples from smaller datasets receive higher weights to ensure balanced representation.
    """

    # How much the sum of weights should add up to
    source_weight_factor: float = 1.0
    target_weight_factor: float = 1.0
