from typing import Literal

from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class BaseWeightingStrategyConfig:
    """Base configuration for all weighting strategies"""

    random_state: int


@pydantic_dataclass
class RandomSelectionStrategyConfig(BaseWeightingStrategyConfig):
    """
    Configuration for the random label selection strategy.

    The random strategy selects a random sample from the cluster and assigns
    its ground truth label to all samples in the cluster.
    """

    pass


@pydantic_dataclass
class SimulatedSelectionStrategyConfig(BaseWeightingStrategyConfig):
    """
    Configuration for the simulated label selection strategy.

    """

    pass


@pydantic_dataclass
class MedoidSelectionStrategyConfig(BaseWeightingStrategyConfig):
    """
    Configuration for the medoid label selection strategy.

    The medoid strategy finds the most representative sample (medoid) within
    the selected cluster and assigns its ground truth label to all samples
    in the cluster.
    """

    distance_metric: Literal["euclidean", "cosine", "manhattan"]  # Distance metric for medoid computation
