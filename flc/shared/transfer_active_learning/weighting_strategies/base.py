from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .configs import BaseWeightingStrategyConfig


class WeightingStrategy(ABC):
    """Abstract base class for sample weighting strategies"""

    def __init__(self, config: BaseWeightingStrategyConfig):
        self.config = config

    @abstractmethod
    def compute_weights(
        self, source_features: np.ndarray, target_features: np.ndarray, iteration: int
    ) -> Tuple[float, float]:
        """
        Compute sample weights for training

        Args:
            source_features: Features from source dataset
            target_features: Features from target dataset (labeled so far)
            iteration: Current iteration number

        Returns:
            Tuple of (source_weight, target_weight) where:
            - source_weight: Weight value to assign to each source sample
            - target_weight: Weight value to assign to each target sample
        """
        pass
