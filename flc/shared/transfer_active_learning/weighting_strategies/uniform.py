import numpy as np
from typing import Tuple
from .base import WeightingStrategy
from .configs import UniformWeightingStrategyConfig


class UniformWeightingStrategy(WeightingStrategy):
    """
    Uniform weighting strategy that assigns equal weights to all samples.

    This is the simplest baseline approach that treats all source and target
    samples with equal importance throughout the active learning process.
    """

    def __init__(self, config: UniformWeightingStrategyConfig):
        super().__init__(config)

    def compute_weights(
        self, source_features: np.ndarray, target_features: np.ndarray, iteration: int
    ) -> Tuple[float, float]:
        """
        Compute uniform weights for all samples

        Args:
            source_features: Features from source dataset
            target_features: Features from target dataset (labeled so far)
            iteration: Current iteration number (unused)

        Returns:
            Tuple of (source_weight, target_weight) with equal weights for both
        """
        total_samples = len(source_features) + len(target_features)

        if total_samples == 0:
            return (0.0, 0.0)

        # Equal weight for all samples
        uniform_weight = 1.0 / total_samples

        return (uniform_weight, uniform_weight)
