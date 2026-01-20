import numpy as np
from typing import Tuple
from .base import WeightingStrategy
from .configs import BalancedWeightingStrategyConfig


class BalancedWeightingStrategy(WeightingStrategy):
    """
    Balanced weighting strategy that adjusts sample weights to balance the contribution
    of source and target datasets.

    This strategy computes weights inversely proportional to dataset sizes, ensuring that
    smaller datasets (typically target) receive higher per-sample weights to balance their
    overall contribution during training. This helps prevent the larger source dataset from
    dominating the learning process.

    The weighting formula ensures that:
    - Source samples get weight: (source_factor * target_size) / total_effective_size
    - Target samples get weight: (target_factor * source_size) / total_effective_size

    This creates balanced contribution when source_factor == target_factor.
    """

    def __init__(self, config: BalancedWeightingStrategyConfig):
        super().__init__(config)

    def compute_weights(
        self, source_features: np.ndarray, target_features: np.ndarray, iteration: int
    ) -> Tuple[float, float]:
        """
        Compute balanced weights for source and target samples

        Args:
            source_features: Features from source dataset (n_source_samples, n_features)
            target_features: Features from target dataset (n_target_samples, n_features)
            iteration: Current iteration number (unused in this strategy)

        Returns:
            Tuple of (source_weight, target_weight) where weights are inversely proportional
            to dataset sizes, modified by the configured factors
        """
        n_source = len(source_features)
        n_target = len(target_features)

        # Handle edge cases
        if n_source == 0 and n_target == 0:
            return (0.0, 0.0)
        elif n_source == 0:
            return (0.0, self.config.target_weight_factor)
        elif n_target == 0:
            return (self.config.source_weight_factor, 0.0)

        # Compute balanced weights
        # The idea is that the fraction of weights should balance the contribution

        # if source_weight_factor is 0.5, the total weight (sum of all source sample weights) will be 0.5
        source_sample_weight = self.config.source_weight_factor / n_source
        target_sample_weight = self.config.target_weight_factor / n_target

        return (
            source_sample_weight,
            target_sample_weight,
        )
