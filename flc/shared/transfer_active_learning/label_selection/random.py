import numpy as np
from .base import LabelSelectionStrategy
from .configs import RandomSelectionStrategyConfig


class RandomSelectionStrategy(LabelSelectionStrategy):
    """
    Random label selection strategy for clustering-based active learning.

    This strategy selects a random sample from the cluster and assigns
    its ground truth label to all samples in the cluster. This provides
    a simple baseline for label selection that doesn't rely on any
    sophisticated selection criteria.
    """

    def __init__(self, config: RandomSelectionStrategyConfig):
        self._config = config
        self._rng = np.random.RandomState(config.random_state)

    def select(
        self,
        sample_features: np.ndarray,
        ground_truth_labels: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Assigns the ground truth label of a randomly selected sample to all samples.

        Args:
            sample_features: Features of the samples to label
                             Shape: (n_samples, n_features)
            ground_truth_labels: Ground-truth labels for the samples
                                 Shape: (n_samples,) for single-label or
                                        (n_samples, n_classes) for multi-label

        Returns:
            Labels for all samples, all set to the randomly selected sample's ground truth label
            Shape: (n_samples, n_classes) for multi-label classification
                   or (n_samples,) for single-label classification
        """
        if len(sample_features) == 0:
            return np.array([])

        # If only one sample, return its ground truth label
        if len(sample_features) == 1:
            return ground_truth_labels.copy()

        # Randomly select a sample
        random_idx = self._rng.choice(len(sample_features))

        # Get the random sample's ground truth label
        random_label = ground_truth_labels[random_idx].copy()

        # Assign the random sample's label to all samples in the cluster
        if ground_truth_labels.ndim == 1:
            # Single-label case
            assigned_labels = np.full(len(sample_features), random_label, dtype=ground_truth_labels.dtype)
        else:
            # Multi-label case
            assigned_labels = np.tile(random_label, (len(sample_features), 1))

        return assigned_labels
