import numpy as np
from .base import LabelSelectionStrategy
from .configs import SimulatedSelectionStrategyConfig


class SimulatedSelectionStrategy(LabelSelectionStrategy):
    """
    Simulated label selection strategy that uses ground truth labels.

    This strategy simulates perfect oracle labeling by looking up ground truth labels
    from the target dataset.
    """

    def __init__(self, config: SimulatedSelectionStrategyConfig):
        self._config = config

    def select(
        self,
        sample_features: np.ndarray,
        ground_truth_labels: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Assigns ground truth labels to every queried sample.

        Args:
            sample_features: Features of the samples to label
                             Shape: (n_samples, n_features)
            ground_truth_labels: Ground-truth labels for the samples
                                 Shape: (n_samples,) for single-label or
                                        (n_samples, n_classes) for multi-label
        Returns:
            Hot-encoded labels for selected samples
            Shape: (n_samples, n_classes) for multi-label classification
                   or (n_samples,) for single-label classification
        """

        return ground_truth_labels
