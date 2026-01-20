from abc import ABC, abstractmethod

import numpy as np


class LabelSelectionStrategy(ABC):
    """Abstract base class for selecting labels for queried samples"""

    @abstractmethod
    def select(
        self,
        sample_features: np.ndarray,
        ground_truth_labels: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Select labels for queried samples

        Args:
            sample_features: Features of the samples to label
                             Shape: (n_samples, n_features)
            ground_truth_labels: Ground-truth labels for the samples
                                 Shape: (n_samples,) for single-label or
                                        (n_samples, n_classes) for multi-label
            **kwargs: Additional context information (reserved for future extensions)
                      
        Returns:
            Hot-encoded labels for selected samples
            Shape: (n_samples, n_classes) for multi-label classification
                   or (n_samples,) for single-label classification
        """
        pass
