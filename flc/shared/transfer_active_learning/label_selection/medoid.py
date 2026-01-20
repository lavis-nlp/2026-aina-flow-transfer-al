import numpy as np
from sklearn.metrics import pairwise_distances
from .base import LabelSelectionStrategy
from .configs import MedoidSelectionStrategyConfig


class MedoidSelectionStrategy(LabelSelectionStrategy):
    """
    Medoid-based label selection strategy for clustering-based active learning.

    This strategy finds the medoid (most representative sample) within the selected
    cluster by computing the sample that has the minimum average distance to all
    other samples in the cluster, or alternatively, the sample closest to the
    cluster centroid. All samples in the cluster are then assigned the ground
    truth label of the medoid sample.

    The medoid approach ensures that the selected label source is actually a
    real sample from the dataset, which can be more robust than using artificial
    centroids for label assignment.
    """

    def __init__(self, config: MedoidSelectionStrategyConfig):
        self._config = config

    def select(
        self,
        sample_features: np.ndarray,
        ground_truth_labels: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Assigns the ground truth label of the medoid sample to all samples.

        The medoid is computed as the sample that minimizes the sum of distances
        to all other samples in the cluster, or alternatively, the sample closest
        to the cluster centroid if more efficient.

        Args:
            sample_features: Features of the samples to label
                             Shape: (n_samples, n_features)
            ground_truth_labels: Ground-truth labels for the samples
                                 Shape: (n_samples,) for single-label or
                                        (n_samples, n_classes) for multi-label

        Returns:
            Labels for all samples, all set to the medoid's ground truth label
            Shape: (n_samples, n_classes) for multi-label classification
                   or (n_samples,) for single-label classification
        """
        if len(sample_features) == 0:
            return np.array([])

        # If only one sample, return its ground truth label
        if len(sample_features) == 1:
            return ground_truth_labels.copy()

        # Find the medoid sample
        medoid_idx = self._find_medoid(sample_features)

        # Get the medoid's ground truth label
        medoid_label = ground_truth_labels[medoid_idx]

        # Assign the medoid's label to all samples in the cluster
        if ground_truth_labels.ndim == 1:
            # Single-label case
            assigned_labels = np.full(len(sample_features), medoid_label, dtype=ground_truth_labels.dtype)
        else:
            # Multi-label case
            assigned_labels = np.tile(medoid_label, (len(sample_features), 1))

        return assigned_labels

    def _find_medoid(self, sample_features: np.ndarray) -> int:
        """
        Find the medoid sample within the given samples.

        The medoid is the sample that minimizes the sum of distances to all
        other samples in the cluster. For efficiency with large clusters,
        we use the sample closest to the centroid as an approximation.

        Args:
            sample_features: Features of samples in the cluster
                           Shape: (n_samples, n_features)

        Returns:
            Index of the medoid sample
        """
        if len(sample_features) == 1:
            return 0

        # Method 1: Find sample closest to centroid (efficient approximation)
        if len(sample_features) > 50:  # Use centroid method for large clusters
            return self._find_medoid_via_centroid(sample_features)

        # Method 2: True medoid computation (for smaller clusters)
        return self._find_true_medoid(sample_features)

    def _find_medoid_via_centroid(self, sample_features: np.ndarray) -> int:
        """
        Find medoid as the sample closest to the cluster centroid.

        This is an efficient approximation that works well in practice.
        The centroid is computed from the sample features directly.

        Args:
            sample_features: Features of samples in the cluster
                           Shape: (n_samples, n_features)

        Returns:
            Index of the sample closest to centroid
        """
        # Compute cluster centroid from the sample features
        centroid = np.mean(sample_features, axis=0, keepdims=True)

        # Compute distances from all samples to centroid
        distances = pairwise_distances(sample_features, centroid, metric=self._config.distance_metric).flatten()

        # Return index of sample with minimum distance to centroid
        return np.argmin(distances)

    def _find_true_medoid(self, sample_features: np.ndarray) -> int:
        """
        Find the true medoid by computing pairwise distances.

        The medoid is the sample that minimizes the sum of distances to all
        other samples in the cluster.

        Args:
            sample_features: Features of samples in the cluster
                           Shape: (n_samples, n_features)

        Returns:
            Index of the true medoid sample
        """
        # Compute pairwise distances between all samples
        distance_matrix = pairwise_distances(sample_features, metric=self._config.distance_metric)

        # Sum distances for each sample to all other samples
        distance_sums = np.sum(distance_matrix, axis=1)

        # Return index of sample with minimum total distance
        return np.argmin(distance_sums)
