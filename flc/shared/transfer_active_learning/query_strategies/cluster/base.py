from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from flc.shared.transfer_active_learning.query_strategies.cluster.configs import BaseClusterQueryStrategyConfig


class ClusterQueryStrategy(ABC):
    """Abstract base class for cluster-based query strategies"""

    def __init__(self, config: BaseClusterQueryStrategyConfig):
        self.config = config
        self.random_state = config.random_state

    def select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select next cluster to label.
        Applies filtering based on config parameters and then delegates to _select_cluster.

        Args:
            unlabeled_sample_ids: IDs of unlabeled samples (n_unlabeled_samples,)
            unlabeled_sample_features: Features of unlabeled samples (n_unlabeled_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for unlabeled samples (n_unlabeled_samples,)
            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next
        """
        # Filter clusters based on config parameters
        filtered_ids, filtered_features, filtered_clusters = self.get_selectable_clusters(
            unlabeled_sample_ids, unlabeled_sample_features, unlabeled_sample_clusters
        )

        # Delegate to strategy-specific implementation
        return self._select_cluster(
            unlabeled_sample_ids=filtered_ids,
            unlabeled_sample_features=filtered_features,
            unlabeled_sample_clusters=filtered_clusters,
            **kwargs,
        )

    @abstractmethod
    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select next cluster to label from filtered data.
        This method contains the strategy-specific logic for cluster selection.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)
            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next
        """
        pass

    def get_selectable_clusters(
        self, sample_ids: np.ndarray, features: np.ndarray, cluster_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter clusters based on min_cluster_size and exclude_noise parameters.

        Args:
            sample_ids: Sample IDs (n_samples,)
            features: Feature array (n_samples, n_features)
            cluster_labels: Cluster labels for samples (n_samples,)

        Returns:
            Tuple of (filtered_sample_ids, filtered_features, filtered_cluster_labels)
        """
        # Apply noise filtering
        sample_ids, features, cluster_labels = self._filter_noise_clusters(sample_ids, features, cluster_labels)

        # Apply size filtering
        sample_ids, features, cluster_labels = self._filter_small_clusters(sample_ids, features, cluster_labels)

        return sample_ids, features, cluster_labels

    def _filter_noise_clusters(
        self, sample_ids: np.ndarray, features: np.ndarray, cluster_labels: np.ndarray, noise_label: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out noise clusters (label -1) if exclude_noise is True.

        Args:
            sample_ids: Sample IDs (n_samples,)
            features: Feature array (n_samples, n_features)
            cluster_labels: Cluster labels for samples (n_samples,)

        Returns:
            Tuple of (filtered_sample_ids, filtered_features, filtered_cluster_labels)
        """
        if not self.config.exclude_noise:
            return sample_ids, features, cluster_labels

        mask = cluster_labels != noise_label
        return sample_ids[mask], features[mask], cluster_labels[mask]

    def _filter_small_clusters(
        self, sample_ids: np.ndarray, features: np.ndarray, cluster_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out clusters smaller than min_cluster_size.

        Args:
            sample_ids: Sample IDs (n_samples,)
            features: Feature array (n_samples, n_features)
            cluster_labels: Cluster labels for samples (n_samples,)

        Returns:
            Tuple of (filtered_sample_ids, filtered_features, filtered_cluster_labels)
        """
        cluster_sizes = self._get_cluster_sizes(cluster_labels)

        # Create mask for clusters that meet minimum size requirement
        mask = np.ones(len(cluster_labels), dtype=bool)
        for cluster_id, size in cluster_sizes.items():
            if size < self.config.min_cluster_size:
                mask &= cluster_labels != cluster_id

        return sample_ids[mask], features[mask], cluster_labels[mask]

    def _get_cluster_sizes(self, sample_clusters: np.ndarray) -> Dict[int, int]:
        """
        Get cluster sizes from sample cluster labels

        Args:
            sample_clusters: Cluster labels for samples

        Returns:
            Dictionary mapping cluster ID to size
        """
        unique_clusters, counts = np.unique(sample_clusters, return_counts=True)
        return dict(zip(unique_clusters.tolist(), counts.tolist()))
