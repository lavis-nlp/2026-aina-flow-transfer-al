from typing import List, Dict

import numpy as np

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import SmallestClusterQueryStrategyConfig


class SmallestClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects samples from the smallest clusters.

    Rationale: Small clusters may represent edge cases, outliers, or rare patterns
    that could be important for model generalization.
    """

    def __init__(self, config: SmallestClusterQueryStrategyConfig):
        super().__init__(config)
        self.cluster_selection_method = config.cluster_selection_method
        self.max_cluster_size_ratio = config.max_cluster_size_ratio

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select the smallest cluster from filtered clusters.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)

        Returns:
            Cluster ID to label next
        """
        cluster_sizes = self._get_cluster_sizes(unlabeled_sample_clusters)
        available_clusters = list(cluster_sizes.keys())

        if not available_clusters:
            raise ValueError("No clusters available for selection")

        # Filter clusters by size ratio if specified
        if self.max_cluster_size_ratio < 1.0:
            max_cluster_size = max(cluster_sizes[cid] for cid in available_clusters)
            size_threshold = max_cluster_size * self.max_cluster_size_ratio
            available_clusters = [cid for cid in available_clusters if cluster_sizes[cid] <= size_threshold]

        if not available_clusters:
            raise ValueError("No small clusters available for selection")

        if self.cluster_selection_method == "smallest_first":
            return self._select_smallest_first(available_clusters, cluster_sizes)
        elif self.cluster_selection_method == "inverse_proportional":
            return self._select_inverse_proportional(available_clusters, cluster_sizes, self.random_state)
        else:
            raise ValueError(f"Unknown cluster_selection_method: {self.cluster_selection_method}")

    @staticmethod
    def _select_smallest_first(available_clusters: List[int], cluster_sizes: Dict[int, int]) -> int:
        """
        Select the smallest cluster from available clusters

        Args:
            available_clusters: List of available cluster IDs
            cluster_sizes: Dictionary mapping cluster ID to size

        Returns:
            Cluster ID of the smallest cluster
        """
        cluster_size_pairs = [(cid, cluster_sizes[cid]) for cid in available_clusters]
        cluster_size_pairs.sort(key=lambda x: x[1])  # ascending order
        return cluster_size_pairs[0][0]

    @staticmethod
    def _select_inverse_proportional(
        available_clusters: List[int], cluster_sizes: Dict[int, int], random_state: int
    ) -> int:
        """
        Select cluster inverse proportional to size (smaller clusters have higher probability)

        Args:
            available_clusters: List of available cluster IDs
            cluster_sizes: Dictionary mapping cluster ID to size
            random_state: Random seed for reproducibility

        Returns:
            Cluster ID selected inverse proportional to size
        """
        cluster_weights = [1.0 / cluster_sizes[cid] for cid in available_clusters]
        total_weight = sum(cluster_weights)
        probabilities = [w / total_weight for w in cluster_weights]

        rng = np.random.default_rng(random_state)
        return rng.choice(available_clusters, p=probabilities)
