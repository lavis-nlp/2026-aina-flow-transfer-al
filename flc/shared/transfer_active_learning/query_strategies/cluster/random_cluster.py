from typing import List

import numpy as np

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import RandomClusterQueryStrategyConfig


class RandomClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects from randomly chosen clusters.

    Rationale: Random cluster selection provides diverse sampling across different
    regions of the target domain without bias towards cluster size or characteristics.
    """

    def __init__(self, config: RandomClusterQueryStrategyConfig):
        super().__init__(config)

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Randomly select a cluster from filtered clusters.

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

        # Randomly select from available clusters
        return self._select_random(available_clusters, self.random_state)

    @staticmethod
    def _select_random(available_clusters: List[int], random_state: int) -> int:
        """
        Randomly select a cluster from available clusters

        Args:
            available_clusters: List of available cluster IDs
            random_state: Random seed for reproducibility

        Returns:
            Randomly selected cluster ID
        """
        # make sure the seed is not always the same, otherwise it will select
        # just keep selecting after the same offset
        rng_seed = len(available_clusters) * random_state * 1000
        rng = np.random.default_rng(rng_seed)
        return rng.choice(available_clusters)
