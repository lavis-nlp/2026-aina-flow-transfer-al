from typing import List, Dict
import numpy as np
from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import BiggestClusterQueryStrategyConfig


class BiggestClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects samples from the largest clusters.

    Rationale: Large clusters likely contain representative samples that can
    provide good coverage of the target domain.
    """

    def __init__(self, config: BiggestClusterQueryStrategyConfig):
        super().__init__(config)
        self.cluster_selection_method = config.cluster_selection_method

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select the largest cluster from filtered clusters.

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

        if self.cluster_selection_method == "largest_first":
            return self._select_largest_first(available_clusters, cluster_sizes)
        elif self.cluster_selection_method == "proportional":
            return self._select_proportional(available_clusters, cluster_sizes, self.random_state)
        else:
            raise ValueError(f"Unknown cluster_selection_method: {self.cluster_selection_method}")

    @staticmethod
    def _select_largest_first(available_clusters: List[int], cluster_sizes: Dict[int, int]) -> int:
        """
        Select the largest cluster from available clusters

        Args:
            available_clusters: List of available cluster IDs
            cluster_sizes: Dictionary mapping cluster ID to size

        Returns:
            Cluster ID of the largest cluster
        """
        cluster_size_pairs = [(cid, cluster_sizes[cid]) for cid in available_clusters]
        cluster_size_pairs.sort(key=lambda x: x[1], reverse=True)
        return cluster_size_pairs[0][0]

    @staticmethod
    def _select_proportional(available_clusters: List[int], cluster_sizes: Dict[int, int], random_state: int) -> int:
        """
        Select cluster proportional to size (larger clusters have higher probability)

        Args:
            available_clusters: List of available cluster IDs
            cluster_sizes: Dictionary mapping cluster ID to size
            random_state: Random seed for reproducibility

        Returns:
            Cluster ID selected proportional to size
        """
        cluster_weights = [cluster_sizes[cid] for cid in available_clusters]
        total_weight = sum(cluster_weights)
        probabilities = [w / total_weight for w in cluster_weights]

        rng = np.random.default_rng(random_state)
        return rng.choice(available_clusters, p=probabilities)
