from typing import Dict
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import NoveltyClusterQueryStrategyConfig


class NoveltyClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects the cluster containing the target sample most far away from all source samples.
    """

    def __init__(self, config: NoveltyClusterQueryStrategyConfig):
        super().__init__(config)
        self.distance_metric = config.distance_metric

        # Cache for sample distances to avoid recomputation
        # Cache key format: {target_sample_id}---{source_sample_id}
        self._distance_cache: Dict[str, float] = {}

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        source_sample_ids: np.ndarray = None,
        source_features: np.ndarray = None,
        **kwargs,
    ) -> int:
        """
        Select the target cluster containing the sample furthest from all source samples.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)
            source_sample_ids: Source sample IDs
            source_features: Source sample features
            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next
        """
        if source_features is None or len(source_features) == 0:
            raise ValueError("NoveltyClusterQueryStrategy requires source samples to compute distances")

        # Find the unlabeled sample with the highest distance to the nearest source sample
        max_min_distance = -1
        selected_sample_cluster = None

        for i, (sample_id, sample_features) in enumerate(zip(unlabeled_sample_ids, unlabeled_sample_features)):
            # Compute minimum distance from this unlabeled sample to any source sample
            min_distance_to_source = float('inf')
            
            for j, source_sample_id in enumerate(source_sample_ids):
                # Check cache first
                cache_key = f"{sample_id}---{source_sample_id}"
                if cache_key in self._distance_cache:
                    distance = self._distance_cache[cache_key]
                else:
                    # Compute and cache distance
                    distance = pairwise_distances(
                        sample_features.reshape(1, -1), 
                        source_features[j].reshape(1, -1), 
                        metric=self.distance_metric
                    )[0, 0]
                    self._distance_cache[cache_key] = distance
                
                min_distance_to_source = min(min_distance_to_source, distance)
            
            # Track the sample with the maximum minimum distance
            if min_distance_to_source > max_min_distance:
                max_min_distance = min_distance_to_source
                selected_sample_cluster = unlabeled_sample_clusters[i]

        if selected_sample_cluster is None:
            raise ValueError("Could not select any cluster for novelty")

        return selected_sample_cluster

