from typing import Dict
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import TotalNoveltyClusterQueryStrategyConfig


class TotalNoveltyClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects the cluster containing the target sample most far away from all annotated samples
    (both source samples and previously annotated target samples).
    """

    def __init__(self, config: TotalNoveltyClusterQueryStrategyConfig):
        super().__init__(config)
        self.distance_metric = config.distance_metric

        # Cache for sample distances to avoid recomputation
        # Cache key format: {target_sample_id}---{annotated_sample_id}
        self._distance_cache: Dict[str, float] = {}

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        annotated_sample_ids: np.ndarray = None,
        annotated_features: np.ndarray = None,
        **kwargs,
    ) -> int:
        """
        Select the target cluster containing the sample furthest from all annotated samples.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)
            annotated_sample_ids: All annotated sample IDs (source + labeled target)
            annotated_features: All annotated sample features (source + labeled target)
            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next
        """
        if annotated_features is None or len(annotated_features) == 0:
            raise ValueError("TotalNoveltyClusterQueryStrategy requires annotated samples to compute distances")

        # Find the unlabeled sample with the highest distance to the nearest annotated sample
        max_min_distance = -1
        selected_sample_cluster = None

        for i, (sample_id, sample_features) in enumerate(zip(unlabeled_sample_ids, unlabeled_sample_features)):
            # Compute minimum distance from this unlabeled sample to any annotated sample
            min_distance_to_annotated = float('inf')
            
            for j, annotated_sample_id in enumerate(annotated_sample_ids):
                # Check cache first
                cache_key = f"{sample_id}---{annotated_sample_id}"
                if cache_key in self._distance_cache:
                    distance = self._distance_cache[cache_key]
                else:
                    # Compute and cache distance
                    distance = pairwise_distances(
                        sample_features.reshape(1, -1), 
                        annotated_features[j].reshape(1, -1), 
                        metric=self.distance_metric
                    )[0, 0]
                    self._distance_cache[cache_key] = distance
                
                min_distance_to_annotated = min(min_distance_to_annotated, distance)
            
            # Track the sample with the maximum minimum distance
            if min_distance_to_annotated > max_min_distance:
                max_min_distance = min_distance_to_annotated
                selected_sample_cluster = unlabeled_sample_clusters[i]

        if selected_sample_cluster is None:
            raise ValueError("Could not select any cluster for total novelty")

        return selected_sample_cluster