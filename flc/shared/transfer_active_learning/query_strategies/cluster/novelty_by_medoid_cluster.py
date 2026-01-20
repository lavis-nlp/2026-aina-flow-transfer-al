from typing import Dict, Set
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from flc.shared.clustering import utils as clustering_utils
from .configs import NoveltyByMedoidClusterQueryStrategyConfig


class NoveltyByMedoidClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects the cluster whose medoid is furthest from all source cluster medoids.

    This strategy computes medoids for both unlabeled clusters and source-containing clusters,
    then selects the unlabeled cluster with the maximum distance to the nearest source cluster medoid.
    """

    def __init__(self, config: NoveltyByMedoidClusterQueryStrategyConfig):
        super().__init__(config)
        self.distance_metric = config.distance_metric

        # Cache for medoids to avoid recomputation
        # Cache key format: cluster_id
        self._medoid_cache: Dict[int, np.ndarray] = {}

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        cluster_assignments: Dict[int, int] = None,
        all_sample_ids: np.ndarray = None,
        all_sample_features: np.ndarray = None,
        source_sample_ids: np.ndarray = None,
        **kwargs,
    ) -> int:
        """
        Select the unlabeled cluster with the maximum distance to the nearest source cluster medoid.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)
            cluster_assignments: Dictionary mapping all sample IDs to cluster IDs
            all_sample_ids: Array of all sample IDs (source + target)
            all_sample_features: Array of all sample features (source + target)
            source_sample_ids: Array of source sample IDs
            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next

        Raises:
            ValueError: If required parameters are missing or no source-containing clusters found
        """
        # Validate required parameters
        if cluster_assignments is None:
            raise ValueError("NoveltyByMedoidClusterQueryStrategy requires cluster_assignments parameter")
        if all_sample_ids is None or all_sample_features is None:
            raise ValueError(
                "NoveltyByMedoidClusterQueryStrategy requires all_sample_ids and all_sample_features parameters"
            )
        if source_sample_ids is None:
            raise ValueError("NoveltyByMedoidClusterQueryStrategy requires source_sample_ids parameter")

        # Get unique unlabeled clusters
        unique_unlabeled_clusters = np.unique(unlabeled_sample_clusters)

        # Identify clusters that contain source samples
        source_containing_clusters = self._identify_source_containing_clusters(source_sample_ids, cluster_assignments)

        if not source_containing_clusters:
            raise ValueError("No clusters containing source samples found")

        # Compute medoids for source-containing clusters
        source_cluster_medoids = {}
        for cluster_id in source_containing_clusters:
            medoid = self._compute_cluster_medoid(cluster_id, cluster_assignments, all_sample_ids, all_sample_features)
            source_cluster_medoids[cluster_id] = medoid

        # For each unlabeled cluster, compute distance to nearest source cluster medoid
        max_min_distance = -1
        selected_cluster_id = None

        for unlabeled_cluster_id in unique_unlabeled_clusters:
            # Compute medoid for this unlabeled cluster
            unlabeled_medoid = self._compute_cluster_medoid(
                unlabeled_cluster_id, cluster_assignments, all_sample_ids, all_sample_features
            )

            # Compute distances to all source cluster medoids
            min_distance_to_source = float("inf")

            for source_cluster_id, source_medoid in source_cluster_medoids.items():
                # Compute distance between medoids
                distance = pairwise_distances(
                    unlabeled_medoid.reshape(1, -1), source_medoid.reshape(1, -1), metric=self.distance_metric
                )[0, 0]

                min_distance_to_source = min(min_distance_to_source, distance)

            # Track the cluster with the maximum minimum distance
            if min_distance_to_source > max_min_distance:
                max_min_distance = min_distance_to_source
                selected_cluster_id = unlabeled_cluster_id

        if selected_cluster_id is None:
            raise ValueError("Could not select any cluster for novelty by medoid")

        return selected_cluster_id

    def _identify_source_containing_clusters(
        self, source_sample_ids: np.ndarray, cluster_assignments: Dict[int, int]
    ) -> Set[int]:
        """
        Identify clusters that contain at least one source sample.

        Args:
            source_sample_ids: Array of source sample IDs
            cluster_assignments: Dictionary mapping all sample IDs to cluster IDs

        Returns:
            Set of cluster IDs that contain at least one source sample
        """
        source_containing_clusters = set()

        for source_sample_id in source_sample_ids:
            if source_sample_id in cluster_assignments:
                cluster_id = cluster_assignments[source_sample_id]
                source_containing_clusters.add(cluster_id)

        return source_containing_clusters

    def _compute_cluster_medoid(
        self,
        cluster_id: int,
        cluster_assignments: Dict[int, int],
        all_sample_ids: np.ndarray,
        all_sample_features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute and cache the medoid of a cluster.

        Args:
            cluster_id: ID of the cluster
            cluster_assignments: Dictionary mapping all sample IDs to cluster IDs
            all_sample_ids: Array of all sample IDs
            all_sample_features: Array of all sample features

        Returns:
            Medoid feature vector for the cluster

        Raises:
            ValueError: If cluster has no samples
        """
        # Check cache first
        if cluster_id in self._medoid_cache:
            return self._medoid_cache[cluster_id]

        # Find all samples in this cluster
        cluster_sample_indices = []
        for i, sample_id in enumerate(all_sample_ids):
            if sample_id in cluster_assignments and cluster_assignments[sample_id] == cluster_id:
                cluster_sample_indices.append(i)

        if not cluster_sample_indices:
            raise ValueError(f"Cluster {cluster_id} has no samples")

        # Get features for samples in this cluster
        cluster_features = all_sample_features[cluster_sample_indices]

        # Compute medoid using clustering utils
        medoid_idx = clustering_utils.get_medoid(cluster_features, self.distance_metric)
        medoid = cluster_features[medoid_idx]

        # Cache and return
        self._medoid_cache[cluster_id] = medoid
        return medoid

    def clear_medoid_cache(self) -> None:
        """Clear the medoid cache"""
        self._medoid_cache.clear()
