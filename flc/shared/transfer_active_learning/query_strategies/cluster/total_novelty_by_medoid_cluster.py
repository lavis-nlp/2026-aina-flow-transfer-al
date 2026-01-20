from typing import Dict, Set, Optional
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from flc.shared.clustering import utils as clustering_utils
from .configs import TotalNoveltyByMedoidClusterQueryStrategyConfig


class TotalNoveltyByMedoidClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects the cluster whose medoid is furthest from all annotated cluster medoids.

    This strategy works with separate clustering of source and target samples. It computes medoids
    for annotated clusters (all source clusters + labeled target clusters) and unlabeled target clusters,
    then selects the unlabeled target cluster with the maximum distance to the nearest annotated cluster medoid.
    Cross-domain distances are computed between target cluster medoids and source cluster medoids.
    """

    def __init__(self, config: TotalNoveltyByMedoidClusterQueryStrategyConfig):
        super().__init__(config)
        self.distance_metric = config.distance_metric

        # Cache for storing computed medoids
        # {cluster_id: medoid_sample_id}
        self._source_medoid_cache: Dict[int, str] = {}
        self._target_medoid_cache: Dict[int, str] = {}

        # Cache for storing distances between medoids
        # Key format: "sample_id1---sample_id2---distance_metric"
        self._medoid_distance_cache: Dict[str, float] = {}

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        cluster2source_ids: Dict[int, Set[str]],
        cluster2target_ids: Dict[int, Set[str]],
        source_id2features: dict[str, np.ndarray],
        target_id2features: dict[str, np.ndarray],
        **kwargs,
    ) -> int:
        """
        Select the unlabeled cluster with the maximum distance to the nearest annotated cluster medoid.

        Args:
            cluster2source_ids: Mapping of source cluster IDs to sample IDs
            cluster2target_ids: Mapping of target cluster IDs to sample IDs
            source_id2features: Dictionary mapping source sample IDs to feature vectors
            target_id2features: Dictionary mapping target sample IDs to feature vectors
            unlabeled_sample_ids: Array of unlabeled target sample IDs

            **kwargs: Additional arguments

        Returns:
            Cluster ID to label next

        Raises:
            ValueError: If required parameters are missing or no annotated clusters found
        """

        # Validate required parameters
        if cluster2source_ids is None or cluster2target_ids is None:
            raise ValueError(
                "TotalNoveltyByMedoidClusterQueryStrategy requires cluster2source_ids and cluster2target_ids parameters"
            )
        if source_id2features is None or target_id2features is None:
            raise ValueError(
                "TotalNoveltyByMedoidClusterQueryStrategy requires source_id2features and target_id2features parameters"
            )
        if unlabeled_sample_ids is None:
            raise ValueError("TotalNoveltyByMedoidClusterQueryStrategy requires unlabeled_sample_ids parameter")
        if len(unlabeled_sample_ids) == 0:
            raise ValueError("No unlabeled samples provided")

        # Convert array to set for efficient lookup
        unlabeled_target_ids = set(unlabeled_sample_ids)

        # Validate non-empty data
        # if not cluster2source_ids:
        #     raise ValueError("No source clusters provided")
        if not cluster2target_ids:
            raise ValueError("No target clusters provided")
        # if not source_id2features:
        #     raise ValueError("No source features provided")
        if not target_id2features:
            raise ValueError("No target features provided")

        # Check for empty clusters
        empty_source_clusters = [cid for cid, samples in cluster2source_ids.items() if not samples]
        empty_target_clusters = [cid for cid, samples in cluster2target_ids.items() if not samples]
        if empty_source_clusters and empty_target_clusters:
            raise ValueError(
                f"Both source and target clusters are empty: source={empty_source_clusters}, target={empty_target_clusters}"
            )

        # remove noise clusters
        if self.config.exclude_noise:
            cluster2source_ids = {cid: samples for cid, samples in cluster2source_ids.items() if cid != -1}
            cluster2target_ids = {cid: samples for cid, samples in cluster2target_ids.items() if cid != -1}

        # Get medoids for source and target clusters
        source_cluster2medoid_id = self._get_medoids("source", cluster2source_ids, source_id2features)
        target_cluster2medoid_id = self._get_medoids("target", cluster2target_ids, target_id2features)

        # Find already annotated target clusters
        annotated_target_clusters = TotalNoveltyByMedoidClusterQueryStrategy._get_annotated_clusters_from_unlabeled(
            unlabeled_ids=unlabeled_target_ids, cluster2sample_ids=cluster2target_ids
        )

        # Get all annotated cluster medoid sample IDs (source + labeled target clusters)
        annotated_medoid_sample_ids = []

        # Add all source cluster medoids
        for cluster_id, medoid_sample_id in source_cluster2medoid_id.items():
            annotated_medoid_sample_ids.append(medoid_sample_id)

        # Add medoids of labeled target clusters
        for cluster_id, medoid_sample_id in target_cluster2medoid_id.items():
            if cluster_id in annotated_target_clusters:
                annotated_medoid_sample_ids.append(medoid_sample_id)

        if not annotated_medoid_sample_ids:
            raise ValueError("No annotated cluster medoids found")

        # For each unlabeled target cluster, compute distance to nearest annotated cluster medoid
        max_min_distance = -1
        selected_cluster_id = None

        unlabeled_target_clusters = set(cluster2target_ids.keys()) - annotated_target_clusters

        for unlabeled_cluster_id in unlabeled_target_clusters:
            if unlabeled_cluster_id not in target_cluster2medoid_id:
                raise ValueError(f"No medoid found for unlabeled target cluster {unlabeled_cluster_id}")

            unlabeled_medoid_sample_id = target_cluster2medoid_id[unlabeled_cluster_id]
            unlabeled_medoid_features = target_id2features[unlabeled_medoid_sample_id]

            # Find distance to nearest annotated cluster medoid using caching
            min_distance_to_annotated = float("inf")

            for annotated_medoid_sample_id in annotated_medoid_sample_ids:
                # Get features for annotated medoid - need to check both source and target
                if annotated_medoid_sample_id in source_id2features:
                    annotated_medoid_features = source_id2features[annotated_medoid_sample_id]
                elif annotated_medoid_sample_id in target_id2features:
                    annotated_medoid_features = target_id2features[annotated_medoid_sample_id]
                else:
                    raise ValueError(f"Features not found for annotated medoid sample {annotated_medoid_sample_id}")

                # Use cached distance computation
                distance = self._compute_medoid_distance(
                    unlabeled_medoid_sample_id,
                    annotated_medoid_sample_id,
                    unlabeled_medoid_features,
                    annotated_medoid_features,
                )
                min_distance_to_annotated = min(min_distance_to_annotated, distance)

            # Track the cluster with maximum distance to nearest annotated medoid
            if min_distance_to_annotated > max_min_distance:
                max_min_distance = min_distance_to_annotated
                selected_cluster_id = unlabeled_cluster_id

        if selected_cluster_id is None:
            raise ValueError("No unlabeled target clusters available for selection")

        return selected_cluster_id

    def _get_medoids(self, typ, cluster2ids, id2features):
        """
        Get medoids for the specified type (source or target) clusters. Uses cached medoids if available.

        Args:
            typ: Type of clusters ('source' or 'target')
            cluster2ids: Mapping of cluster IDs to sample IDs
            id2features: Dictionary mapping sample IDs to feature vectors

        Returns:
            Dictionary mapping cluster IDs to their medoid sample IDs
        """
        medoids = {}

        for cluster_id, sample_ids in cluster2ids.items():
            if self._get_cached_medoid(typ, cluster_id) is None:
                # Compute medoid if not cached
                sample_ids_list = list(sample_ids)  # Convert set to list for indexing

                # Check that all sample IDs have features
                missing_features = [sid for sid in sample_ids_list if sid not in id2features]
                if missing_features:
                    raise ValueError(f"Missing features for {typ} samples: {missing_features}")

                cluster_features = np.array([id2features[sample_id] for sample_id in sample_ids_list])
                medoid_idx = clustering_utils.get_medoid(cluster_features, self.distance_metric)
                medoid_sample_id = sample_ids_list[medoid_idx]  # Get actual sample ID
                self._set_cached_medoid(typ, cluster_id, medoid_sample_id)

            medoid = self._get_cached_medoid(typ, cluster_id)
            assert medoid is not None
            medoids[cluster_id] = medoid

        return medoids

    def _get_cached_distance(self, sample_id1: str, sample_id2: str) -> Optional[float]:
        """
        Get cached distance between two medoid sample IDs.

        Args:
            sample_id1: First medoid sample ID
            sample_id2: Second medoid sample ID

        Returns:
            Cached distance or None if not found
        """
        cache_key = f"{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        return self._medoid_distance_cache.get(cache_key, None)

    def _set_cached_distance(self, sample_id1: str, sample_id2: str, distance: float) -> None:
        """
        Set cached distance between two medoid sample IDs.

        Args:
            sample_id1: First medoid sample ID
            sample_id2: Second medoid sample ID
            distance: Distance value to cache
        """
        cache_key = f"{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        self._medoid_distance_cache[cache_key] = distance

    def _compute_medoid_distance(
        self, sample_id1: str, sample_id2: str, features1: np.ndarray, features2: np.ndarray
    ) -> float:
        """
        Compute distance between two medoids with caching.

        Args:
            sample_id1: First medoid sample ID
            sample_id2: Second medoid sample ID
            features1: Feature vector of first medoid
            features2: Feature vector of second medoid

        Returns:
            Distance between the medoids
        """
        # Check cache first
        cached_distance = self._get_cached_distance(sample_id1, sample_id2)
        if cached_distance is not None:
            return cached_distance

        # Compute distance
        distance = pairwise_distances(features1.reshape(1, -1), features2.reshape(1, -1), metric=self.distance_metric)[
            0, 0
        ]

        # Cache and return
        self._set_cached_distance(sample_id1, sample_id2, distance)
        return distance

    def _get_cached_medoid(self, typ, cluster_id):
        """
        Get cached medoid for the specified cluster type and ID.

        Args:
            typ: Type of clusters ('source' or 'target')
            cluster_id: ID of the cluster

        Returns:
            Cached medoid sample ID or None if not found
        """
        if typ == "source":
            return self._source_medoid_cache.get(cluster_id, None)
        elif typ == "target":
            return self._target_medoid_cache.get(cluster_id, None)
        else:
            raise ValueError(f"Unknown cluster type: {typ}")

    def _set_cached_medoid(self, typ, cluster_id, medoid_id):
        """
        Set cached medoid for the specified cluster type and ID.

        Args:
            typ: Type of clusters ('source' or 'target')
            cluster_id: ID of the cluster
            medoid_id: Sample ID of the medoid
        """
        if typ == "source":
            self._source_medoid_cache[cluster_id] = medoid_id
        elif typ == "target":
            self._target_medoid_cache[cluster_id] = medoid_id
        else:
            raise ValueError(f"Unknown cluster type: {typ}")

    def clear_medoid_cache(self) -> None:
        """Clear both source and target medoid caches"""
        self._source_medoid_cache.clear()
        self._target_medoid_cache.clear()

    def clear_distance_cache(self) -> None:
        """Clear the medoid distance cache"""
        self._medoid_distance_cache.clear()

    def clear_all_caches(self) -> None:
        """Clear all caches (medoids and distances)"""
        self.clear_medoid_cache()
        self.clear_distance_cache()

    @staticmethod
    def _get_annotated_clusters(labeled_ids: Set[str], cluster2sample_ids: Dict[int, Set[str]]) -> Set[int]:
        annotated = set()
        for cluster_id, sample_ids in cluster2sample_ids.items():
            if labeled_ids.intersection(sample_ids):

                # sanity check: all samples of cluster should be labeled
                assert sample_ids.issubset(labeled_ids), f"Cluster {cluster_id} has unlabeled samples"

                annotated.add(cluster_id)

        return annotated

    @staticmethod
    def _get_annotated_clusters_from_unlabeled(
        unlabeled_ids: Set[str], cluster2sample_ids: Dict[int, Set[str]]
    ) -> Set[int]:
        """
        Get annotated (fully labeled) clusters based on unlabeled sample IDs.
        A cluster is considered annotated if it has NO intersection with unlabeled_ids.

        Args:
            unlabeled_ids: Set of unlabeled sample IDs
            cluster2sample_ids: Dictionary mapping cluster IDs to sets of sample IDs

        Returns:
            Set of cluster IDs that are fully annotated (contain no unlabeled samples)
        """
        annotated = set()
        for cluster_id, sample_ids in cluster2sample_ids.items():
            if not unlabeled_ids.intersection(sample_ids):
                # Cluster has no unlabeled samples, so it's fully annotated
                annotated.add(cluster_id)

        return annotated
