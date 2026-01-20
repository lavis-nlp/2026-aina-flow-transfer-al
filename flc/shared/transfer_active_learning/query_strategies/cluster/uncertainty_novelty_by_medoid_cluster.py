from typing import Dict, Set, Optional
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from flc.shared.classification.base import ClassificationModel
from flc.shared.clustering import utils as clustering_utils
from .configs import UncertaintyNoveltyByMedoidClusterQueryStrategyConfig


class UncertaintyNoveltyByMedoidClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that combines uncertainty and novelty scores to select clusters using medoid-based novelty.

    Computes a weighted combination of:
    - Uncertainty: Average prediction uncertainty of samples in each cluster
    - Novelty: Distance from cluster medoid to nearest reference cluster medoid

    Final score: uncertainty_weight * uncertainty + novelty_weight * novelty
    """

    def __init__(self, config: UncertaintyNoveltyByMedoidClusterQueryStrategyConfig):
        super().__init__(config)
        self.uncertainty_measure = config.uncertainty_measure
        self.distance_metric = config.distance_metric
        self.novelty_type = config.novelty_type
        self.uncertainty_weight = config.uncertainty_weight
        self.novelty_weight = config.novelty_weight

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
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        classifier: ClassificationModel = None,
        cluster2source_ids: Dict[int, Set[str]] = None,
        cluster2target_ids: Dict[int, Set[str]] = None,
        source_id2features: dict[str, np.ndarray] = None,
        target_id2features: dict[str, np.ndarray] = None,
        **kwargs,
    ) -> int:
        """
        Select cluster with highest combined uncertainty + novelty score.

        Args:
            unlabeled_sample_ids: Array of unlabeled target sample IDs
            unlabeled_sample_features: Features of filtered unlabeled samples
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples
            classifier: Trained classification model for uncertainty computation
            cluster2source_ids: Mapping of source cluster IDs to sample IDs
            cluster2target_ids: Mapping of target cluster IDs to sample IDs
            source_id2features: Dictionary mapping source sample IDs to feature vectors
            target_id2features: Dictionary mapping target sample IDs to feature vectors
            **kwargs: Additional arguments

        Returns:
            Cluster ID with highest combined score
        """
        # Validate required parameters
        if classifier is None:
            raise ValueError("UncertaintyNoveltyByMedoidClusterQueryStrategy requires a trained classifier")
        if cluster2source_ids is None or cluster2target_ids is None:
            raise ValueError(
                "UncertaintyNoveltyByMedoidClusterQueryStrategy requires cluster2source_ids and cluster2target_ids parameters"
            )
        if source_id2features is None or target_id2features is None:
            raise ValueError(
                "UncertaintyNoveltyByMedoidClusterQueryStrategy requires source_id2features and target_id2features parameters"
            )
        if unlabeled_sample_ids is None:
            raise ValueError("UncertaintyNoveltyByMedoidClusterQueryStrategy requires unlabeled_sample_ids parameter")
        if len(unlabeled_sample_ids) == 0:
            raise ValueError("No unlabeled samples provided")

        # Convert array to set for efficient lookup
        unlabeled_target_ids = set(unlabeled_sample_ids)

        # Validate non-empty data
        if not cluster2target_ids:
            raise ValueError("No target clusters provided")
        if not target_id2features:
            raise ValueError("No target features provided")

        # remove noise clusters
        if self.config.exclude_noise:
            cluster2source_ids = {cid: samples for cid, samples in cluster2source_ids.items() if cid != -1}
            cluster2target_ids = {cid: samples for cid, samples in cluster2target_ids.items() if cid != -1}

        # Get medoids for source and target clusters
        source_cluster2medoid_id = self._get_medoids("source", cluster2source_ids, source_id2features)
        target_cluster2medoid_id = self._get_medoids("target", cluster2target_ids, target_id2features)

        # Find already annotated target clusters
        annotated_target_clusters = self._get_annotated_clusters_from_unlabeled(
            unlabeled_ids=unlabeled_target_ids, cluster2sample_ids=cluster2target_ids
        )

        # Get available unlabeled clusters
        available_clusters = np.unique(unlabeled_sample_clusters).tolist()

        if not available_clusters:
            raise ValueError("No clusters available for selection")

        # Compute uncertainty scores per cluster
        cluster_uncertainties = self._compute_cluster_uncertainties(
            unlabeled_sample_features, unlabeled_sample_clusters, classifier, available_clusters
        )

        # Compute novelty scores per cluster using new efficient approach
        cluster_novelties = self._compute_cluster_novelties_efficient(
            available_clusters,
            source_cluster2medoid_id,
            target_cluster2medoid_id,
            annotated_target_clusters,
            source_id2features,
            target_id2features,
        )

        # Normalize scores to [0, 1] range
        normalized_uncertainties = self._normalize_scores(cluster_uncertainties)
        normalized_novelties = self._normalize_scores(cluster_novelties)

        # Compute combined scores
        cluster_scores = {}
        for cluster_id in available_clusters:
            uncertainty_score = normalized_uncertainties[cluster_id]
            novelty_score = normalized_novelties[cluster_id]
            combined_score = self.uncertainty_weight * uncertainty_score + self.novelty_weight * novelty_score
            cluster_scores[cluster_id] = combined_score

        # Select cluster with highest combined score
        selected_cluster_id = max(cluster_scores.keys(), key=lambda cid: cluster_scores[cid])
        return selected_cluster_id

    def _compute_cluster_uncertainties(
        self,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        classifier: ClassificationModel,
        available_clusters: list,
    ) -> Dict[int, float]:
        """Compute average uncertainty per cluster."""
        # Get prediction probabilities for all samples
        try:
            probabilities = classifier.predict_proba(unlabeled_sample_features)
        except AttributeError:
            raise ValueError("Classifier must have predict_proba method for uncertainty computation")

        # Calculate uncertainty scores for all samples
        if self.uncertainty_measure == "entropy":
            sample_uncertainties = self._calculate_entropy(probabilities)
        elif self.uncertainty_measure == "margin":
            sample_uncertainties = self._calculate_margin(probabilities)
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")

        # Calculate average uncertainty per cluster
        cluster_uncertainties = {}
        for cluster_id in available_clusters:
            cluster_mask = unlabeled_sample_clusters == cluster_id
            cluster_sample_uncertainties = sample_uncertainties[cluster_mask]
            cluster_uncertainties[cluster_id] = np.mean(cluster_sample_uncertainties)

        return cluster_uncertainties

    def _compute_cluster_novelties_efficient(
        self,
        available_clusters: list,
        source_cluster2medoid_id: Dict[int, str],
        target_cluster2medoid_id: Dict[int, str],
        annotated_target_clusters: Set[int],
        source_id2features: dict[str, np.ndarray],
        target_id2features: dict[str, np.ndarray],
    ) -> Dict[int, float]:
        """Compute novelty scores per cluster using efficient medoid-to-medoid distances."""

        # Determine reference medoid sample IDs based on novelty type
        reference_medoid_sample_ids = []

        if self.novelty_type == "source_only":
            # Only use source cluster medoids as reference
            for cluster_id, medoid_sample_id in source_cluster2medoid_id.items():
                reference_medoid_sample_ids.append(medoid_sample_id)

        elif self.novelty_type == "total":
            # Use both source cluster medoids and labeled target cluster medoids
            # Add all source cluster medoids
            for cluster_id, medoid_sample_id in source_cluster2medoid_id.items():
                reference_medoid_sample_ids.append(medoid_sample_id)

            # Add medoids of labeled target clusters
            for cluster_id, medoid_sample_id in target_cluster2medoid_id.items():
                if cluster_id in annotated_target_clusters:
                    reference_medoid_sample_ids.append(medoid_sample_id)
        else:
            raise ValueError(f"Unknown novelty_type: {self.novelty_type}")

        if not reference_medoid_sample_ids:
            raise ValueError("No reference cluster medoids found for novelty computation")

        # Compute novelty for each available cluster
        cluster_novelties = {}
        for cluster_id in available_clusters:
            if cluster_id not in target_cluster2medoid_id:
                raise ValueError(f"No medoid found for cluster {cluster_id}")

            unlabeled_medoid_sample_id = target_cluster2medoid_id[cluster_id]
            unlabeled_medoid_features = target_id2features[unlabeled_medoid_sample_id]

            # Find distance to nearest reference cluster medoid using caching
            min_distance_to_reference = float("inf")

            for reference_medoid_sample_id in reference_medoid_sample_ids:
                # Get features for reference medoid - need to check both source and target
                if reference_medoid_sample_id in source_id2features:
                    reference_medoid_features = source_id2features[reference_medoid_sample_id]
                elif reference_medoid_sample_id in target_id2features:
                    reference_medoid_features = target_id2features[reference_medoid_sample_id]
                else:
                    raise ValueError(f"Features not found for reference medoid sample {reference_medoid_sample_id}")

                # Use cached distance computation
                distance = self._compute_medoid_distance(
                    unlabeled_medoid_sample_id,
                    reference_medoid_sample_id,
                    unlabeled_medoid_features,
                    reference_medoid_features,
                )
                min_distance_to_reference = min(min_distance_to_reference, distance)

            cluster_novelties[cluster_id] = min_distance_to_reference

        return cluster_novelties

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        values = list(scores.values())
        if len(values) == 0:
            return scores

        min_val = min(values)
        max_val = max(values)

        # Handle case where all scores are the same
        if max_val == min_val:
            return {cluster_id: 1.0 for cluster_id in scores.keys()}

        # Min-max normalization
        normalized = {}
        for cluster_id, score in scores.items():
            normalized[cluster_id] = (score - min_val) / (max_val - min_val)

        return normalized

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

    def _calculate_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate entropy-based uncertainty scores for multi-label classification

        Args:
            probabilities: Prediction probabilities of shape (n_samples, n_labels)
                          Each value is independent label probability from sigmoid output

        Returns:
            Array of entropy scores (higher = more uncertain)
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        # Multi-label case: calculate binary entropy for each label independently
        # Binary entropy: -(p * log2(p) + (1-p) * log2(1-p))
        entropy_per_label = -(probabilities * np.log2(probabilities) + (1 - probabilities) * np.log2(1 - probabilities))

        # Aggregate per sample: mean entropy across all labels
        entropy = np.mean(entropy_per_label, axis=1)

        return entropy

    def _calculate_margin(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate margin-based uncertainty scores for multi-label classification

        Args:
            probabilities: Prediction probabilities of shape (n_samples, n_labels)
                          Each value is independent label probability from sigmoid output

        Returns:
            Array of margin scores (higher = more uncertain)
        """
        # Multi-label case: calculate distance from 0.5 decision boundary for each label
        # Distance from decision boundary: abs(p - 0.5)
        margin_per_label = np.abs(probabilities - 0.5)

        # Aggregate per sample: mean margin across all labels
        margin = np.mean(margin_per_label, axis=1)

        # Negate so higher values mean more uncertain
        return -margin
