from typing import List, Dict

import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from .configs import MostDiverseClusterQueryStrategyConfig


class MostDiverseClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects samples from clusters with highest diversity.

    Rationale: Diverse clusters contain samples with varied characteristics,
    potentially providing richer information for model training.
    """

    def __init__(self, config: MostDiverseClusterQueryStrategyConfig):
        super().__init__(config)
        self.diversity_metric = config.diversity_metric
        self.cluster_selection_method = config.cluster_selection_method
        self.min_samples_for_diversity = config.min_samples_for_diversity

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select the most diverse cluster from filtered clusters.

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

        # Calculate diversity scores for each available cluster
        diversity_scores = self._calculate_cluster_diversity_scores(
            unlabeled_sample_features, unlabeled_sample_clusters, available_clusters
        )

        if self.cluster_selection_method == "most_diverse_first":
            return self._select_most_diverse_first(available_clusters, diversity_scores)
        elif self.cluster_selection_method == "diversity_proportional":
            return self._select_diversity_proportional(available_clusters, diversity_scores, self.random_state)
        else:
            raise ValueError(f"Unknown cluster_selection_method: {self.cluster_selection_method}")

    @staticmethod
    def _select_most_diverse_first(available_clusters: List[int], diversity_scores: Dict[int, float]) -> int:
        """
        Select the cluster with highest diversity score

        Args:
            available_clusters: List of available cluster IDs
            diversity_scores: Dictionary mapping cluster ID to diversity score

        Returns:
            Cluster ID with highest diversity score
        """
        cluster_diversities = [(cid, diversity_scores.get(cid, 0.0)) for cid in available_clusters]
        cluster_diversities.sort(key=lambda x: x[1], reverse=True)
        return cluster_diversities[0][0]

    @staticmethod
    def _select_diversity_proportional(
        available_clusters: List[int], diversity_scores: Dict[int, float], random_state: int
    ) -> int:
        """
        Select cluster proportional to diversity scores (more diverse clusters have higher probability)

        Args:
            available_clusters: List of available cluster IDs
            diversity_scores: Dictionary mapping cluster ID to diversity score
            random_state: Random seed for reproducibility

        Returns:
            Cluster ID selected proportional to diversity
        """
        cluster_weights = [diversity_scores.get(cid, 0.0) for cid in available_clusters]
        total_weight = sum(cluster_weights)

        if total_weight == 0:
            # Fallback to random selection if no diversity info
            rng = np.random.default_rng(random_state)
            return rng.choice(available_clusters)
        else:
            probabilities = [w / total_weight for w in cluster_weights]
            rng = np.random.default_rng(random_state)
            return rng.choice(available_clusters, p=probabilities)

    def _calculate_cluster_diversity_scores(
        self, unlabeled_sample_features: np.ndarray, sample_clusters: np.ndarray, available_clusters: List[int]
    ) -> Dict[int, float]:
        """
        Calculate diversity scores for each cluster based on the specified metric
        """
        diversity_scores = {}

        for cluster_id in available_clusters:
            # Get samples belonging to this cluster
            cluster_mask = sample_clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < self.min_samples_for_diversity:
                diversity_scores[cluster_id] = 0.0
                continue

            cluster_features = unlabeled_sample_features[cluster_indices]

            if self.diversity_metric == "variance":
                diversity_scores[cluster_id] = self._calculate_variance_diversity(cluster_features)
            elif self.diversity_metric == "centroid_distance":
                diversity_scores[cluster_id] = self._calculate_centroid_distance_diversity(cluster_features)
            elif self.diversity_metric == "silhouette":
                # Silhouette calculation using available unlabeled data
                diversity_scores[cluster_id] = self._calculate_silhouette_diversity(
                    unlabeled_sample_features, sample_clusters, cluster_id, cluster_indices
                )
            else:
                raise ValueError(f"Unknown diversity_metric: {self.diversity_metric}")

        return diversity_scores

    def _calculate_variance_diversity(self, cluster_features: np.ndarray) -> float:
        """Calculate diversity as sum of feature variances"""
        if len(cluster_features) < 2:
            return 0.0

        variances = np.var(cluster_features, axis=0)
        return float(np.sum(variances))

    def _calculate_centroid_distance_diversity(self, cluster_features: np.ndarray) -> float:
        """Calculate diversity as average distance from cluster centroid"""
        if len(cluster_features) < 2:
            return 0.0

        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        return float(np.mean(distances))

    def _calculate_silhouette_diversity(
        self,
        unlabeled_sample_features: np.ndarray,
        sample_clusters: np.ndarray,
        cluster_id: int,
        cluster_indices: np.ndarray,
    ) -> float:
        """Calculate diversity using silhouette score for cluster samples"""
        if len(cluster_indices) < 2:
            return 0.0

        try:
            # Calculate pairwise distances for efficiency
            cluster_features = unlabeled_sample_features[cluster_indices]

            # Calculate intra-cluster distances
            intra_distances = pairwise_distances(cluster_features)
            avg_intra_distance = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])

            # For inter-cluster distance, we use distance to nearest other cluster
            other_cluster_ids = np.unique(sample_clusters)
            other_cluster_ids = other_cluster_ids[other_cluster_ids != cluster_id]

            if len(other_cluster_ids) == 0:
                return float(avg_intra_distance)  # Only one cluster

            min_inter_distance = float("inf")
            for other_id in other_cluster_ids:
                other_mask = sample_clusters == other_id
                other_indices = np.where(other_mask)[0]
                other_features = unlabeled_sample_features[other_indices]

                if len(other_features) > 0:
                    inter_distances = pairwise_distances(cluster_features, other_features)
                    avg_inter_distance = np.mean(inter_distances)
                    min_inter_distance = min(min_inter_distance, float(avg_inter_distance))

            # Return intra-cluster distance as diversity measure
            # (high intra-cluster distance = high diversity)
            return float(avg_intra_distance)

        except Exception:
            return 0.0
