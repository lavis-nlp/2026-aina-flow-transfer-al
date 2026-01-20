from typing import Dict
import numpy as np
from sklearn.metrics import pairwise_distances

from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from flc.shared.classification.base import ClassificationModel
from .configs import UncertaintyNoveltyClusterQueryStrategyConfig


class UncertaintyNoveltyClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that combines uncertainty and novelty scores to select clusters.

    Computes a weighted combination of:
    - Uncertainty: Average prediction uncertainty of samples in each cluster
    - Novelty: Average novelty (distance to reference samples) of samples in each cluster

    Final score: uncertainty_weight * uncertainty + novelty_weight * novelty
    """

    def __init__(self, config: UncertaintyNoveltyClusterQueryStrategyConfig):
        super().__init__(config)
        self.uncertainty_measure = config.uncertainty_measure
        self.distance_metric = config.distance_metric
        self.novelty_type = config.novelty_type
        self.uncertainty_weight = config.uncertainty_weight
        self.novelty_weight = config.novelty_weight

        # Cache for sample distances to avoid recomputation
        self._distance_cache: Dict[str, float] = {}

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        classifier: ClassificationModel = None,
        source_sample_ids: np.ndarray = None,
        source_features: np.ndarray = None,
        annotated_sample_ids: np.ndarray = None,
        annotated_features: np.ndarray = None,
        **kwargs,
    ) -> int:
        """
        Select cluster with highest combined uncertainty + novelty score.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples
            unlabeled_sample_features: Features of filtered unlabeled samples
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples
            classifier: Trained classification model for uncertainty computation
            source_sample_ids: Source sample IDs (for source_only novelty)
            source_features: Source sample features (for source_only novelty)
            annotated_sample_ids: All annotated sample IDs (for total novelty)
            annotated_features: All annotated sample features (for total novelty)
            **kwargs: Additional arguments

        Returns:
            Cluster ID with highest combined score
        """
        if classifier is None:
            raise ValueError("UncertaintyNoveltyClusterQueryStrategy requires a trained classifier")

        # Determine reference samples for novelty computation
        if self.novelty_type == "source_only":
            if source_features is None or len(source_features) == 0:
                raise ValueError("source_only novelty requires source samples")
            reference_sample_ids = source_sample_ids
            reference_features = source_features
        elif self.novelty_type == "total":
            if annotated_features is None or len(annotated_features) == 0:
                raise ValueError("total novelty requires annotated samples")
            reference_sample_ids = annotated_sample_ids
            reference_features = annotated_features
        else:
            raise ValueError(f"Unknown novelty_type: {self.novelty_type}")

        # Get available clusters
        available_clusters = np.unique(unlabeled_sample_clusters).tolist()

        if not available_clusters:
            raise ValueError("No clusters available for selection")

        # Compute uncertainty scores per cluster
        cluster_uncertainties = self._compute_cluster_uncertainties(
            unlabeled_sample_features, unlabeled_sample_clusters, classifier, available_clusters
        )

        # Compute novelty scores per cluster
        cluster_novelties = self._compute_cluster_novelties(
            unlabeled_sample_ids,
            unlabeled_sample_features,
            unlabeled_sample_clusters,
            reference_sample_ids,
            reference_features,
            available_clusters,
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

    def _compute_cluster_novelties(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        reference_sample_ids: np.ndarray,
        reference_features: np.ndarray,
        available_clusters: list,
    ) -> Dict[int, float]:
        """Compute average novelty (minimum distance to reference samples) per cluster."""
        # Compute novelty for each sample (minimum distance to reference samples)
        sample_novelties = []

        for i, (sample_id, sample_features) in enumerate(zip(unlabeled_sample_ids, unlabeled_sample_features)):
            min_distance_to_reference = float("inf")

            for j, ref_sample_id in enumerate(reference_sample_ids):
                # Check cache first
                cache_key = f"novelty_{sample_id}---{ref_sample_id}"
                if cache_key in self._distance_cache:
                    distance = self._distance_cache[cache_key]
                else:
                    # Compute and cache distance
                    distance = pairwise_distances(
                        sample_features.reshape(1, -1),
                        reference_features[j].reshape(1, -1),
                        metric=self.distance_metric,
                    )[0, 0]
                    self._distance_cache[cache_key] = distance

                min_distance_to_reference = min(min_distance_to_reference, distance)

            sample_novelties.append(min_distance_to_reference)

        sample_novelties = np.array(sample_novelties)

        # Calculate average novelty per cluster
        cluster_novelties = {}
        for cluster_id in available_clusters:
            cluster_mask = unlabeled_sample_clusters == cluster_id
            cluster_sample_novelties = sample_novelties[cluster_mask]
            cluster_novelties[cluster_id] = np.mean(cluster_sample_novelties)

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

    def _calculate_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty scores."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        if probabilities.ndim == 1:
            # Binary classification case
            prob_pos = probabilities
            prob_neg = 1 - prob_pos
            entropy = -(prob_pos * np.log2(prob_pos) + prob_neg * np.log2(prob_neg))
        else:
            # Multi-class classification case
            log_probs = np.log2(probabilities)
            entropy = -np.sum(probabilities * log_probs, axis=1)

        return entropy

    def _calculate_margin(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate margin-based uncertainty scores."""
        if probabilities.ndim == 1:
            # Binary classification case - margin is distance from 0.5
            margin = np.abs(probabilities - 0.5)
            # Negate so higher values mean more uncertain
            return -margin
        else:
            # Multi-class case - margin is difference between top 2 predictions
            sorted_probs = np.sort(probabilities, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            # Negate so higher values mean more uncertain
            return -margin
