from typing import Dict
import numpy as np
from flc.shared.transfer_active_learning.query_strategies.cluster.base import ClusterQueryStrategy
from flc.shared.classification.base import ClassificationModel
from .configs import HighestUncertaintyClusterQueryStrategyConfig


class HighestUncertaintyClusterQueryStrategy(ClusterQueryStrategy):
    """
    Query strategy that selects the cluster with highest average prediction uncertainty.

    Rationale: Clusters with high average uncertainty likely contain samples
    where the classifier is least confident, making them valuable for labeling.
    """

    def __init__(self, config: HighestUncertaintyClusterQueryStrategyConfig):
        super().__init__(config)
        self.uncertainty_measure = config.uncertainty_measure

    def _select_cluster(
        self,
        unlabeled_sample_ids: np.ndarray,
        unlabeled_sample_features: np.ndarray,
        unlabeled_sample_clusters: np.ndarray,
        **kwargs,
    ) -> int:
        """
        Select the cluster with highest average prediction uncertainty from filtered clusters.

        Args:
            unlabeled_sample_ids: IDs of filtered unlabeled samples (n_filtered_samples,)
            unlabeled_sample_features: Features of filtered unlabeled samples (n_filtered_samples, n_features)
            unlabeled_sample_clusters: Cluster labels for filtered unlabeled samples (n_filtered_samples,)
            **kwargs: Must include 'classifier' - Trained classification model with predict_proba method

        Returns:
            Cluster ID with highest average uncertainty
        """
        classifier: ClassificationModel = kwargs.get("classifier")
        if classifier is None:
            raise ValueError("HighestUncertaintyClusterQueryStrategy requires a trained classifier")

        cluster_sizes = self._get_cluster_sizes(unlabeled_sample_clusters)
        available_clusters = list(cluster_sizes.keys())

        if not available_clusters:
            raise ValueError("No clusters available for selection")

        # Get prediction probabilities for all samples
        try:
            probabilities = classifier.predict_proba(unlabeled_sample_features)
        except AttributeError:
            raise ValueError("Classifier must have predict_proba method for uncertainty-based selection")

        # Calculate uncertainty scores for all samples
        if self.uncertainty_measure == "entropy":
            sample_uncertainties = self._calculate_entropy(probabilities)
        elif self.uncertainty_measure == "margin":
            sample_uncertainties = self._calculate_margin(probabilities)
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")

        # Calculate average uncertainty per cluster
        cluster_avg_uncertainties = {}
        for cluster_id in available_clusters:
            cluster_mask = unlabeled_sample_clusters == cluster_id
            cluster_sample_uncertainties = sample_uncertainties[cluster_mask]
            cluster_avg_uncertainties[cluster_id] = np.mean(cluster_sample_uncertainties)

        # Select cluster with highest average uncertainty
        selected_cluster_id = max(cluster_avg_uncertainties.keys(), 
                                key=lambda cid: cluster_avg_uncertainties[cid])
        
        return selected_cluster_id

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
        entropy_per_label = -(probabilities * np.log2(probabilities) + 
                             (1 - probabilities) * np.log2(1 - probabilities))
        
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