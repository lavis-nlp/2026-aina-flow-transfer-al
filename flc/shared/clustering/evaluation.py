import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from typing import Dict, Any, Optional, List, Tuple
import warnings


class ClusteringEvaluator:
    """Evaluates clustering quality using various metrics"""

    @staticmethod
    def evaluate(features: np.ndarray, labels: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute clustering evaluation metrics

        Args:
            features: Input features used for clustering
            labels: Predicted cluster labels
            true_labels: Ground truth labels (optional, for supervised metrics)

        Returns:
            Dictionary containing evaluation metrics
        """
        n_clusters = len(np.unique(labels))
        n_samples = len(labels)
        n_noise = np.sum(labels == -1) if -1 in labels else 0

        # Basic info
        result = {
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_ratio": n_noise / n_samples if n_samples > 0 else 0.0,
        }

        # Handle edge cases
        if n_clusters < 2:
            result.update(
                {
                    "error": "Less than 2 clusters found",
                    "silhouette_score": None,
                    "calinski_harabasz_score": None,
                    "davies_bouldin_score": None,
                }
            )
        else:
            # Filter out noise points for internal metrics
            if n_noise > 0:
                mask = labels != -1
                filtered_features = features[mask]
                filtered_labels = labels[mask]
            else:
                filtered_features = features
                filtered_labels = labels

            # Only compute if we have enough samples and clusters
            if len(filtered_features) >= 2 and len(np.unique(filtered_labels)) >= 2:
                try:
                    # Internal clustering metrics (don't require ground truth)
                    result["silhouette_score"] = silhouette_score(filtered_features, filtered_labels)
                    result["calinski_harabasz_score"] = calinski_harabasz_score(filtered_features, filtered_labels)
                    result["davies_bouldin_score"] = davies_bouldin_score(filtered_features, filtered_labels)
                except Exception as e:
                    warnings.warn(f"Error computing internal metrics: {e}")
                    result.update(
                        {
                            "silhouette_score": None,
                            "calinski_harabasz_score": None,
                            "davies_bouldin_score": None,
                            "error": str(e),
                        }
                    )
            else:
                result.update(
                    {
                        "silhouette_score": None,
                        "calinski_harabasz_score": None,
                        "davies_bouldin_score": None,
                        "error": "Insufficient samples or clusters for evaluation",
                    }
                )

        # External metrics (require ground truth labels)
        if true_labels is not None:
            try:
                result.update(
                    {
                        "adjusted_rand_score": adjusted_rand_score(true_labels, labels),
                        "adjusted_mutual_info_score": adjusted_mutual_info_score(true_labels, labels),
                        "homogeneity_score": homogeneity_score(true_labels, labels),
                        "completeness_score": completeness_score(true_labels, labels),
                        "v_measure_score": v_measure_score(true_labels, labels),
                    }
                )
            except Exception as e:
                warnings.warn(f"Error computing external metrics: {e}")
                result["external_metrics_error"] = str(e)

        return result

    @staticmethod
    def get_cluster_statistics(
        features: np.ndarray, labels: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for each cluster

        Args:
            features: Input features
            labels: Cluster labels
            feature_names: Names of features (optional)

        Returns:
            Dictionary with cluster statistics
        """
        unique_labels = np.unique(labels)
        stats = {}

        for label in unique_labels:
            mask = labels == label
            cluster_features = features[mask]

            cluster_name = f"cluster_{label}" if label != -1 else "noise"

            stats[cluster_name] = {
                "size": np.sum(mask),
                "percentage": np.sum(mask) / len(labels) * 100,
                "feature_means": np.mean(cluster_features, axis=0),
                "feature_stds": np.std(cluster_features, axis=0),
                "centroid": np.mean(cluster_features, axis=0),
            }

            # Add feature names if provided
            if feature_names is not None:
                stats[cluster_name]["feature_names"] = feature_names

        return stats

    @staticmethod
    def multi_to_single_label(multi_label: np.ndarray) -> np.ndarray:
        """
        Convert multi-label format to single-label format by taking the argmax.

        Args:
            multi_label: Input multi-label array (2D)

        Returns:
            Single-label array (1D)
        """
        return np.argmax(multi_label, axis=1)
