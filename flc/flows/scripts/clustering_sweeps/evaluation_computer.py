from typing import Dict, Any
import numpy as np
import warnings

from flc.shared.clustering.evaluation import ClusteringEvaluator
from .dataset_loader import create_true_labels_for_evaluation


def evaluate_clustering(
    features: np.ndarray, 
    cluster_labels: np.ndarray, 
    true_labels_multihot: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate clustering using existing ClusteringEvaluator.
    
    Args:
        features: Feature matrix used for clustering (n_samples, n_features)
        cluster_labels: Cluster assignments from algorithm (n_samples,)
        true_labels_multihot: Multi-hot encoded true labels (n_samples, n_classes)
    
    Returns:
        Dictionary with all evaluation metrics
    """
    try:
        # Convert multi-hot labels to single label for external metrics
        true_labels_single = create_true_labels_for_evaluation(true_labels_multihot)
        
        # Use existing ClusteringEvaluator
        metrics = ClusteringEvaluator.evaluate(
            features=features,
            labels=cluster_labels,
            true_labels=true_labels_single
        )
        
        # Add additional metrics
        metrics.update(_compute_additional_metrics(cluster_labels, true_labels_multihot))
        
        return metrics
    
    except Exception as e:
        warnings.warn(f"Error in clustering evaluation: {str(e)}")
        return _create_error_metrics(str(e))


def _compute_additional_metrics(cluster_labels: np.ndarray, true_labels_multihot: np.ndarray) -> Dict[str, Any]:
    """
    Compute additional metrics not covered by ClusteringEvaluator.
    
    Args:
        cluster_labels: Cluster assignments
        true_labels_multihot: Multi-hot encoded true labels
    
    Returns:
        Dictionary with additional metrics
    """
    metrics = {}
    
    try:
        # Compute label purity metrics for multi-label case
        metrics.update(_compute_label_purity_metrics(cluster_labels, true_labels_multihot))
        
        # Compute cluster balance metrics
        metrics.update(_compute_cluster_balance_metrics(cluster_labels))
        
    except Exception as e:
        warnings.warn(f"Error computing additional metrics: {str(e)}")
        metrics['additional_metrics_error'] = str(e)
    
    return metrics


def _compute_label_purity_metrics(cluster_labels: np.ndarray, true_labels_multihot: np.ndarray) -> Dict[str, Any]:
    """
    Compute label purity metrics for multi-label clustering evaluation.
    
    Args:
        cluster_labels: Cluster assignments
        true_labels_multihot: Multi-hot encoded true labels
    
    Returns:
        Dictionary with label purity metrics
    """
    metrics = {}
    
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    
    if len(unique_clusters) == 0:
        return {
            'avg_cluster_label_entropy': None,
            'avg_cluster_label_diversity': None,
            'label_coverage_per_cluster': None
        }
    
    entropies = []
    diversities = []
    coverages = []
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_labels_multihot = true_labels_multihot[cluster_mask]
        
        if len(cluster_labels_multihot) == 0:
            continue
        
        # Compute label frequencies in this cluster
        label_frequencies = np.mean(cluster_labels_multihot, axis=0)
        
        # Entropy: measure of label uncertainty in cluster
        entropy = _compute_entropy(label_frequencies)
        entropies.append(entropy)
        
        # Diversity: number of distinct labels in cluster
        diversity = np.sum(label_frequencies > 0)
        diversities.append(diversity)
        
        # Coverage: proportion of all possible labels present in cluster
        coverage = diversity / true_labels_multihot.shape[1]
        coverages.append(coverage)
    
    if entropies:
        metrics['avg_cluster_label_entropy'] = np.mean(entropies)
        metrics['avg_cluster_label_diversity'] = np.mean(diversities)
        metrics['avg_label_coverage_per_cluster'] = np.mean(coverages)
    else:
        metrics['avg_cluster_label_entropy'] = None
        metrics['avg_cluster_label_diversity'] = None
        metrics['avg_label_coverage_per_cluster'] = None
    
    return metrics


def _compute_cluster_balance_metrics(cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute metrics related to cluster size balance.
    
    Args:
        cluster_labels: Cluster assignments
    
    Returns:
        Dictionary with cluster balance metrics
    """
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    
    if len(unique_clusters) <= 1:
        return {
            'cluster_size_std': None,
            'cluster_size_cv': None,
            'largest_cluster_ratio': None
        }
    
    # Compute cluster sizes
    cluster_sizes = []
    for cluster_id in unique_clusters:
        size = np.sum(cluster_labels == cluster_id)
        cluster_sizes.append(size)
    
    cluster_sizes = np.array(cluster_sizes)
    
    # Standard deviation of cluster sizes
    size_std = np.std(cluster_sizes)
    
    # Coefficient of variation (std/mean)
    size_mean = np.mean(cluster_sizes)
    size_cv = size_std / size_mean if size_mean > 0 else None
    
    # Ratio of largest cluster to total non-noise points
    largest_cluster_size = np.max(cluster_sizes)
    total_non_noise = np.sum(cluster_labels != -1)
    largest_cluster_ratio = largest_cluster_size / total_non_noise if total_non_noise > 0 else None
    
    return {
        'cluster_size_std': size_std,
        'cluster_size_cv': size_cv,
        'largest_cluster_ratio': largest_cluster_ratio
    }


def _compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of probability distribution.
    
    Args:
        probabilities: Probability distribution
    
    Returns:
        Shannon entropy
    """
    # Filter out zero probabilities to avoid log(0)
    p_nonzero = probabilities[probabilities > 0]
    
    if len(p_nonzero) == 0:
        return 0.0
    
    return -np.sum(p_nonzero * np.log2(p_nonzero))


def _create_error_metrics(error_message: str) -> Dict[str, Any]:
    """
    Create metrics dictionary for failed evaluation.
    
    Args:
        error_message: Error message
    
    Returns:
        Dictionary with None values and error message
    """
    return {
        'n_samples': None,
        'n_clusters': None,
        'n_noise_points': None,
        'noise_ratio': None,
        'silhouette_score': None,
        'calinski_harabasz_score': None,
        'davies_bouldin_score': None,
        'adjusted_rand_score': None,
        'adjusted_mutual_info_score': None,
        'homogeneity_score': None,
        'completeness_score': None,
        'v_measure_score': None,
        'avg_cluster_label_entropy': None,
        'avg_cluster_label_diversity': None,
        'avg_label_coverage_per_cluster': None,
        'cluster_size_std': None,
        'cluster_size_cv': None,
        'largest_cluster_ratio': None,
        'evaluation_error': error_message
    }