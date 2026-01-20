import time
from typing import Tuple, Dict, Any
import numpy as np

from flc.shared.clustering.factory import ClusteringFactory


def execute_clustering(
    features: np.ndarray, 
    algorithm_name: str, 
    hyperparameters: Dict[str, Any],
    random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Execute clustering using existing ClusteringFactory.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        algorithm_name: Name of clustering algorithm
        hyperparameters: Hyperparameters for the algorithm
        random_state: Random state for reproducible results
    
    Returns:
        - cluster_labels: Cluster assignments (n_samples,)
        - execution_time: Time taken for clustering in seconds
    
    Raises:
        ValueError: If algorithm name is invalid or clustering fails
        RuntimeError: If clustering execution fails
    """
    try:
        # Add random_state to hyperparameters if algorithm supports it
        hyperparams_with_seed = hyperparameters.copy()
        
        # Only add random_state for algorithms that support it
        algorithm_supports_random_state = algorithm_name.lower() in ['kmeans', 'k-means']
        if algorithm_supports_random_state:
            hyperparams_with_seed['random_state'] = random_state
        
        # Create clustering algorithm using existing factory
        clustering_algo = ClusteringFactory.create_with_defaults(
            algorithm_name, 
            **hyperparams_with_seed
        )
        
        # Execute clustering with timing
        start_time = time.time()
        cluster_labels = clustering_algo.fit_predict(features)
        execution_time = time.time() - start_time
        
        return cluster_labels, execution_time
    
    except Exception as e:
        raise RuntimeError(f"Failed to execute {algorithm_name} clustering: {str(e)}") from e


def validate_clustering_inputs(features: np.ndarray, algorithm_name: str, hyperparameters: Dict[str, Any]) -> None:
    """
    Validate inputs for clustering execution.
    
    Args:
        features: Feature matrix
        algorithm_name: Name of clustering algorithm
        hyperparameters: Hyperparameters for the algorithm
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate features
    if not isinstance(features, np.ndarray):
        raise ValueError("Features must be numpy array")
    
    if features.ndim != 2:
        raise ValueError("Features must be 2D array")
    
    if features.shape[0] == 0:
        raise ValueError("Features array is empty")
    
    if features.shape[1] == 0:
        raise ValueError("Features array has no features")
    
    # Check for NaN or infinite values
    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values")
    
    # Validate algorithm name
    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("Algorithm name must be non-empty string")
    
    # Validate hyperparameters
    if not isinstance(hyperparameters, dict):
        raise ValueError("Hyperparameters must be dictionary")


def get_clustering_summary(cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Get summary statistics about clustering results.
    
    Args:
        cluster_labels: Cluster assignments
    
    Returns:
        Dictionary with clustering summary statistics
    """
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    n_noise = np.sum(cluster_labels == -1) if -1 in cluster_labels else 0
    n_samples = len(cluster_labels)
    
    # Compute cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        if label == -1:
            cluster_sizes['noise'] = np.sum(cluster_labels == -1)
        else:
            cluster_sizes[f'cluster_{label}'] = np.sum(cluster_labels == label)
    
    return {
        'n_clusters': n_clusters,
        'n_noise_samples': n_noise,
        'n_samples': n_samples,
        'noise_ratio': n_noise / n_samples if n_samples > 0 else 0.0,
        'cluster_sizes': cluster_sizes,
        'has_noise': n_noise > 0
    }