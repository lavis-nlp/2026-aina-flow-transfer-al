import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base import ClusteringAlgorithm, ClusteringConfig


@dataclass
class HierarchicalConfig(ClusteringConfig):
    """Configuration for Hierarchical clustering"""
    n_clusters: int = Field(8, description="Number of clusters")
    linkage: str = Field("ward", description="Linkage criterion")
    metric: str = Field("euclidean", description="Distance metric")
    compute_distances: bool = Field(False, description="Compute distances for dendrogram")


class HierarchicalClustering(ClusteringAlgorithm):
    """Hierarchical (Agglomerative) clustering implementation"""
    
    def __init__(self, config: HierarchicalConfig):
        super().__init__(config)
        self._model = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            linkage=config.linkage,
            metric=config.metric,
            compute_distances=config.compute_distances
        )
    
    def fit(self, features: np.ndarray) -> 'HierarchicalClustering':
        """Fit Hierarchical clustering to the features"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        self._model.fit(features)
        self._is_fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels
        
        Note: Hierarchical clustering doesn't have a traditional predict method.
        This will fit the model on the new data.
        """
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        # Hierarchical clustering doesn't have predict method, so we refit
        return self._model.fit_predict(features)
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        labels = self._model.fit_predict(features)
        self._is_fitted = True
        return labels
    
    def get_distances(self) -> np.ndarray:
        """Get distances for dendrogram construction"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self._model, 'distances_'):
            return self._model.distances_
        else:
            raise ValueError("Distances not computed. Set compute_distances=True in config.")