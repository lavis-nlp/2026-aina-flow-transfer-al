import numpy as np
from sklearn.cluster import KMeans
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base import ClusteringAlgorithm, ClusteringConfig


@dataclass
class KMeansConfig(ClusteringConfig):
    """Configuration for K-Means clustering"""
    n_clusters: int = Field(8, description="Number of clusters")
    init: str = Field("k-means++", description="Initialization method")
    max_iter: int = Field(300, description="Maximum number of iterations")
    tol: float = Field(1e-4, description="Tolerance for convergence")
    n_init: int = Field(10, description="Number of random initializations")


class KMeansClustering(ClusteringAlgorithm):
    """K-Means clustering implementation"""
    
    def __init__(self, config: KMeansConfig):
        super().__init__(config)
        self._model = KMeans(
            n_clusters=config.n_clusters,
            init=config.init,
            max_iter=config.max_iter,
            tol=config.tol,
            n_init=config.n_init,
            random_state=config.random_state
        )
    
    def fit(self, features: np.ndarray) -> 'KMeansClustering':
        """Fit K-Means to the features"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        self._model.fit(features)
        self._is_fitted = True
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        return self._model.predict(features)
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")
        
        labels = self._model.fit_predict(features)
        self._is_fitted = True
        return labels