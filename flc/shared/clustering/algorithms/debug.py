import numpy as np
from sklearn.cluster import DBSCAN
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base import ClusteringAlgorithm, ClusteringConfig


@dataclass
class DebugConfig(ClusteringConfig):
    """Configuration for DBSCAN clustering"""

    pass


class DebugClustering(ClusteringAlgorithm):
    """DBSCAN clustering implementation"""

    def __init__(self, config: DebugConfig):
        super().__init__(config)

    def fit(self, features: np.ndarray) -> "DebugClustering":
        """Fit DBSCAN to the features"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")

        self._is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels

        Note: DBSCAN doesn't have a traditional predict method.
        This will fit the model on the new data.
        """
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")

        # DBSCAN doesn't have predict method, so we refit
        return self.fit_predict(features)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")

        labels = np.array([i for i in range(features.shape[0])])  # Dummy labels for debugging
        self._is_fitted = True
        return labels
