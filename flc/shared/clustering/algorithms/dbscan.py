import numpy as np
from sklearn.cluster import DBSCAN
from pydantic.dataclasses import dataclass
from pydantic import Field

from ..base import ClusteringAlgorithm, ClusteringConfig


@dataclass
class DBSCANConfig(ClusteringConfig):
    """Configuration for DBSCAN clustering"""

    eps: float = Field(0.5, description="Maximum distance between samples in a neighborhood")
    min_samples: int = Field(5, description="Minimum number of samples in a neighborhood")
    metric: str = Field("euclidean", description="Distance metric")
    algorithm: str = Field("auto", description="Algorithm for nearest neighbors")
    transform_noise: bool = Field(False, description="If True, noise points will be assigned their own cluster label")


class DBSCANClustering(ClusteringAlgorithm):
    """DBSCAN clustering implementation"""

    def __init__(self, config: DBSCANConfig):
        super().__init__(config)
        # Note: DBSCAN doesn't use random_state in sklearn
        self._model = DBSCAN(
            eps=config.eps,
            min_samples=config.min_samples,
            metric=config.metric,
            algorithm=config.algorithm,
        )
        self._transform_noise = config.transform_noise

    def fit(self, features: np.ndarray) -> "DBSCANClustering":
        """Fit DBSCAN to the features"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")

        self._model.fit(features)
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
        labels = self._model.fit_predict(features)

        if self._transform_noise:
            labels = _transform_noise(labels)

        return labels

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        if features.ndim != 2:
            raise ValueError("Features must be 2D array of shape (n_samples, n_features)")

        labels = self._model.fit_predict(features)
        self._is_fitted = True

        if self._transform_noise:
            labels = _transform_noise(labels)

        return labels


def _transform_noise(labels: np.ndarray) -> np.ndarray:
    """
    Transform noise points to individual cluster labels
    """

    # Assign each noise point to its own individual cluster label
    highest_label = np.max(labels)
    output_labels = np.copy(labels)

    for i in range(len(labels)):
        if labels[i] == -1:
            highest_label += 1
            output_labels[i] = highest_label

    return output_labels
