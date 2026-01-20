from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field


@dataclass
class ClusteringConfig:
    """Base configuration for clustering algorithms"""
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")
    

class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._is_fitted = False
        self._model = None
    
    @abstractmethod
    def fit(self, features: np.ndarray) -> 'ClusteringAlgorithm':
        """
        Fit the clustering algorithm to the features
        
        Args:
            features: Input features of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for the features
        
        Args:
            features: Input features of shape (n_samples, n_features)
            
        Returns:
            Cluster labels as numpy array of shape (n_samples,)
        """
        pass
    
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step
        
        Args:
            features: Input features of shape (n_samples, n_features)
            
        Returns:
            Cluster labels as numpy array of shape (n_samples,)
        """
        return self.fit(features).predict(features)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the algorithm has been fitted"""
        return self._is_fitted
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers if available
        
        Returns:
            Cluster centers or None if not available
        """
        if hasattr(self._model, 'cluster_centers_'):
            return self._model.cluster_centers_
        return None
    
    def get_labels(self) -> Optional[np.ndarray]:
        """
        Get training labels if available
        
        Returns:
            Training labels or None if not available
        """
        if hasattr(self._model, 'labels_'):
            return self._model.labels_
        return None