"""
Clustering module for flow features

This module provides a generic clustering interface that can be applied
to any set of numerical features, with specific utilities for flow features.
"""

from .factory import ClusteringFactory
from .base import ClusteringAlgorithm, ClusteringConfig
from .evaluation import ClusteringEvaluator

# Algorithm imports
from .algorithms.kmeans import KMeansClustering, KMeansConfig
from .algorithms.dbscan import DBSCANClustering, DBSCANConfig
from .algorithms.hierarchical import HierarchicalClustering, HierarchicalConfig

# Utils import
from . import utils

__all__ = [
    # Factory and base classes
    "ClusteringFactory",
    "ClusteringAlgorithm",
    "ClusteringConfig",
    # Evaluation
    "ClusteringEvaluator",
    # Specific algorithms
    "KMeansClustering",
    "KMeansConfig",
    "DBSCANClustering",
    "DBSCANConfig",
    "HierarchicalClustering",
    "HierarchicalConfig",
    # Utility functions
    "utils",
]
