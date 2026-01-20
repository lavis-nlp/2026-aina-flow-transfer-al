"""
Transfer Active Learning Module

This module provides a comprehensive framework for transfer active learning,
enabling adaptation of classifiers from source to target datasets through
strategic sample selection and labeling.

The module is split into two distinct approaches:
1. ClassifierLearner - uses classifier uncertainty for sample selection
2. TransferActiveLearningFromClustering - uses initial clustering for cluster-based selection
"""

from .base import (
    ActiveLearningConfig,
    ClassifierActiveLearningConfig,
    ClusteringActiveLearningConfig,
)
from .query_strategies.classifier.base import ClassifierQueryStrategy
from .query_strategies.cluster.base import ClusterQueryStrategy
from .label_selection import LabelSelectionStrategy
from .weighting_strategies import WeightingStrategy

__all__ = [
    "ActiveLearningConfig",
    "ClassifierActiveLearningConfig",
    "ClusteringActiveLearningConfig",
    "ClassifierQueryStrategy",
    "ClusterQueryStrategy",
    "WeightingStrategy",
    "LabelSelectionStrategy",
]
