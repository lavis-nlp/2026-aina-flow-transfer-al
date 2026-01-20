"""Classification module for single-label and multi-label classification"""

from .base import ClassificationModel, ClassificationConfig
from .factory import ClassificationFactory
from .evaluation import ClassificationEvaluator

__all__ = [
    'ClassificationModel',
    'ClassificationConfig', 
    'ClassificationFactory',
    'ClassificationEvaluator'
]