from abc import ABC, abstractmethod
import numpy as np

from .configs import BaseQueryStrategyConfig


class ClassifierQueryStrategy(ABC):
    """Abstract base class for classifier-based query strategies"""

    def __init__(self, config: BaseQueryStrategyConfig):
        self.config = config
        self.random_state = config.random_state

    @abstractmethod
    def select_samples(
        self,
        model,
        unlabeled_features: np.ndarray,
        n_samples: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Select next samples to label based on classifier predictions

        Args:
            model: Trained classification model
            unlabeled_features: Features of unlabeled samples
            n_samples: Number of samples to select

        Returns:
            Array of selected sample indices in the unlabeled dataset
        """
        pass
