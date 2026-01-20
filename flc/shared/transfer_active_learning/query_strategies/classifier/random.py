import numpy as np
from .base import ClassifierQueryStrategy
from .configs import RandomQueryStrategyConfig


class RandomQueryStrategy(ClassifierQueryStrategy):
    """
    Random query strategy that randomly selects unlabeled samples.

    This strategy provides a baseline for comparison with other active learning
    strategies by randomly selecting samples without any heuristic.
    """

    def __init__(self, config: RandomQueryStrategyConfig):
        super().__init__(config)
        self.random_state = config.random_state

    def select_samples(
        self,
        model,
        unlabeled_features: np.ndarray,
        n_samples: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Randomly select samples from unlabeled data

        Args:
            model: Trained classification model (unused)
            unlabeled_features: Features of unlabeled samples
            n_samples: Number of samples to select

        Returns:
            Array of selected sample indices in the unlabeled dataset
        """
        if len(unlabeled_features) == 0:
            return np.array([], dtype=int)

        if n_samples <= 0:
            return np.array([], dtype=int)

        if n_samples >= len(unlabeled_features):
            return np.arange(len(unlabeled_features))

        # Randomly select n_samples without replacement
        seed = self.random_state * len(unlabeled_features)
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(unlabeled_features), size=n_samples, replace=False)

        return sample_indices
