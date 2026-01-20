import numpy as np
from .base import ClassifierQueryStrategy
from .configs import UncertaintyQueryStrategyConfig


class UncertaintyQueryStrategy(ClassifierQueryStrategy):
    """
    Uncertainty-based query strategy that selects samples with highest prediction uncertainty.

    Supports both entropy-based and margin-based uncertainty measures for single-label
    and multi-label classification.
    """

    def __init__(self, config: UncertaintyQueryStrategyConfig):
        super().__init__(config)
        self.uncertainty_measure = config.uncertainty_measure

    def select_samples(
        self,
        model,
        unlabeled_features: np.ndarray,
        n_samples: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Select samples with highest prediction uncertainty

        Args:
            model: Trained classification model with predict_proba method
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

        # Get prediction probabilities
        try:
            probabilities = model.predict_proba(unlabeled_features)
        except AttributeError:
            raise ValueError("Model must have predict_proba method for uncertainty-based selection")

        # Calculate uncertainty scores
        if self.uncertainty_measure == "entropy":
            uncertainty_scores = self._calculate_entropy(probabilities)
        elif self.uncertainty_measure == "margin":
            uncertainty_scores = self._calculate_margin(probabilities)
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")

        # Select top n_samples with highest uncertainty
        top_indices = np.argsort(uncertainty_scores)[-n_samples:]
        selected_indices = top_indices[::-1]  # Descending order of uncertainty

        return selected_indices

    def _calculate_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate entropy-based uncertainty scores

        Args:
            probabilities: Prediction probabilities of shape (n_samples, n_classes)

        Returns:
            Array of entropy scores (higher = more uncertain)
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        if probabilities.ndim == 1:
            # Binary classification case
            prob_pos = probabilities
            prob_neg = 1 - prob_pos
            entropy = -(prob_pos * np.log2(prob_pos) + prob_neg * np.log2(prob_neg))
        else:
            # Multi-label case: calculate binary entropy for each label independently
            # Binary entropy: -(p * log2(p) + (1-p) * log2(1-p))
            entropy_per_label = -(
                probabilities * np.log2(probabilities) + (1 - probabilities) * np.log2(1 - probabilities)
            )

            # Aggregate per sample: mean entropy across all labels
            entropy = np.mean(entropy_per_label, axis=1)

        return entropy

    def _calculate_margin(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate margin-based uncertainty scores

        Args:
            probabilities: Prediction probabilities of shape (n_samples, n_classes)

        Returns:
            Array of margin scores (lower = more uncertain, so we negate for consistency)
        """
        if probabilities.ndim == 1:
            # Binary classification case - margin is distance from 0.5
            margin = np.abs(probabilities - 0.5)
            # Negate so higher values mean more uncertain
            return -margin
        else:
            # Multi-label case: calculate distance from 0.5 decision boundary for each label
            # Distance from decision boundary: abs(p - 0.5)
            margin_per_label = np.abs(probabilities - 0.5)

            # Aggregate per sample: mean margin across all labels
            margin = np.mean(margin_per_label, axis=1)

            # Negate so higher values mean more uncertain
            return -margin
