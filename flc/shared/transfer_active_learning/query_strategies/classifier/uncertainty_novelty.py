from typing import Dict, Optional
import numpy as np
from sklearn.metrics import pairwise_distances

from .base import ClassifierQueryStrategy
from .configs import UncertaintyNoveltyQueryStrategyConfig


class UncertaintyNoveltyQueryStrategy(ClassifierQueryStrategy):
    """
    Query strategy that combines uncertainty and novelty scores to select individual samples.

    Computes a weighted combination of:
    - Uncertainty: Prediction uncertainty of individual samples
    - Novelty: Distance to reference samples (source_only or total)

    Final score: uncertainty_weight * uncertainty + novelty_weight * novelty
    """

    def __init__(self, config: UncertaintyNoveltyQueryStrategyConfig):
        super().__init__(config)
        self.uncertainty_measure = config.uncertainty_measure
        self.distance_metric = config.distance_metric
        self.novelty_type = config.novelty_type
        self.uncertainty_weight = config.uncertainty_weight
        self.novelty_weight = config.novelty_weight

        # Cache for sample distances to avoid recomputation
        self._distance_cache: Dict[str, float] = {}

    def select_samples(
        self,
        model,
        unlabeled_features: np.ndarray,
        source_features: Optional[np.ndarray] = None,
        annotated_features: Optional[np.ndarray] = None,
        n_samples: int = 1,
        unlabeled_sample_ids: Optional[np.ndarray] = None,
        source_sample_ids: Optional[np.ndarray] = None,
        annotated_sample_ids: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Select samples with highest combined uncertainty + novelty scores

        Args:
            model: Trained classification model with predict_proba method
            unlabeled_features: Features of unlabeled samples (n_unlabeled, n_features)
            source_features: Source sample features (for source_only novelty)
            annotated_features: All annotated sample features (for total novelty)
            n_samples: Number of samples to select
            unlabeled_sample_ids: Optional sample IDs for unlabeled samples (for caching)
            source_sample_ids: Optional sample IDs for source samples (for caching)
            annotated_sample_ids: Optional sample IDs for annotated samples (for caching)
            **kwargs: Additional unused arguments for interface compatibility

        Returns:
            Array of selected sample indices in the unlabeled dataset

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if len(unlabeled_features) == 0:
            return np.array([], dtype=int)

        if n_samples <= 0:
            return np.array([], dtype=int)

        if n_samples >= len(unlabeled_features):
            return np.arange(len(unlabeled_features))

        # Validate model has predict_proba method
        try:
            probabilities = model.predict_proba(unlabeled_features)
        except AttributeError:
            raise ValueError("Model must have predict_proba method for uncertainty computation")

        # Determine reference samples for novelty computation
        if self.novelty_type == "source_only":
            if source_features is None or len(source_features) == 0:
                raise ValueError("source_only novelty requires source_features parameter")
            reference_features = source_features
            reference_sample_ids = source_sample_ids
        elif self.novelty_type == "total":
            if annotated_features is None or len(annotated_features) == 0:
                raise ValueError("total novelty requires annotated_features parameter")
            reference_features = annotated_features
            reference_sample_ids = annotated_sample_ids
        else:
            raise ValueError(f"Unknown novelty_type: {self.novelty_type}")

        # Validate feature dimensions
        if unlabeled_features.shape[1] != reference_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: unlabeled ({unlabeled_features.shape[1]}) "
                f"vs reference ({reference_features.shape[1]})"
            )

        # Compute uncertainty scores for all unlabeled samples
        uncertainty_scores = self._compute_sample_uncertainties(probabilities)

        # Compute novelty scores for all unlabeled samples
        novelty_scores = self._compute_sample_novelties(
            unlabeled_features, reference_features, unlabeled_sample_ids, reference_sample_ids
        )

        # Normalize scores to [0, 1] range
        normalized_uncertainties = self._normalize_scores(uncertainty_scores)
        normalized_novelties = self._normalize_scores(novelty_scores)

        # Compute combined scores
        combined_scores = (
            self.uncertainty_weight * normalized_uncertainties + 
            self.novelty_weight * normalized_novelties
        )

        # Select top n_samples with highest combined scores
        top_indices = np.argsort(combined_scores)[-n_samples:]
        selected_indices = top_indices[::-1]  # Descending order of combined score

        return selected_indices

    def _compute_sample_uncertainties(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores for individual samples

        Args:
            probabilities: Prediction probabilities of shape (n_samples, n_classes)

        Returns:
            Array of uncertainty scores (higher = more uncertain)
        """
        if self.uncertainty_measure == "entropy":
            return self._calculate_entropy(probabilities)
        elif self.uncertainty_measure == "margin":
            return self._calculate_margin(probabilities)
        else:
            raise ValueError(f"Unknown uncertainty measure: {self.uncertainty_measure}")

    def _compute_sample_novelties(
        self,
        unlabeled_features: np.ndarray,
        reference_features: np.ndarray,
        unlabeled_sample_ids: Optional[np.ndarray] = None,
        reference_sample_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute novelty scores for individual samples.
        Novelty is defined as the distance to the nearest reference sample.

        Args:
            unlabeled_features: Features of unlabeled samples (n_unlabeled, n_features)
            reference_features: Features of reference samples (n_reference, n_features)
            unlabeled_sample_ids: Optional sample IDs for caching
            reference_sample_ids: Optional sample IDs for caching

        Returns:
            Array of novelty scores (higher = more novel)
        """
        n_unlabeled = len(unlabeled_features)
        novelty_scores = np.zeros(n_unlabeled)

        # Check if we can use caching
        use_caching = (
            unlabeled_sample_ids is not None and 
            reference_sample_ids is not None and
            len(unlabeled_sample_ids) == n_unlabeled and
            len(reference_sample_ids) == len(reference_features)
        )

        for i, unlabeled_feature in enumerate(unlabeled_features):
            min_distance_to_reference = float('inf')

            for j, reference_feature in enumerate(reference_features):
                if use_caching:
                    # Safe to subscript since use_caching validates not None
                    assert unlabeled_sample_ids is not None and reference_sample_ids is not None
                    distance = self._compute_cached_distance(
                        unlabeled_sample_ids[i],
                        reference_sample_ids[j],
                        unlabeled_feature,
                        reference_feature
                    )
                else:
                    distance = self._compute_distance(unlabeled_feature, reference_feature)

                min_distance_to_reference = min(min_distance_to_reference, distance)

            novelty_scores[i] = min_distance_to_reference

        return novelty_scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization

        Args:
            scores: Array of scores to normalize

        Returns:
            Array of normalized scores
        """
        if len(scores) == 0:
            return scores

        min_val = np.min(scores)
        max_val = np.max(scores)

        # Handle case where all scores are the same
        if max_val == min_val:
            return np.ones_like(scores)

        # Min-max normalization
        normalized = (scores - min_val) / (max_val - min_val)
        return normalized

    def _compute_distance(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute distance between two feature vectors without caching

        Args:
            features1: First feature vector
            features2: Second feature vector

        Returns:
            Distance between the feature vectors
        """
        return pairwise_distances(
            features1.reshape(1, -1), 
            features2.reshape(1, -1), 
            metric=self.distance_metric
        )[0, 0]

    def _compute_cached_distance(
        self,
        sample_id1: str,
        sample_id2: str,
        features1: np.ndarray,
        features2: np.ndarray,
    ) -> float:
        """
        Compute distance between two samples with caching

        Args:
            sample_id1: First sample ID
            sample_id2: Second sample ID
            features1: First sample features
            features2: Second sample features

        Returns:
            Distance between the samples
        """
        # Check cache first
        cached_distance = self._get_cached_distance(sample_id1, sample_id2)
        if cached_distance is not None:
            return cached_distance

        # Compute distance
        distance = self._compute_distance(features1, features2)

        # Cache and return
        self._set_cached_distance(sample_id1, sample_id2, distance)
        return distance

    def _get_cached_distance(self, sample_id1: str, sample_id2: str) -> Optional[float]:
        """
        Get cached distance between two sample IDs

        Args:
            sample_id1: First sample ID
            sample_id2: Second sample ID

        Returns:
            Cached distance or None if not found
        """
        cache_key = f"novelty_{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        return self._distance_cache.get(cache_key, None)

    def _set_cached_distance(self, sample_id1: str, sample_id2: str, distance: float) -> None:
        """
        Set cached distance between two sample IDs

        Args:
            sample_id1: First sample ID
            sample_id2: Second sample ID
            distance: Distance value to cache
        """
        cache_key = f"novelty_{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        self._distance_cache[cache_key] = distance

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
            entropy_per_label = -(probabilities * np.log2(probabilities) + 
                                 (1 - probabilities) * np.log2(1 - probabilities))
            
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

    def clear_distance_cache(self) -> None:
        """Clear the distance cache"""
        self._distance_cache.clear()