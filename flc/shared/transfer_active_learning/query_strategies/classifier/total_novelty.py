from typing import Dict, Optional
import numpy as np
from sklearn.metrics import pairwise_distances

from .base import ClassifierQueryStrategy
from .configs import TotalNoveltyQueryStrategyConfig


class TotalNoveltyQueryStrategy(ClassifierQueryStrategy):
    """
    Query strategy that selects samples furthest from all labeled samples.

    This strategy computes the distance from each unlabeled sample to all labeled samples
    (both source and already-labeled target samples), finds the minimum distance for each
    unlabeled sample (nearest neighbor), then selects the sample(s) with maximum minimum distance.
    This maximizes novelty by selecting samples that are in regions poorly covered by labeled data.
    """

    def __init__(self, config: TotalNoveltyQueryStrategyConfig):
        super().__init__(config)
        self.distance_metric = config.distance_metric

        # Cache for storing distances between sample pairs
        # Key format: "sample_id1---sample_id2---distance_metric"
        self._distance_cache: Dict[str, float] = {}

    def select_samples(
        self,
        model,
        unlabeled_features: np.ndarray,
        labeled_features: np.ndarray,
        n_samples: int = 1,
        unlabeled_sample_ids: Optional[np.ndarray] = None,
        labeled_sample_ids: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Select samples with maximum distance to nearest labeled sample

        Args:
            model: Trained classification model (unused)
            unlabeled_features: Features of unlabeled samples (n_unlabeled, n_features)
            n_samples: Number of samples to select
            labeled_features: Features of all labeled samples (source + target)
            unlabeled_sample_ids: Optional sample IDs for unlabeled samples (for caching)
            labeled_sample_ids: Optional sample IDs for labeled samples (for caching)
            **kwargs: Additional unused arguments for interface compatibility

        Returns:
            Array of selected sample indices in the unlabeled dataset

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Model is unused for distance-based novelty computation
        _ = model
        if len(unlabeled_features) == 0:
            return np.array([], dtype=int)

        if n_samples <= 0:
            return np.array([], dtype=int)

        if n_samples >= len(unlabeled_features):
            return np.arange(len(unlabeled_features))

        if len(labeled_features) == 0:
            raise ValueError("No labeled samples provided - cannot compute novelty")

        # Validate feature dimensions
        if unlabeled_features.shape[1] != labeled_features.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: unlabeled ({unlabeled_features.shape[1]}) "
                f"vs labeled ({labeled_features.shape[1]})"
            )

        # Compute novelty scores for each unlabeled sample
        novelty_scores = self._compute_novelty_scores(
            unlabeled_features, labeled_features, unlabeled_sample_ids, labeled_sample_ids
        )

        # Select top n_samples with highest novelty (maximum distance to nearest labeled)
        top_indices = np.argsort(novelty_scores)[-n_samples:]
        selected_indices = top_indices[::-1]  # Descending order of novelty

        return selected_indices

    def _compute_novelty_scores(
        self,
        unlabeled_features: np.ndarray,
        labeled_features: np.ndarray,
        unlabeled_sample_ids: Optional[np.ndarray] = None,
        labeled_sample_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute novelty scores for each unlabeled sample.
        Novelty is defined as the distance to the nearest labeled sample.

        Args:
            unlabeled_features: Features of unlabeled samples (n_unlabeled, n_features)
            labeled_features: Features of labeled samples (n_labeled, n_features)
            unlabeled_sample_ids: Optional sample IDs for caching
            labeled_sample_ids: Optional sample IDs for caching

        Returns:
            Array of novelty scores (higher = more novel)
        """
        n_unlabeled = len(unlabeled_features)
        novelty_scores = np.zeros(n_unlabeled)

        # Check if we can use caching
        use_caching = (unlabeled_sample_ids is not None and 
                      labeled_sample_ids is not None and
                      len(unlabeled_sample_ids) == n_unlabeled and
                      len(labeled_sample_ids) == len(labeled_features))

        for i, unlabeled_feature in enumerate(unlabeled_features):
            min_distance = float('inf')

            for j, labeled_feature in enumerate(labeled_features):
                if use_caching:
                    # Safe to subscript since use_caching validates not None
                    assert unlabeled_sample_ids is not None and labeled_sample_ids is not None
                    distance = self._compute_cached_distance(
                        unlabeled_sample_ids[i],
                        labeled_sample_ids[j], 
                        unlabeled_feature,
                        labeled_feature
                    )
                else:
                    distance = self._compute_distance(unlabeled_feature, labeled_feature)

                min_distance = min(min_distance, distance)

            novelty_scores[i] = min_distance

        return novelty_scores

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
        cache_key = f"{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        return self._distance_cache.get(cache_key, None)

    def _set_cached_distance(self, sample_id1: str, sample_id2: str, distance: float) -> None:
        """
        Set cached distance between two sample IDs

        Args:
            sample_id1: First sample ID
            sample_id2: Second sample ID
            distance: Distance value to cache
        """
        cache_key = f"{min(sample_id1, sample_id2)}---{max(sample_id1, sample_id2)}---{self.distance_metric}"
        self._distance_cache[cache_key] = distance

    def clear_distance_cache(self) -> None:
        """Clear the distance cache"""
        self._distance_cache.clear()