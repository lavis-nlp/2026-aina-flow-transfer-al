from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field


@dataclass
class ClassificationConfig:
    """Base configuration for classification models"""

    random_state: int
    multi_label: Optional[bool] = None


class ClassificationModel(ABC):
    """Abstract base class for classification models supporting both single-label and multi-label classification"""

    def __init__(self, config: ClassificationConfig):
        self.config = config
        self._is_fitted = False
        self._model = None
        self._is_multi_label = None
        self._n_classes = None

    def _detect_label_format(self, labels: np.ndarray) -> bool:
        """
        Detect if labels are single-label or multi-label format

        Args:
            labels: Label array to analyze

        Returns:
            True if multi-label, False if single-label
        """
        if self.config.multi_label is not None:
            return self.config.multi_label

        if labels.ndim == 1:
            return False
        elif labels.ndim == 2:
            return True
        else:
            raise ValueError(f"Labels must be 1D or 2D array, got {labels.ndim}D")

    def _validate_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Validate and potentially reshape labels

        Args:
            labels: Input labels

        Returns:
            Validated labels array
        """
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        if labels.ndim == 1:
            self._is_multi_label = False
            self._n_classes = len(np.unique(labels))
        elif labels.ndim == 2:
            self._is_multi_label = True
            self._n_classes = labels.shape[1]
        else:
            raise ValueError(f"Labels must be 1D or 2D array, got {labels.ndim}D")

        return labels

    @abstractmethod
    def fit(
        self, features: np.ndarray, labels: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "ClassificationModel":
        """
        Fit the classification model to the features and labels

        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target labels - 1D array for single-label or 2D binary array for multi-label
            sample_weight: Sample weights of shape (n_samples,). If None, samples are equally weighted.

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the features

        Args:
            features: Input features of shape (n_samples, n_features)

        Returns:
            Predicted labels - format matches training labels (1D for single-label, 2D for multi-label)
        """
        pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the features

        Args:
            features: Input features of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        pass

    def fit_predict(
        self, features: np.ndarray, labels: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit and predict in one step

        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target labels
            sample_weight: Sample weights of shape (n_samples,). If None, samples are equally weighted.

        Returns:
            Predicted labels - format matches input labels
        """
        return self.fit(features, labels, sample_weight).predict(features)

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute accuracy score for the predictions

        Args:
            features: Input features of shape (n_samples, n_features)
            labels: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(features)

        if self._is_multi_label:
            # For multi-label: exact match accuracy
            return np.mean(np.all(predictions == labels, axis=1))
        else:
            # For single-label: standard accuracy
            return np.mean(predictions == labels)

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted"""
        return self._is_fitted

    @property
    def is_multi_label(self) -> Optional[bool]:
        """Check if this is a multi-label classifier"""
        return self._is_multi_label

    @property
    def n_classes(self) -> Optional[int]:
        """Get number of classes"""
        return self._n_classes

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available

        Returns:
            Feature importance array or None if not available
        """
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            # For linear models, use absolute coefficients as importance
            coef = self._model.coef_
            if coef.ndim == 1:
                return np.abs(coef)
            else:
                # For multi-class/multi-label, average across classes
                return np.mean(np.abs(coef), axis=0)
        return None

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization"""
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization"""
        self.__dict__.update(state)
