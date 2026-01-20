from typing import Optional
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from ..base import ClassificationModel, ClassificationConfig


@dataclass
class RandomForestConfig(ClassificationConfig):
    """Configuration for Random Forest classifier"""
    n_estimators: int = Field(100, description="Number of trees in the forest")
    criterion: str = Field("gini", description="Function to measure quality of split")
    max_depth: Optional[int] = Field(None, description="Maximum depth of tree")
    min_samples_split: int = Field(2, description="Minimum samples required to split internal node")
    min_samples_leaf: int = Field(1, description="Minimum samples required at leaf node")
    max_features: str = Field("sqrt", description="Number of features to consider when looking for best split")
    max_leaf_nodes: Optional[int] = Field(None, description="Grow trees with max_leaf_nodes in best-first fashion")
    min_impurity_decrease: float = Field(0.0, description="Minimum impurity decrease required for split")
    bootstrap: bool = Field(True, description="Whether bootstrap samples are used when building trees")
    oob_score: bool = Field(False, description="Whether to use out-of-bag samples to estimate generalization score")
    n_jobs: Optional[int] = Field(None, description="Number of jobs to run in parallel")


class RandomForestModel(ClassificationModel):
    """Random Forest classifier supporting both single-label and multi-label classification"""
    
    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.config: RandomForestConfig = config
    
    def fit(self, features: np.ndarray, labels: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'RandomForestModel':
        """
        Fit the Random Forest model to the features and labels
        
        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target labels - 1D array for single-label or 2D binary array for multi-label
            sample_weight: Sample weights of shape (n_samples,). If None, samples are equally weighted.
            
        Returns:
            Self for method chaining
        """
        labels = self._validate_labels(labels)
        
        # Create base Random Forest classifier
        base_model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_impurity_decrease=self.config.min_impurity_decrease,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state
        )
        
        if self._is_multi_label:
            # Use MultiOutputClassifier for multi-label classification
            self._model = MultiOutputClassifier(base_model, n_jobs=self.config.n_jobs)
        else:
            # Use base classifier for single-label classification
            self._model = base_model
        
        # Fit the model
        self._model.fit(features, labels, sample_weight=sample_weight)
        self._is_fitted = True
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for the features
        
        Args:
            features: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels - format matches training labels (1D for single-label, 2D for multi-label)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self._model.predict(features)
        
        # Ensure output format matches input format
        if self._is_multi_label and predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the features
        
        Args:
            features: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._is_multi_label:
            # For multi-label, get probabilities for each output
            probabilities = []
            for estimator in self._model.estimators_:
                proba = estimator.predict_proba(features)
                # Get probability of positive class (class 1)
                if proba.shape[1] == 2:
                    probabilities.append(proba[:, 1])
                else:
                    # If only one class present during training
                    probabilities.append(np.zeros(features.shape[0]))
            
            return np.column_stack(probabilities)
        else:
            # For single-label, return standard probabilities
            return self._model.predict_proba(features)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance from the Random Forest
        
        Returns:
            Feature importance array
        """
        if not self._is_fitted:
            return None
        
        if self._is_multi_label:
            # Average feature importance across all estimators
            importances = []
            for estimator in self._model.estimators_:
                importances.append(estimator.feature_importances_)
            return np.mean(importances, axis=0)
        else:
            return self._model.feature_importances_
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available
        
        Returns:
            OOB score or None if not available
        """
        if not self._is_fitted or not self.config.oob_score:
            return None
        
        if self._is_multi_label:
            # For multi-label, OOB score not directly available
            return None
        else:
            return self._model.oob_score_