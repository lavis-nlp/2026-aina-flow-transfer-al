from typing import Optional
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

from ..base import ClassificationModel, ClassificationConfig


@dataclass
class DecisionTreeConfig(ClassificationConfig):
    """Configuration for Decision Tree classifier"""
    criterion: str = Field("gini", description="Function to measure quality of split")
    max_depth: Optional[int] = Field(None, description="Maximum depth of tree")
    min_samples_split: int = Field(2, description="Minimum samples required to split internal node")
    min_samples_leaf: int = Field(1, description="Minimum samples required at leaf node")
    max_features: Optional[str] = Field(None, description="Number of features to consider when looking for best split")
    max_leaf_nodes: Optional[int] = Field(None, description="Grow tree with max_leaf_nodes in best-first fashion")
    min_impurity_decrease: float = Field(0.0, description="Minimum impurity decrease required for split")


class DecisionTreeModel(ClassificationModel):
    """Decision Tree classifier supporting both single-label and multi-label classification"""
    
    def __init__(self, config: DecisionTreeConfig):
        super().__init__(config)
        self.config: DecisionTreeConfig = config
    
    def fit(self, features: np.ndarray, labels: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'DecisionTreeModel':
        """
        Fit the Decision Tree model to the features and labels
        
        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target labels - 1D array for single-label or 2D binary array for multi-label
            sample_weight: Sample weights of shape (n_samples,). If None, samples are equally weighted.
            
        Returns:
            Self for method chaining
        """
        labels = self._validate_labels(labels)
        
        # Create base Decision Tree classifier
        base_model = DecisionTreeClassifier(
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_impurity_decrease=self.config.min_impurity_decrease,
            random_state=self.config.random_state
        )
        
        if self._is_multi_label:
            # Use MultiOutputClassifier for multi-label classification
            self._model = MultiOutputClassifier(base_model)
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
        Get feature importance from the Decision Tree
        
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