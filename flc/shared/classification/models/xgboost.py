from typing import Optional, Union
import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import Field
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

from ..base import ClassificationModel, ClassificationConfig


@dataclass
class XGBoostConfig(ClassificationConfig):
    """Configuration for XGBoost classifier"""
    n_estimators: int = Field(100, description="Number of boosting rounds")
    max_depth: int = Field(6, description="Maximum depth of trees")
    learning_rate: float = Field(0.3, description="Boosting learning rate")
    subsample: float = Field(1.0, description="Subsample ratio of training instances")
    colsample_bytree: float = Field(1.0, description="Subsample ratio of columns when constructing each tree")
    colsample_bylevel: float = Field(1.0, description="Subsample ratio of columns for each level")
    colsample_bynode: float = Field(1.0, description="Subsample ratio of columns for each node")
    reg_alpha: float = Field(0.0, description="L1 regularization term on weights")
    reg_lambda: float = Field(1.0, description="L2 regularization term on weights")
    gamma: float = Field(0.0, description="Minimum loss reduction required to make further partition")
    min_child_weight: int = Field(1, description="Minimum sum of instance weight needed in a child")
    max_delta_step: int = Field(0, description="Maximum delta step allowed for each tree's weight estimation")
    scale_pos_weight: Optional[float] = Field(None, description="Balancing of positive and negative weights")
    objective: Optional[str] = Field(None, description="Learning objective")
    eval_metric: Optional[str] = Field(None, description="Evaluation metric")
    early_stopping_rounds: Optional[int] = Field(None, description="Early stopping rounds")
    n_jobs: Optional[int] = Field(1, description="Number of parallel threads")


class XGBoostModel(ClassificationModel):
    """XGBoost classifier supporting both single-label and multi-label classification"""
    
    def __init__(self, config: XGBoostConfig):
        super().__init__(config)
        self.config: XGBoostConfig = config
    
    def fit(self, features: np.ndarray, labels: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'XGBoostModel':
        """
        Fit the XGBoost model to the features and labels
        
        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target labels - 1D array for single-label or 2D binary array for multi-label
            sample_weight: Sample weights of shape (n_samples,). If None, samples are equally weighted.
            
        Returns:
            Self for method chaining
        """
        labels = self._validate_labels(labels)
        
        # Determine objective based on problem type
        if self._is_multi_label:
            objective = self.config.objective or "binary:logistic"
        else:
            if self._n_classes == 2:
                objective = self.config.objective or "binary:logistic"
            else:
                objective = self.config.objective or "multi:softprob"
        
        # Create base XGBoost classifier
        base_model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            colsample_bylevel=self.config.colsample_bylevel,
            colsample_bynode=self.config.colsample_bynode,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            gamma=self.config.gamma,
            min_child_weight=self.config.min_child_weight,
            max_delta_step=self.config.max_delta_step,
            scale_pos_weight=self.config.scale_pos_weight,
            objective=objective,
            eval_metric=self.config.eval_metric,
            early_stopping_rounds=self.config.early_stopping_rounds,
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
        Get feature importance from XGBoost
        
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
    
    def get_booster(self):
        """
        Get the underlying XGBoost booster object
        
        Returns:
            XGBoost booster or None for multi-label models
        """
        if not self._is_fitted:
            return None
        
        if self._is_multi_label:
            # For multi-label, return list of boosters
            return [estimator.get_booster() for estimator in self._model.estimators_]
        else:
            return self._model.get_booster()