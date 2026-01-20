import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError
from typing import Optional, Dict, Any, Union, Tuple
import warnings

from flc.shared.preprocessing.config import PreprocessingConfig


class FitPreprocessor(BaseEstimator, TransformerMixin):
    """Feature preprocessor with fit/transform interface for consistent transformations"""

    def __init__(
        self,
        config: PreprocessingConfig,
        log_offset: float = 1e-6,
    ):
        """
        Initialize fit preprocessor

        Args:
            config: PreprocessingConfig instance containing preprocessing settings
            log_offset: Small value to add before log transformation to handle zeros
        """
        self.config = config
        self.log_offset = log_offset

        # Validate configuration
        self._validate_config()

        # Initialize state variables
        self._is_fitted = False
        self._clip_bounds = None
        self._scaler = None

    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.enabled and self.config.clip_quantiles:
            lower, upper = self.config.clip_quantiles
            if not (0 <= lower < upper <= 1):
                raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

        if self.log_offset <= 0:
            raise ValueError("Log offset must be positive")

    def _validate_fitted(self):
        """Check if preprocessor has been fitted"""
        if not self._is_fitted:
            raise NotFittedError(
                "This FitPreprocessor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def _compute_clip_bounds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute clipping bounds from training data"""
        if not self.config.enabled or self.config.clip_quantiles is None:
            return None, None

        lower_q, upper_q = self.config.clip_quantiles
        lower_bounds = np.percentile(X, lower_q * 100, axis=0)
        upper_bounds = np.percentile(X, upper_q * 100, axis=0)
        return lower_bounds, upper_bounds

    def _apply_clipping(self, X: np.ndarray) -> np.ndarray:
        """Apply clipping using fitted bounds"""
        if not self.config.enabled or self._clip_bounds is None:
            return X

        lower_bounds, upper_bounds = self._clip_bounds
        return np.clip(X, lower_bounds, upper_bounds)

    def _apply_log_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply safe log transformation"""
        if not self.config.enabled or not self.config.log_transform:
            return X

        # Handle negative values and zeros
        X_safe = X + self.log_offset

        # Ensure no negative values
        X_safe = np.maximum(X_safe, self.log_offset)

        # Check for warnings about negative values
        if np.any(X < 0):
            warnings.warn(
                f"Negative values detected in features. Adding offset {self.log_offset} before log transformation."
            )

        return np.log(X_safe)

    def _create_scaler(self) -> Optional[object]:
        """Create sklearn scaler based on configuration"""
        if not self.config.enabled or self.config.scaler_type is None:
            return None

        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        Learn transformation parameters from training data

        Args:
            X: Training features as numpy array or DataFrame
            y: Ignored, present for API compatibility

        Returns:
            self: Fitted preprocessor instance
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Check that all columns are numeric
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                raise ValueError(f"Non-numeric features found: {non_numeric_cols}. All features must be numeric.")
            X_array = X.values
        else:
            # Handle numpy array input
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError(f"Non-numeric array provided. Array dtype: {X.dtype}. All features must be numeric.")

            if X.ndim > 2:
                raise ValueError(f"Arrays with more than 2 dimensions are not supported. Got {X.ndim}D array.")

            # Handle 1D arrays
            if X.ndim == 1:
                X_array = X.reshape(1, -1)
            else:
                X_array = X

        if not self.config.enabled:
            self._is_fitted = True
            return self

        # Step 1: Compute clipping bounds
        self._clip_bounds = self._compute_clip_bounds(X_array)

        # Step 2: Apply clipping for scaler fitting
        X_clipped = self._apply_clipping(X_array)

        # Step 3: Apply log transform for scaler fitting
        X_log = self._apply_log_transform(X_clipped)

        # Step 4: Fit scaler
        self._scaler = self._create_scaler()
        if self._scaler is not None:
            self._scaler.fit(X_log)

        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply learned transformations to new data

        Args:
            X: Features to transform as numpy array or DataFrame

        Returns:
            Transformed features in same format as input
        """
        self._validate_fitted()

        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            # Check that all columns are numeric
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                raise ValueError(f"Non-numeric features found: {non_numeric_cols}. All features must be numeric.")

            X_array = X.values
            is_dataframe = True
            columns = X.columns
            index = X.index
        else:
            # Handle numpy array input
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError(f"Non-numeric array provided. Array dtype: {X.dtype}. All features must be numeric.")

            if X.ndim > 2:
                raise ValueError(f"Arrays with more than 2 dimensions are not supported. Got {X.ndim}D array.")

            is_dataframe = False

            # Handle 1D arrays
            if X.ndim == 1:
                X_array = X.reshape(1, -1)
                is_1d = True
            else:
                X_array = X
                is_1d = False

        if not self.config.enabled:
            return X  # Return original data unchanged

        # Apply transformation pipeline
        X_transformed = X_array

        # Step 1: Apply clipping
        X_transformed = self._apply_clipping(X_transformed)

        # Step 2: Apply log transformation
        X_transformed = self._apply_log_transform(X_transformed)

        # Step 3: Apply scaling
        if self._scaler is not None:
            X_transformed = self._scaler.transform(X_transformed)

        # Return in same format as input
        if is_dataframe:
            return pd.DataFrame(X_transformed, columns=columns, index=index)
        else:
            # Restore original shape for 1D arrays
            if is_1d:
                X_transformed = X_transformed.flatten()
            return X_transformed

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit to data, then transform it

        Args:
            X: Training features as numpy array or DataFrame
            y: Ignored, present for API compatibility

        Returns:
            Transformed features in same format as input
        """
        return self.fit(X, y).transform(X)

    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about configured transformations"""
        info = {
            "preprocessing_enabled": self.config.enabled,
            "clipping_enabled": self.config.clip_quantiles is not None,
            "clip_quantiles": self.config.clip_quantiles,
            "log_transform": self.config.log_transform,
            "log_offset": self.log_offset if self.config.log_transform else None,
            "scaler_type": self.config.scaler_type,
            "is_fitted": self._is_fitted,
        }

        if self._is_fitted and self._clip_bounds is not None:
            info["fitted_clip_bounds"] = {
                "lower_bounds": self._clip_bounds[0],
                "upper_bounds": self._clip_bounds[1],
            }

        return info

    def get_config(self) -> PreprocessingConfig:
        """Get the preprocessing configuration"""
        return self.config
