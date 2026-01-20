import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional, Dict, Any, Union, Tuple
import re
import warnings

from flc.shared.preprocessing.config import PreprocessingConfig


class FeaturePreprocessor:
    """Handles feature preprocessing for clustering with clipping and log transformation"""

    def __init__(
        self,
        config: PreprocessingConfig,
        log_offset: float = 1e-6,
    ):
        """
        Initialize feature preprocessor

        Args:
            config: PreprocessingConfig instance containing preprocessing settings
            log_offset: Small value to add before log transformation to handle zeros
        """
        self._config = config
        self._log_offset = log_offset

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters"""
        if self._config.enabled and self._config.clip_quantiles:
            lower, upper = self._config.clip_quantiles
            if not (0 <= lower < upper <= 1):
                raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

        if self._log_offset <= 0:
            raise ValueError("Log offset must be positive")


    def _apply_clipping(self, X: np.ndarray) -> np.ndarray:
        """Apply clipping using quantiles computed on-the-fly"""
        if not self._config.enabled or self._config.clip_quantiles is None:
            return X

        lower_q, upper_q = self._config.clip_quantiles
        lower_bounds = np.percentile(X, lower_q * 100, axis=0)
        upper_bounds = np.percentile(X, upper_q * 100, axis=0)
        return np.clip(X, lower_bounds, upper_bounds)

    def _apply_log_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply safe log transformation"""
        if not self._config.enabled or not self._config.log_transform:
            return X

        # Handle negative values and zeros
        X_safe = X + self._log_offset

        # Ensure no negative values
        X_safe = np.maximum(X_safe, self._log_offset)

        # Check for warnings about negative values
        if np.any(X < 0):
            warnings.warn(
                f"Negative values detected in features. Adding offset {self._log_offset} before log transformation."
            )

        return np.log(X_safe)

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """Apply scaling using sklearn scalers created on-the-fly"""
        if not self._config.enabled or self._config.scaler_type is None:
            return X

        if self._config.scaler_type == "standard":
            scaler = StandardScaler()
        elif self._config.scaler_type == "minmax":
            scaler = MinMaxScaler()
        elif self._config.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self._config.scaler_type}")

        return scaler.fit_transform(X)

    def transform_df(self, features: pd.DataFrame, only_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform DataFrame features using stateless preprocessing
        
        Args:
            features: Input features as DataFrame with all numeric columns
            only_columns: If specified, only these columns will be preprocessed. Other columns remain unchanged.
            
        Returns:
            Transformed features as DataFrame (preserves column names and index)
            
        Raises:
            ValueError: If non-numeric features are found in columns to be processed
        """
        if not self._config.enabled:
            return features  # Return original DataFrame unchanged
        
        # Determine which columns to process
        if only_columns is not None:
            # Validate that specified columns exist
            missing_cols = [col for col in only_columns if col not in features.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            
            cols_to_process = only_columns
        else:
            cols_to_process = features.columns.tolist()
        
        # Check that columns to process are numeric
        features_to_check = features[cols_to_process]
        non_numeric_cols = features_to_check.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise ValueError(f"Non-numeric features found in columns to process: {non_numeric_cols}. All features must be numeric.")
        
        # Create a copy of the original DataFrame
        result_df = features.copy()
        
        # Extract numpy array data for columns to process
        X = features[cols_to_process].values
        
        # Apply transformation pipeline
        if self._config.clip_quantiles is not None:
            X = self._apply_clipping(X)

        if self._config.log_transform is True:
            X = self._apply_log_transform(X)

        if self._config.scaler_type is not None:
            X = self._apply_scaling(X)
        
        # Update only the processed columns in the result DataFrame
        result_df[cols_to_process] = X
        
        return result_df
    
    def transform_np(self, features: np.ndarray) -> np.ndarray:
        """
        Transform numpy array features using stateless preprocessing
        
        Supports 1D and 2D arrays only. For 2D arrays, preprocessing is applied column-wise.
        
        Args:
            features: Input features as 1D or 2D numpy array with numeric data
            
        Returns:
            Transformed features as numpy array
            
        Raises:
            ValueError: If non-numeric array is provided or array has more than 2 dimensions
        """
        # For numpy arrays, ensure they are numeric
        if not np.issubdtype(features.dtype, np.number):
            raise ValueError(
                f"Non-numeric array provided. Array dtype: {features.dtype}. All features must be numeric."
            )
        
        # Check array dimensions
        if features.ndim > 2:
            raise ValueError(f"Arrays with more than 2 dimensions are not supported. Got {features.ndim}D array.")
        
        if not self._config.enabled:
            return features  # Return original array unchanged
        
        # Handle 1D arrays - sklearn scalers need 2D input
        if features.ndim == 1:
            X = features.reshape(1, -1)
            is_1d = True
        else:
            X = features
            is_1d = False
        
        # Apply transformation pipeline
        if self._config.clip_quantiles is not None:
            X = self._apply_clipping(X)

        if self._config.log_transform is True:
            X = self._apply_log_transform(X)

        if self._config.scaler_type is not None:
            X = self._apply_scaling(X)
        
        # Restore original shape for 1D arrays
        if is_1d:
            X = X.flatten()
        
        return X

    def transform(self, features: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features using stateless preprocessing

        All features must be numeric. Raises ValueError if non-numeric features are found.

        Returns the same type as input: DataFrame input returns DataFrame output,
        numpy array input returns numpy array output.

        Transformation pipeline (when enabled):
        1. Validate all features are numeric
        2. Apply clipping (if enabled)
        3. Apply log transformation (if enabled)
        4. Apply scaling (if enabled)

        Args:
            features: Input features as numpy array or DataFrame with all numeric columns

        Returns:
            Transformed features in same format as input:
            - DataFrame input → DataFrame output (preserves column names and index)
            - numpy array input → numpy array output

        Raises:
            ValueError: If non-numeric features are found
        """
        if isinstance(features, pd.DataFrame):
            return self.transform_df(features)
        else:
            return self.transform_np(features)

    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about configured transformations"""
        return {
            "preprocessing_enabled": self._config.enabled,
            "clipping_enabled": self._config.clip_quantiles is not None,
            "clip_quantiles": self._config.clip_quantiles,
            "log_transform": self._config.log_transform,
            "log_offset": self._log_offset if self._config.log_transform else None,
            "scaler_type": self._config.scaler_type,
        }

    def get_config(self) -> PreprocessingConfig:
        """Get the preprocessing configuration"""
        return self._config


def filter_features(
    feature_dict: Dict[str, Any],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Filter features based on key patterns

    Args:
        feature_dict: Input feature dictionary
        include_patterns: List of regex patterns for features to include
        exclude_patterns: List of regex patterns for features to exclude

    Returns:
        Filtered feature dictionary
    """
    filtered_dict = {}

    for key, value in feature_dict.items():
        include = True

        # Check include patterns
        if include_patterns:
            include = any(re.search(pattern, key) for pattern in include_patterns)

        # Check exclude patterns
        if exclude_patterns and include:
            include = not any(re.search(pattern, key) for pattern in exclude_patterns)

        if include:
            filtered_dict[key] = value

    return filtered_dict


def apply_quantile_clipping(
    X: np.ndarray, lower_q: float = 0.05, upper_q: float = 0.95
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Standalone function for quantile clipping

    Args:
        X: Input data array
        lower_q: Lower quantile for clipping (0-1)
        upper_q: Upper quantile for clipping (0-1)

    Returns:
        Tuple of (clipped_data, (lower_bounds, upper_bounds))
    """
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1")

    lower_bounds = np.percentile(X, lower_q * 100, axis=0)
    upper_bounds = np.percentile(X, upper_q * 100, axis=0)

    clipped_X = np.clip(X, lower_bounds, upper_bounds)

    return clipped_X, (lower_bounds, upper_bounds)


def apply_log_transform(X: np.ndarray, offset: float = 1e-6) -> np.ndarray:
    """
    Standalone function for log transformation

    Args:
        X: Input data array
        offset: Small value to add before log transformation to handle zeros

    Returns:
        Log-transformed data
    """
    if offset <= 0:
        raise ValueError("Offset must be positive")

    # Handle negative values and zeros
    X_safe = X + offset

    # Ensure no negative values
    X_safe = np.maximum(X_safe, offset)

    # Check for warnings about negative values
    if np.any(X < 0):
        warnings.warn(f"Negative values detected in features. Adding offset {offset} before log transformation.")

    return np.log(X_safe)
