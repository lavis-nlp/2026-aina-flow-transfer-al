import time
import logging
import numpy as np
from typing import Dict, Any, Tuple

from flc.shared.classification import ClassificationFactory
from flc.shared.classification.base import ClassificationModel


logger = logging.getLogger(__name__)


def execute_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    """
    Execute classification model training and prediction.

    Args:
        X_train: Training features
        y_train: Training labels (multi-hot encoded)
        X_test: Test features
        y_test: Test labels (multi-hot encoded)
        algorithm_name: Name of classification algorithm
        hyperparameters: Hyperparameters for the algorithm
        random_state: Random state for reproducibility

    Returns:
        Tuple of (y_pred, y_pred_proba, model_summary, execution_time)
        - y_pred: Predicted labels (multi-hot encoded)
        - y_pred_proba: Prediction probabilities
        - model_summary: Dictionary with model information
        - execution_time: Time taken for training and prediction

    Raises:
        ValueError: If algorithm is not supported or parameters are invalid
        RuntimeError: If model training or prediction fails
    """
    logger.debug(f"Executing {algorithm_name} with hyperparameters: {hyperparameters}")

    # Validate inputs
    validate_classification_inputs(X_train, y_train, X_test, y_test, algorithm_name, hyperparameters)

    # Start timing
    start_time = time.time()

    try:
        # Create model with hyperparameters
        model = create_classification_model(algorithm_name, hyperparameters, random_state)

        # Train model
        logger.debug(f"Training {algorithm_name} on {X_train.shape[0]} samples")
        model.fit(X_train, y_train)

        # Make predictions
        logger.debug(f"Predicting on {X_test.shape[0]} test samples")
        y_pred = model.predict(X_test)

        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            logger.debug(f"Prediction probabilities not available for {algorithm_name}")
            y_pred_proba = None

        # Calculate execution time
        execution_time = time.time() - start_time

        # Generate model summary
        model_config = model.config
        model_summary = generate_model_summary(model, algorithm_name, model_config.__dict__, X_train, y_train)

        logger.debug(f"Classification completed in {execution_time:.2f}s")

        return y_pred, y_pred_proba, model_summary, execution_time

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Classification failed for {algorithm_name}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_classification_model(
    algorithm_name: str, hyperparameters: Dict[str, Any], random_state: int
) -> ClassificationModel:
    """
    Create a classification model using the factory.

    Args:
        algorithm_name: Name of the classification algorithm
        hyperparameters: Hyperparameters for the model
        random_state: Random state for reproducibility

    Returns:
        Configured classification model

    Raises:
        ValueError: If algorithm is not supported
    """
    # Add random_state to hyperparameters if the model supports it
    model_hyperparams = dict(hyperparameters)

    # Most scikit-learn models support random_state
    if algorithm_name.lower() in ["decision_tree", "random_forest", "xgboost"]:
        model_hyperparams["random_state"] = random_state
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}. Supported: {get_supported_algorithms()}")

    try:
        model = ClassificationFactory.create_with_defaults(algorithm_name, **model_hyperparams)
        logger.info(f"Created {algorithm_name} model with hyperparameters: {model_hyperparams}")
        return model
    except Exception as e:
        raise ValueError(f"Failed to create {algorithm_name} model: {str(e)}") from e


def validate_classification_inputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithm_name: str,
    hyperparameters: Dict[str, Any],
) -> None:
    """
    Validate inputs for classification execution.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        algorithm_name: Algorithm name
        hyperparameters: Algorithm hyperparameters

    Raises:
        ValueError: If any input is invalid
    """
    # Check array shapes and types
    if not isinstance(X_train, np.ndarray):
        raise ValueError("X_train must be a numpy array")
    if not isinstance(y_train, np.ndarray):
        raise ValueError("y_train must be a numpy array")
    if not isinstance(X_test, np.ndarray):
        raise ValueError("X_test must be a numpy array")
    if not isinstance(y_test, np.ndarray):
        raise ValueError("y_test must be a numpy array")

    # Check dimensions
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")
    if y_train.ndim != 2:
        raise ValueError(f"y_train must be 2D (multi-hot), got shape {y_train.shape}")
    if y_test.ndim != 2:
        raise ValueError(f"y_test must be 2D (multi-hot), got shape {y_test.shape}")

    # Check sample consistency
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train and y_train sample count mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test and y_test sample count mismatch: {X_test.shape[0]} vs {y_test.shape[0]}")

    # Check feature consistency
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature count mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")

    # Check label consistency
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError(f"Label count mismatch: train={y_train.shape[1]}, test={y_test.shape[1]}")

    # Check for empty datasets
    if X_train.shape[0] == 0:
        raise ValueError("Training set is empty")
    if X_test.shape[0] == 0:
        raise ValueError("Test set is empty")

    # Check algorithm name
    if not algorithm_name or not isinstance(algorithm_name, str):
        raise ValueError("Algorithm name must be a non-empty string")

    # Check hyperparameters
    if not isinstance(hyperparameters, dict):
        raise ValueError("Hyperparameters must be a dictionary")

    # Check for NaN/infinite values
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("X_train contains NaN or infinite values")
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        raise ValueError("X_test contains NaN or infinite values")
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        raise ValueError("y_train contains NaN or infinite values")
    if np.any(np.isnan(y_test)) or np.any(np.isinf(y_test)):
        raise ValueError("y_test contains NaN or infinite values")


def generate_model_summary(
    model: ClassificationModel,
    algorithm_name: str,
    algorithm_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Generate summary information about the trained model.

    Args:
        model: Trained classification model
        algorithm_name: Name of the algorithm
        hyperparameters: Hyperparameters used (only swept parameters)
        X_train: Training features used
        y_train: Training labels used

    Returns:
        Dictionary with model summary information
    """
    # Get ALL hyperparameters by merging swept params with model config defaults
    # all_hyperparameters = get_all_hyperparameters(algorithm_name, hyperparameters)

    summary = {
        "algorithm": algorithm_name,
        "hyperparameters": algorithm_config,
        "n_train_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "n_classes": y_train.shape[1],
        "multi_label": True,  # Always multi-label in our setup
    }

    # Add feature importance if available
    try:
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            summary["has_feature_importance"] = True
            summary["feature_importance_shape"] = feature_importance.shape
            # Store top feature indices for analysis
            if len(feature_importance.shape) == 1:
                # Single importance vector
                top_features = np.argsort(feature_importance)[-10:].tolist()
                summary["top_10_features"] = top_features
            else:
                # Per-class importance matrix
                summary["feature_importance_per_class"] = True
        else:
            summary["has_feature_importance"] = False
    except (AttributeError, NotImplementedError):
        summary["has_feature_importance"] = False

    # Add algorithm-specific information
    if algorithm_name.lower() == "random_forest":
        try:
            # Try to get out-of-bag score if available
            oob_score = getattr(model, "get_oob_score", lambda: None)()
            if oob_score is not None:
                summary["oob_score"] = oob_score
        except (AttributeError, ValueError):
            pass

    elif algorithm_name.lower() == "xgboost":
        try:
            # Get number of boosting rounds actually used
            booster = getattr(model, "get_booster", lambda: None)()
            if booster is not None:
                summary["n_boosting_rounds"] = len(booster.get_dump())
        except (AttributeError, ValueError):
            pass

    return summary


def get_supported_algorithms() -> list[str]:
    """
    Get list of supported classification algorithms.

    Returns:
        List of algorithm names supported by the factory
    """
    # This would ideally be exposed by the ClassificationFactory
    return ["decision_tree", "random_forest", "xgboost"]


def get_all_hyperparameters(algorithm_name: str, swept_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ALL hyperparameters by merging swept parameters with config defaults.

    Args:
        algorithm_name: Name of the algorithm
        swept_hyperparameters: Hyperparameters that were swept (potentially incomplete)

    Returns:
        Dictionary with all hyperparameters (swept + defaults)
    """
    try:
        # Get the config class for this algorithm
        config_class = ClassificationFactory.get_model_config_class(algorithm_name)

        # Create a config instance with swept parameters to get all defaults
        config = config_class(**swept_hyperparameters)

        # Convert config to dictionary to get all parameters
        all_hyperparameters = {}
        for field_name in config.__dataclass_fields__:
            value = getattr(config, field_name)
            # Convert to JSON-serializable types
            if value is not None:
                all_hyperparameters[field_name] = value

        return all_hyperparameters

    except Exception as e:
        logger.warning(f"Could not get all hyperparameters for {algorithm_name}: {e}")
        # Fallback to swept parameters only
        return swept_hyperparameters


def validate_algorithm_hyperparameters(algorithm_name: str, hyperparameters: Dict[str, Any]) -> None:
    """
    Validate that hyperparameters are appropriate for the given algorithm.

    Args:
        algorithm_name: Name of the algorithm
        hyperparameters: Hyperparameters to validate

    Raises:
        ValueError: If hyperparameters are invalid for the algorithm
    """
    supported_algorithms = get_supported_algorithms()
    if algorithm_name not in supported_algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}. Supported: {supported_algorithms}")

    # Additional validation could be added here for specific algorithms
    # For now, we rely on the factory and model validation
