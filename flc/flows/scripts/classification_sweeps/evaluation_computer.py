import logging
import numpy as np
from typing import Dict, Any, Optional

from flc.shared.classification.evaluation import ClassificationEvaluator


logger = logging.getLogger(__name__)


def evaluate_classification_sweep(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate classification predictions using comprehensive metrics.

    Args:
        y_true: True labels (multi-hot encoded)
        y_pred: Predicted labels (multi-hot encoded)
        y_pred_proba: Prediction probabilities (optional)

    Returns:
        Dictionary containing all evaluation metrics

    Raises:
        ValueError: If inputs are invalid
    """
    logger.debug(f"Evaluating classification: {y_true.shape[0]} samples, {y_true.shape[1]} classes")

    try:
        # Validate inputs
        validate_evaluation_inputs(y_true, y_pred, y_pred_proba)

        # Use the existing classification evaluator
        metrics = ClassificationEvaluator.evaluate(y_true, y_pred)

        # Add additional sweep-specific metrics
        additional_metrics = compute_additional_metrics(y_true, y_pred)
        metrics.update(additional_metrics)

        # Handle any missing or invalid metrics
        metrics = clean_metrics(metrics)

        logger.debug(f"Evaluation completed: {len(metrics)} metrics computed")
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {
            "evaluation_error": str(e),
            "n_samples": y_true.shape[0] if hasattr(y_true, "shape") else None,
            "n_classes": y_true.shape[1] if hasattr(y_true, "shape") and len(y_true.shape) > 1 else None,
        }


def validate_evaluation_inputs(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
) -> None:
    """
    Validate inputs for evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)

    Raises:
        ValueError: If inputs are invalid
    """
    # Check types
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred, np.ndarray):
        raise ValueError("y_pred must be a numpy array")

    # Check shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Check dimensions
    if y_true.ndim != 2:
        raise ValueError(f"Labels must be 2D (multi-hot), got {y_true.ndim}D")

    # Check for empty arrays
    if y_true.shape[0] == 0:
        raise ValueError("Empty label arrays")

    # Check probability array if provided
    if y_pred_proba is not None:
        if not isinstance(y_pred_proba, np.ndarray):
            raise ValueError("y_pred_proba must be a numpy array")
        if y_pred_proba.shape != y_true.shape:
            raise ValueError(f"Probability shape mismatch: {y_pred_proba.shape} vs {y_true.shape}")
        if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
            raise ValueError("Probabilities must be in [0, 1] range")

    # Check for valid values
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0s and 1s (multi-hot encoding)")
    if not np.all(np.isin(y_pred, [0, 1])):
        raise ValueError("y_pred must contain only 0s and 1s (multi-hot encoding)")


def compute_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute additional metrics specific to classification sweeps.

    Args:
        y_true: True labels (multi-hot)
        y_pred: Predicted labels (multi-hot)

    Returns:
        Dictionary with additional metrics
    """
    metrics = {}

    # Label statistics
    metrics["n_samples"] = y_true.shape[0]
    metrics["n_classes"] = y_true.shape[1]

    # True label statistics
    true_labels_per_sample = np.sum(y_true, axis=1)
    metrics["mean_true_labels_per_sample"] = np.mean(true_labels_per_sample)
    metrics["std_true_labels_per_sample"] = np.std(true_labels_per_sample)
    metrics["max_true_labels_per_sample"] = np.max(true_labels_per_sample)

    # Predicted label statistics
    pred_labels_per_sample = np.sum(y_pred, axis=1)
    metrics["mean_pred_labels_per_sample"] = np.mean(pred_labels_per_sample)
    metrics["std_pred_labels_per_sample"] = np.std(pred_labels_per_sample)
    metrics["max_pred_labels_per_sample"] = np.max(pred_labels_per_sample)

    # Class frequency statistics
    true_class_frequencies = np.sum(y_true, axis=0)
    pred_class_frequencies = np.sum(y_pred, axis=0)

    metrics["min_true_class_frequency"] = np.min(true_class_frequencies)
    metrics["max_true_class_frequency"] = np.max(true_class_frequencies)
    metrics["mean_true_class_frequency"] = np.mean(true_class_frequencies)

    metrics["min_pred_class_frequency"] = np.min(pred_class_frequencies)
    metrics["max_pred_class_frequency"] = np.max(pred_class_frequencies)
    metrics["mean_pred_class_frequency"] = np.mean(pred_class_frequencies)

    # Label prediction coverage
    classes_predicted = np.sum(pred_class_frequencies > 0)
    classes_with_true_labels = np.sum(true_class_frequencies > 0)

    metrics["classes_predicted"] = int(classes_predicted)
    metrics["classes_with_true_labels"] = int(classes_with_true_labels)
    metrics["class_prediction_coverage"] = (
        classes_predicted / classes_with_true_labels if classes_with_true_labels > 0 else 0.0
    )

    # Perfect predictions
    exact_matches = np.all(y_true == y_pred, axis=1)
    metrics["exact_match_ratio"] = np.mean(exact_matches)

    # Samples with no predictions
    no_predictions = np.sum(y_pred, axis=1) == 0
    metrics["no_prediction_ratio"] = np.mean(no_predictions)

    # Samples with no true labels (edge case)
    no_true_labels = np.sum(y_true, axis=1) == 0
    metrics["no_true_labels_ratio"] = np.mean(no_true_labels)

    return metrics


def clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metrics dictionary by handling NaN/infinite values.

    Args:
        metrics: Raw metrics dictionary

    Returns:
        Cleaned metrics dictionary
    """
    cleaned = {}

    for key, value in metrics.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                cleaned[key] = None
            else:
                cleaned[key] = float(value)
        elif isinstance(value, np.ndarray):
            # Convert arrays to lists for JSON serialization
            if value.size == 1:
                cleaned[key] = float(value.item())
            else:
                cleaned[key] = value.tolist()
        else:
            cleaned[key] = value

    return cleaned
