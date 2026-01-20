"""Evaluation metrics for classification algorithms"""

from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report as sklearn_classification_report,
    hamming_loss,
)


class ClassificationEvaluator:
    """Stateless classification evaluation class for single-label and multi-label classification"""

    @staticmethod
    def get_metric_names(multi_label: bool = False) -> List[str]:
        """
        Get the names of all metrics that are computed by evaluate

        Args:
            multi_label: If True, return multi-label metrics; if False, return single-label metrics

        Returns:
            List of metric names
        """
        if multi_label:
            return [
                "exact_match_accuracy",
                "hamming_loss",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
            ]
        else:
            return [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            ]

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        compute_per_label_metrics: bool = False,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, Dict]]:
        """
        Comprehensive evaluation function for both single-label and multi-label classification

        Args:
            y_true: True labels
            y_pred: Predicted labels
            compute_per_label_metrics: If True, compute metrics for each label/class individually
            class_names: Optional list of class names for per-label metrics

        Returns:
            Dictionary containing various metrics and optionally per-label metrics
        """
        is_multi_label = y_true.ndim == 2

        if is_multi_label:
            # Multi-label classification
            metrics = {
                "exact_match_accuracy": ClassificationEvaluator._multi_label_accuracy(y_true, y_pred),
                "hamming_loss": hamming_loss(y_true, y_pred),
                "macro_precision": ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, precision_score
                ),
                "macro_recall": ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, recall_score
                ),
                "macro_f1": ClassificationEvaluator._compute_macro_metric_with_support(y_true, y_pred, f1_score),
                "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
                "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
            }

            if compute_per_label_metrics:
                per_label_metrics = {}
                n_labels = y_true.shape[1]
                label_names = class_names if class_names else [f"label_{i}" for i in range(n_labels)]

                for i, label_name in enumerate(label_names):
                    per_label_metrics[label_name] = {
                        "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                        "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                        "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                        "support": np.sum(y_true[:, i]),
                    }

                metrics["per_label"] = per_label_metrics
        else:
            # Single-label classification
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, precision_score
                ),
                "macro_recall": ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, recall_score
                ),
                "macro_f1": ClassificationEvaluator._compute_macro_metric_with_support(y_true, y_pred, f1_score),
                "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
                "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }

            if compute_per_label_metrics:
                per_class_metrics = {}
                unique_labels = np.unique(y_true)
                label_names = class_names if class_names else [f"class_{label}" for label in unique_labels]

                for i, label in enumerate(unique_labels):
                    class_name = label_names[i] if i < len(label_names) else f"class_{label}"
                    y_true_binary = (y_true == label).astype(int)
                    y_pred_binary = (y_pred == label).astype(int)

                    per_class_metrics[class_name] = {
                        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                        "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
                        "support": np.sum(y_true == label),
                    }

                metrics["per_class"] = per_class_metrics

        return metrics

    @staticmethod
    def classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_dict: bool = True,
        compute_per_label_metrics: bool = False,
    ) -> Union[str, Dict]:
        """
        Generate a comprehensive classification report using sklearn

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            output_dict: If True, return as dict; if False, return as string
            compute_per_label_metrics: If True, include detailed per-label metrics

        Returns:
            Classification report (dict or string)
        """
        target_names = class_names

        if y_true.ndim == 2:
            # Multi-label classification - compute per-label metrics
            reports = {}
            if target_names is None:
                target_names = [f"label_{i}" for i in range(y_true.shape[1])]

            if compute_per_label_metrics:
                for i, label_name in enumerate(target_names):
                    label_report = sklearn_classification_report(y_true[:, i], y_pred[:, i], output_dict=output_dict)
                    reports[label_name] = label_report

            # Add overall metrics with corrected macro computation
            reports["hamming_loss"] = hamming_loss(y_true, y_pred)
            reports["exact_match_ratio"] = accuracy_score(y_true, y_pred)

            # Add corrected macro metrics
            if output_dict:
                reports["macro_avg"] = {
                    "precision": ClassificationEvaluator._compute_macro_metric_with_support(
                        y_true, y_pred, precision_score
                    ),
                    "recall": ClassificationEvaluator._compute_macro_metric_with_support(y_true, y_pred, recall_score),
                    "f1-score": ClassificationEvaluator._compute_macro_metric_with_support(y_true, y_pred, f1_score),
                    "support": np.sum(y_true),
                }

            return reports
        else:
            # Single-label classification
            report = sklearn_classification_report(y_true, y_pred, target_names=target_names, output_dict=output_dict)

            # Replace sklearn's macro averages with corrected ones
            if output_dict and "macro avg" in report:
                report["macro avg"]["precision"] = ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, precision_score
                )
                report["macro avg"]["recall"] = ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, recall_score
                )
                report["macro avg"]["f1-score"] = ClassificationEvaluator._compute_macro_metric_with_support(
                    y_true, y_pred, f1_score
                )

            return report

    @staticmethod
    def _multi_label_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute exact match accuracy for multi-label classification

        Args:
            y_true: True binary labels of shape (n_samples, n_classes)
            y_pred: Predicted binary labels of shape (n_samples, n_classes)

        Returns:
            Exact match accuracy
        """
        return np.mean(np.all(y_pred == y_true, axis=1))

    @staticmethod
    def _compute_macro_metric_with_support(
        y_true: np.ndarray, y_pred: np.ndarray, metric_func, **kwargs
    ) -> np.floating:
        """
        Compute macro-averaged metric considering only labels with support > 0

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_func: Metric function (precision_score, recall_score, f1_score)
            **kwargs: Additional arguments for metric function

        Returns:
            Macro-averaged metric excluding classes with 0 support
        """
        if y_true.ndim == 2:
            # Multi-label: compute per-label and average only those with support > 0
            scores = []
            for i in range(y_true.shape[1]):
                if np.sum(y_true[:, i]) > 0:  # Only include labels with support > 0
                    score = metric_func(y_true[:, i], y_pred[:, i], zero_division=0, **kwargs)
                    scores.append(score)
            return np.mean(scores) if scores else np.float64(0.0)
        else:
            # Single-label: compute per-class and average only those with support > 0
            unique_labels = np.unique(y_true)
            scores = []
            for label in unique_labels:
                if np.sum(y_true == label) > 0:  # Only include classes with support > 0
                    y_true_binary = (y_true == label).astype(int)
                    y_pred_binary = (y_pred == label).astype(int)
                    score = metric_func(y_true_binary, y_pred_binary, zero_division=0, **kwargs)
                    scores.append(score)
            return np.mean(scores) if scores else np.float64(0.0)
