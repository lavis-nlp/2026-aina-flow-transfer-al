import os
import csv
import json
import pandas as pd
from typing import Dict, List, Any, Set, Optional, Tuple
from datetime import datetime

from .sweep_generator import SweepCombination
from .config import DatasetPairConfig
from flc.shared.classification.evaluation import ClassificationEvaluator


class ReportManager:
    """Manages CSV report for classification sweep results with extensible parameter support"""

    def __init__(self, report_path: str):
        """
        Initialize report manager.

        Args:
            report_path: Path to CSV report file
        """
        self.report_path = report_path
        self._ensure_report_directory()

    def _ensure_report_directory(self) -> None:
        """Ensure the directory for the report file exists"""
        report_dir = os.path.dirname(self.report_path)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir)

    def get_completed_sweep_ids(self) -> Set[str]:
        """
        Get set of sweep IDs that have already been completed.

        Returns:
            Set of completed sweep IDs
        """
        if not os.path.exists(self.report_path):
            return set()

        try:
            df = pd.read_csv(self.report_path)
            if "sweep_id" in df.columns:
                return set(df["sweep_id"].dropna().astype(str))
            else:
                return set()
        except Exception:
            return set()

    def get_completed_combinations(self) -> Set[Tuple[str, str, str, str]]:
        """
        Get completed combinations for exact parameter matching.

        Returns:
            Set of (train_dataset_path, test_dataset_path, algorithm, parameter_hash) tuples
        """
        if not os.path.exists(self.report_path):
            return set()

        try:
            df = pd.read_csv(self.report_path)
            required_cols = ["train_dataset_path", "test_dataset_path", "algorithm", "parameter_hash"]
            if all(col in df.columns for col in required_cols):
                completed = set()
                for _, row in df.iterrows():
                    if all(pd.notna(row[col]) for col in required_cols):
                        completed.add(
                            (
                                str(row["train_dataset_path"]),
                                str(row["test_dataset_path"]),
                                str(row["algorithm"]),
                                str(row["parameter_hash"]),
                            )
                        )
                return completed
            else:
                return set()
        except Exception:
            return set()

    def check_combination_completed(self, dataset_pair: DatasetPairConfig, algorithm: str, parameter_hash: str) -> bool:
        """
        Check if exact combination was completed.

        Args:
            dataset_pair: Dataset pair configuration
            algorithm: Algorithm name
            parameter_hash: Hash of hyperparameters

        Returns:
            True if combination was completed
        """
        completed_combinations = self.get_completed_combinations()
        return (
            dataset_pair.train_split_path,
            dataset_pair.test_split_path,
            algorithm,
            parameter_hash,
        ) in completed_combinations

    def append_result(
        self,
        dataset_pair: DatasetPairConfig,
        sweep_combination: SweepCombination,
        dataset_metadata: Dict[str, Any],
        model_summary: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
        execution_time: float,
        config_version: str = "1.0",
        error_message: Optional[str] = None,
    ) -> None:
        """
        Append a single result to the CSV report.

        Args:
            dataset_pair: Dataset pair configuration
            sweep_combination: Sweep combination that was executed
            dataset_metadata: Metadata from dataset loading
            model_summary: Summary from model training
            evaluation_metrics: Evaluation metrics
            execution_time: Time taken for execution
            config_version: Configuration version for tracking
            error_message: Error message if execution failed
        """
        # Create result row
        result_row = self._create_result_row(
            dataset_pair=dataset_pair,
            sweep_combination=sweep_combination,
            dataset_metadata=dataset_metadata,
            model_summary=model_summary,
            evaluation_metrics=evaluation_metrics,
            execution_time=execution_time,
            config_version=config_version,
            error_message=error_message,
        )

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(self.report_path)

        # Determine if this is multi-label classification
        is_multi_label = model_summary["multi_label"]

        # Get column names specific to the classification type
        column_names = self._get_column_names(multi_label=is_multi_label)

        # Write to CSV
        with open(self.report_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=column_names)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write result row
            writer.writerow(result_row)

    def _create_result_row(
        self,
        dataset_pair: DatasetPairConfig,
        sweep_combination: SweepCombination,
        dataset_metadata: Dict[str, Any],
        model_summary: Dict[str, Any],
        evaluation_metrics: Dict[str, Any],
        execution_time: float,
        config_version: str,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a single result row for the CSV report"""

        # Base information
        row = {
            "timestamp": datetime.now().isoformat(),
            "config_version": config_version,
            "train_dataset_path": dataset_pair.train_split_path,
            "test_dataset_path": dataset_pair.test_split_path,
            "algorithm": sweep_combination.algorithm,
            "sweep_id": sweep_combination.sweep_id,
            "parameter_hash": sweep_combination.parameter_hash,
            "execution_time": execution_time,
            "error_message": error_message or "",
        }

        # Dataset metadata
        row.update(
            {
                "n_train_flows": dataset_metadata.get("n_train_flows"),
                "n_test_flows": dataset_metadata.get("n_test_flows"),
                "n_features": dataset_metadata.get("n_features"),
                "n_classes": dataset_metadata.get("n_classes"),
                "dataset_name": dataset_metadata.get("dataset_name"),
                "label_type": dataset_metadata.get("label_type"),
                "train_split_name": dataset_metadata.get("train_split_name"),
                "test_split_name": dataset_metadata.get("test_split_name"),
                "original_n_train_flows": dataset_metadata.get("original_n_train_flows"),
                "max_flows_used": dataset_metadata.get("max_flows_used"),
                "train_flows_sampled": dataset_metadata.get("train_flows_sampled"),
                "preprocessing_enabled": dataset_metadata.get("preprocessing_enabled"),
                "scaler_type": dataset_metadata.get("scaler_type"),
                "clip_quantiles": (
                    json.dumps(dataset_metadata.get("clip_quantiles")) if dataset_metadata.get("clip_quantiles") else ""
                ),
                "log_transform": dataset_metadata.get("log_transform"),
            }
        )

        # Algorithm hyperparameters (as JSON string)
        row["hyperparameters"] = json.dumps(sweep_combination.hyperparameters, sort_keys=True)

        # Model summary
        row.update(
            {
                "n_train_samples": model_summary.get("n_train_samples"),
                "multi_label": model_summary.get("multi_label"),
                "has_feature_importance": model_summary.get("has_feature_importance", False),
            }
        )

        # Add algorithm-specific model info
        if "oob_score" in model_summary:
            row["oob_score"] = model_summary["oob_score"]
        else:
            row["oob_score"] = None

        if "n_boosting_rounds" in model_summary:
            row["n_boosting_rounds"] = model_summary["n_boosting_rounds"]
        else:
            row["n_boosting_rounds"] = None

        # Evaluation metrics - include metrics appropriate for the classification type
        is_multi_label = model_summary.get("multi_label", False)
        metric_columns = self._get_evaluation_metric_columns(is_multi_label)
        for metric in metric_columns:
            row[metric] = evaluation_metrics.get(metric)

        # Handle evaluation errors
        if "evaluation_error" in evaluation_metrics:
            row["evaluation_error"] = evaluation_metrics["evaluation_error"]
        else:
            row["evaluation_error"] = ""

        return row

    def _get_column_names(self, multi_label: bool) -> List[str]:
        """Get standardized column names for the CSV report"""
        base_columns = [
            "timestamp",
            "config_version",
            "train_dataset_path",
            "test_dataset_path",
            "algorithm",
            "hyperparameters",
            "sweep_id",
            "parameter_hash",
        ]

        dataset_columns = [
            "n_train_flows",
            "n_test_flows",
            "n_features",
            "n_classes",
            "dataset_name",
            "label_type",
            "train_split_name",
            "test_split_name",
            "original_n_train_flows",
            "max_flows_used",
            "train_flows_sampled",
            "preprocessing_enabled",
            "scaler_type",
            "clip_quantiles",
            "log_transform",
        ]

        model_columns = ["n_train_samples", "multi_label", "has_feature_importance", "oob_score", "n_boosting_rounds"]

        evaluation_columns = self._get_evaluation_metric_columns(multi_label)

        final_columns = ["execution_time", "error_message", "evaluation_error"]

        return base_columns + dataset_columns + model_columns + evaluation_columns + final_columns

    def _get_evaluation_metric_columns(self, multi_label: bool) -> List[str]:
        """Get list of evaluation metric column names based on what evaluator actually computes"""
        # Use specific metrics based on multi_label flag
        evaluator_metrics = ClassificationEvaluator.get_metric_names(multi_label=multi_label)

        # Additional metrics computed by the sweep evaluation
        additional_sweep_metrics = [
            "n_samples",
            "n_classes",
            "mean_true_labels_per_sample",
            "std_true_labels_per_sample",
            "max_true_labels_per_sample",
            "mean_pred_labels_per_sample",
            "std_pred_labels_per_sample",
            "max_pred_labels_per_sample",
            "min_true_class_frequency",
            "max_true_class_frequency",
            "mean_true_class_frequency",
            "min_pred_class_frequency",
            "max_pred_class_frequency",
            "mean_pred_class_frequency",
            "classes_predicted",
            "classes_with_true_labels",
            "class_prediction_coverage",
            "exact_match_ratio",
            "no_prediction_ratio",
            "no_true_labels_ratio",
        ]

        return evaluator_metrics + additional_sweep_metrics

    def get_report_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from the current report.

        Returns:
            Dictionary with report summary statistics
        """
        if not os.path.exists(self.report_path):
            return {
                "total_experiments": 0,
                "successful_experiments": 0,
                "failed_experiments": 0,
                "algorithms_tested": [],
                "datasets_tested": [],
                "config_versions": [],
            }

        try:
            df = pd.read_csv(self.report_path)

            total_experiments = len(df)
            failed_experiments = len(df[df["error_message"].notna() & (df["error_message"] != "")])
            successful_experiments = total_experiments - failed_experiments

            algorithms_tested = df["algorithm"].unique().tolist()

            # Get unique dataset pairs
            dataset_pairs = df[["train_dataset_path", "test_dataset_path"]].drop_duplicates()
            datasets_tested = [
                f"{row['train_dataset_path']} -> {row['test_dataset_path']}" for _, row in dataset_pairs.iterrows()
            ]

            config_versions = df["config_version"].unique().tolist() if "config_version" in df.columns else []

            return {
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
                "failed_experiments": failed_experiments,
                "algorithms_tested": algorithms_tested,
                "datasets_tested": datasets_tested,
                "config_versions": config_versions,
            }

        except Exception as e:
            return {
                "error": f"Failed to read report: {str(e)}",
                "total_experiments": 0,
                "successful_experiments": 0,
                "failed_experiments": 0,
                "algorithms_tested": [],
                "datasets_tested": [],
                "config_versions": [],
            }

    def load_results_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Load the complete results as a pandas DataFrame.

        Returns:
            DataFrame with all results, or None if file doesn't exist
        """
        if not os.path.exists(self.report_path):
            return None

        try:
            return pd.read_csv(self.report_path)
        except Exception as e:
            raise ValueError(f"Failed to load results: {str(e)}")

    def get_best_results(
        self, metric: str = "f1_macro", top_k: int = 10, algorithm: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get top-k best results by a specific metric.

        Args:
            metric: Metric to sort by
            top_k: Number of top results to return
            algorithm: Filter by specific algorithm (optional)

        Returns:
            DataFrame with top results, or None if no data
        """
        df = self.load_results_dataframe()
        if df is None or len(df) == 0:
            return None

        # Filter by algorithm if specified
        if algorithm:
            df = df[df["algorithm"] == algorithm]

        # Filter out failed experiments
        df = df[df["error_message"].isna() | (df["error_message"] == "")]

        # Check if metric exists
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        # Sort by metric (descending for most metrics)
        df_sorted = df.sort_values(metric, ascending=False, na_last=True)

        return df_sorted.head(top_k)
