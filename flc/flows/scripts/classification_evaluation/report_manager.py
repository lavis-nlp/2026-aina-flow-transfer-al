"""Report management for classification evaluation results"""

import os
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class ReportManager:
    """Manages CSV reports and aggregates evaluation results"""

    def __init__(self, report_path: str):
        self.report_path = report_path
        self.results: List[Dict[str, Any]] = []

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a single evaluation result"""
        self.results.append(result)

    def save_csv_report(self) -> None:
        """Save all results to CSV report"""
        if not self.results:
            print("No results to save")
            return

        # Prepare CSV data
        csv_data = []
        for result in self.results:
            # Extract train metrics
            train_metrics = result["train_metrics"]

            # Create row for train split
            train_row = {
                "timestamp": result["timestamp"],
                "train_dataset_name": result["train_dataset_name"],
                "test_dataset_name": result["train_dataset_name"],
                "split_type": result["train_split_type"],
                "model_name": result["model_name"],
                "hyperparameters": json.dumps(result["hyperparameters"], sort_keys=True),
                "predictions_path": result["predictions_path"],
                "run_output_path": result["run_output_path"],
                "n_flows": result["train_flows"],
            }
            # Add metrics
            train_row.update(self._flatten_metrics(train_metrics))
            csv_data.append(train_row)

            # Create rows for each test dataset
            for test_result in result["test_results"]:
                test_row = {
                    "timestamp": result["timestamp"],
                    "train_dataset_name": result["train_dataset_name"],
                    "test_dataset_name": test_result["test_dataset_name"],
                    "split_type": test_result["test_split_type"],
                    "model_name": result["model_name"],
                    "hyperparameters": json.dumps(result["hyperparameters"], sort_keys=True),
                    "predictions_path": result["predictions_path"],
                    "run_output_path": result["run_output_path"],
                    "n_flows": test_result["test_flows"],
                }
                # Add metrics
                test_row.update(self._flatten_metrics(test_result["test_metrics"]))
                csv_data.append(test_row)

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

        # Save CSV
        df.to_csv(self.report_path, index=False)
        print(f"Saved CSV report to: {self.report_path}")

    def _flatten_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten metrics dictionary to individual columns"""
        flattened = {}

        # Handle different metric types based on what's available
        metric_mappings = {
            # Multi-label metrics
            "exact_match_accuracy": "exact_match_accuracy",
            "hamming_loss": "hamming_loss",
            "macro_precision": "macro_precision",
            "macro_recall": "macro_recall",
            "macro_f1": "macro_f1",
            "micro_precision": "micro_precision",
            "micro_recall": "micro_recall",
            "micro_f1": "micro_f1",
            # Single-label metrics
            "accuracy": "accuracy",
            "weighted_precision": "weighted_precision",
            "weighted_recall": "weighted_recall",
            "weighted_f1": "weighted_f1",
        }

        for metric_key, column_name in metric_mappings.items():
            if metric_key in metrics:
                value = metrics[metric_key]
                # Convert numpy types to float
                if isinstance(value, np.floating):
                    value = float(value)
                flattened[column_name] = value
            else:
                flattened[column_name] = None

        return flattened

    def print_summary(self) -> None:
        """Print summary of all results"""
        if not self.results:
            print("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("CLASSIFICATION EVALUATION SUMMARY")
        print("=" * 80)

        # Group results by train dataset and model
        dataset_models = {}
        for result in self.results:
            train_dataset_name = result["train_dataset_name"]
            model_name = result["model_name"]
            key = f"{train_dataset_name}_{model_name}"

            if key not in dataset_models:
                dataset_models[key] = {
                    "train_dataset_name": train_dataset_name,
                    "model_name": model_name,
                    "hyperparameters": result["hyperparameters"],
                    "train_metrics": result["train_metrics"],
                    "test_results": result["test_results"],
                    "train_flows": result["train_flows"],
                }

        # Print summary for each dataset-model combination
        for key, data in dataset_models.items():
            print(f"\nTrain Dataset: {data['train_dataset_name']}")
            print(f"Model: {data['model_name']}")
            print(f"Hyperparameters: {json.dumps(data['hyperparameters'], sort_keys=True)}")
            print(f"Train flows: {data['train_flows']}")

            # Print train metrics
            train_metrics = data["train_metrics"]
            print("\nTrain Metrics:")
            if "accuracy" in train_metrics:
                print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Train Macro F1: {train_metrics['macro_f1']:.4f}")
            else:
                print(f"  Train Exact Match Accuracy: {train_metrics['exact_match_accuracy']:.4f}")
                print(f"  Train Hamming Loss: {train_metrics['hamming_loss']:.4f}")
                print(f"  Train Macro F1: {train_metrics['macro_f1']:.4f}")

            # Print test results for each test dataset
            print("\nTest Results:")
            for test_result in data["test_results"]:
                test_metrics = test_result["test_metrics"]
                print(f"  Test Dataset: {test_result['test_dataset_name']} ({test_result['test_flows']} flows)")

                if "accuracy" in test_metrics:
                    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"    Macro F1: {test_metrics['macro_f1']:.4f}")
                    print(f"    Micro F1: {test_metrics['micro_f1']:.4f}")
                else:
                    print(f"    Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
                    print(f"    Hamming Loss: {test_metrics['hamming_loss']:.4f}")
                    print(f"    Macro F1: {test_metrics['macro_f1']:.4f}")
                    print(f"    Micro F1: {test_metrics['micro_f1']:.4f}")

            print("-" * 60)

        print(f"\nTotal experiments: {len(self.results)}")
        print(f"CSV report saved to: {self.report_path}")
        print("=" * 80)
