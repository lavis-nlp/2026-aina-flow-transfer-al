"""Core evaluation logic for classification experiments"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from flc.flows.dataset.dataset import FlowClassificationDataset
from flc.shared.classification import ClassificationFactory, ClassificationEvaluator
from .config import DatasetConfig, ModelConfig


class EvaluationExecutor:
    """Executes classification model training and evaluation"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_single_evaluation(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        output_directory: str = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single model evaluation: train once on train dataset, evaluate on all test datasets

        Args:
            dataset_config: Dataset configuration with train and multiple test datasets
            model_config: Model configuration
            output_directory: Optional output directory for detailed results

        Returns:
            Dictionary containing evaluation results and metadata for all test datasets
        """
        timestamp = datetime.now().isoformat()

        # set

        # Load train dataset
        print(f"Loading train dataset: {dataset_config.train_split_path}")
        train_dataset = FlowClassificationDataset(dataset_config.train_split_path)

        # Set preprocessing for train dataset
        train_dataset.set_preprocessor(
            config=dataset_config.get_preprocessing_config(defaults.get("preprocessing", None) if defaults else None)
        )

        # Get training data
        X_train, y_train = train_dataset.to_sklearn_format(preprocessed=True)
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")

        # Create and train model
        print(f"Creating {model_config.name} model with hyperparameters: {model_config.hyperparameters}")
        model = ClassificationFactory.create_with_defaults(
            model_config.name, random_state=self.random_state, **model_config.hyperparameters
        )

        print("Training model...")
        model.fit(X_train, y_train)

        # Make predictions on training data
        y_pred_train = model.predict(X_train)
        train_metrics = ClassificationEvaluator.evaluate(y_train, y_pred_train)

        # Get label names for output
        label_names = train_dataset.get_label_names()

        # Evaluate on all test datasets
        test_results = []
        all_test_predictions = []
        all_test_flow_ids = []

        for test_config in dataset_config.test_datasets:
            print(f"Loading test dataset: {test_config.name} ({test_config.path})")
            test_dataset = FlowClassificationDataset(test_config.path)

            # Set preprocessing for test dataset
            test_dataset.set_preprocessor(
                config=dataset_config.get_preprocessing_config(
                    defaults.get("preprocessing", None) if defaults else None
                )
            )

            # Get test data and make predictions
            X_test, y_test = test_dataset.to_sklearn_format(preprocessed=True)
            print(f"Test data shape for {test_config.name}: {X_test.shape}")

            y_pred_test = model.predict(X_test)
            test_metrics = ClassificationEvaluator.evaluate(y_test, y_pred_test)

            test_result = {
                "test_dataset_name": test_config.name,
                "test_dataset_path": test_config.path,
                "test_split_type": test_dataset.split.split_type.value,
                "test_metrics": test_metrics,
                "test_flows": len(test_dataset),
                "predictions": y_pred_test,
                "flow_ids": test_dataset.get_flow_ids(),
            }
            test_results.append(test_result)

            # Collect for combined predictions file
            all_test_predictions.append(y_pred_test)
            all_test_flow_ids.extend(test_dataset.get_flow_ids())

        # Prepare combined result
        result = {
            "timestamp": timestamp,
            "train_dataset_name": dataset_config.name,
            "train_split_type": train_dataset.split.split_type.value,
            "model_name": model_config.name,
            "hyperparameters": model_config.hyperparameters,
            "train_metrics": train_metrics,
            "test_results": test_results,
            "train_flows": len(train_dataset),
            "n_features": X_train.shape[1],
            "label_names": label_names,
            "predictions_path": None,
            "run_output_path": None,
        }

        # Save detailed outputs if output directory specified
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            run_dir = self._create_run_directory(output_directory, timestamp, dataset_config.name, model_config.name)

            # Combine all test predictions
            all_test_predictions_combined = (
                np.vstack(all_test_predictions) if all_test_predictions else np.empty((0, y_pred_train.shape[1]))
            )

            # Save predictions
            predictions_path = self._save_predictions(
                run_dir,
                train_dataset.get_flow_ids(),
                all_test_flow_ids,
                y_pred_train,
                all_test_predictions_combined,
                label_names,
            )

            # Save detailed run report
            run_report_path = self._save_run_report(run_dir, result, train_dataset)

            result["predictions_path"] = predictions_path
            result["run_output_path"] = run_report_path

        return result

    def _create_run_directory(self, output_directory: str, timestamp: str, dataset_name: str, model_name: str) -> str:
        """Create timestamped run directory"""
        # Clean timestamp for filesystem
        clean_timestamp = timestamp.replace(":", "-").replace(".", "-")
        run_dir_name = f"{clean_timestamp}_{dataset_name}_{model_name}"
        run_dir = os.path.join(output_directory, run_dir_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _save_predictions(
        self,
        run_dir: str,
        train_flow_ids: list,
        test_flow_ids: list,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        label_names: Dict[int, str],
    ) -> str:
        """Save predictions in same format as dataset labels"""
        predictions_path = os.path.join(run_dir, "predictions.csv")

        # Combine train and test predictions
        all_flow_ids = train_flow_ids + test_flow_ids
        all_predictions = np.vstack([y_pred_train, y_pred_test])

        # Convert multi-hot predictions to label_idxs format
        predictions_data = []
        for flow_id, pred_row in zip(all_flow_ids, all_predictions):
            # Get indices where prediction is 1
            label_idxs = [int(idx) for idx in np.where(pred_row == 1)[0]]
            predictions_data.append({"flow_id": flow_id, "label_idxs": label_idxs})

        # Save as CSV
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(predictions_path, index=False)

        return predictions_path

    def _save_run_report(
        self,
        run_dir: str,
        result: Dict[str, Any],
        train_dataset: FlowClassificationDataset,
    ) -> str:
        """Save detailed run report as JSON"""
        report_path = os.path.join(run_dir, "run_report.json")

        # Create comprehensive report
        test_datasets_info = []
        test_metrics_info = []

        for test_result in result["test_results"]:
            test_datasets_info.append(
                {
                    "name": test_result["test_dataset_name"],
                    "path": test_result["test_dataset_path"],
                    "n_flows": test_result["test_flows"],
                }
            )
            test_metrics_info.append(
                {
                    "dataset_name": test_result["test_dataset_name"],
                    "metrics": {
                        k: float(v) if isinstance(v, np.floating) else v for k, v in test_result["test_metrics"].items()
                    },
                }
            )

        report = {
            "metadata": {
                "timestamp": result["timestamp"],
                "train_dataset_name": result["train_dataset_name"],
                "model_name": result["model_name"],
                "hyperparameters": result["hyperparameters"],
            },
            "dataset_info": {
                "train": {
                    "split_path": train_dataset.split.split_name,
                    "n_flows": result["train_flows"],
                    "statistics": train_dataset.get_statistics(),
                },
                "test_datasets": test_datasets_info,
                "n_features": result["n_features"],
                "label_names": result["label_names"],
            },
            "evaluation": {
                "train_metrics": {
                    k: float(v) if isinstance(v, np.floating) else v for k, v in result["train_metrics"].items()
                },
                "test_metrics": test_metrics_info,
            },
            "outputs": {"predictions_file": result["predictions_path"], "run_report_file": report_path},
        }

        # Save report
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path
