"""
Classification sweeps module for hyperparameter tuning of flow classification models.

This module provides systematic hyperparameter sweeps across different classification algorithms
on flow classification datasets with support for:
- Multiple algorithms (DecisionTree, RandomForest, XGBoost)
- Configurable hyperparameter grids for each algorithm
- Train/test dataset split evaluation
- Resumable execution (skips already completed experiments)
- Extensible parameter addition without losing previous results
- Comprehensive CSV reporting with all metrics and metadata

Usage:
    python -m flc.flows.scripts.classification_sweeps.run --config path/to/config.yaml
"""