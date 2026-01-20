import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd

from flc.flows.dataset.split import DatasetSplit
from flc.flows.feature_extraction.flow_summary import Direction
from flc.flows.labels import FlowLabelType
from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.group_labels import FlowGroupLabel
from flc.shared.preprocessing.preprocessor import FeaturePreprocessor
from flc.shared.preprocessing.config import PreprocessingConfig

# Default features to exclude when loading datasets
DEFAULT_EXCLUDED_FEATURES = ["protocols", "start_timestamp", "end_timestamp"]

# Default feature mappings to apply when loading datasets
DEFAULT_FEATURE_MAPPINGS = {"direction": Direction.to_idx}

# Default features to exclude from preprocessing
DEFAULT_PREPROCESSING_EXCLUDED_FEATURES = ["direction"]


class FlowClassificationDataset:
    """
    Dataset class for loading and accessing flow classification datasets.

    This class loads flow classification datasets from a dataset directory containing:
    - config.yaml (DatasetSplit configuration)
    - features.csv (flow features with integer flow_id)
    - labels.csv (flow labels with integer flow_id)
    - id2pcap.csv (mapping from integer flow_id to original pcap filenames)

    The dataset enforces that every flow must have at least one label for consistent
    machine learning workflows.
    """

    def __init__(
        self,
        dataset_path: str,
        exclude_features: Optional[List[str]] = DEFAULT_EXCLUDED_FEATURES,
        feature_mappings: Optional[Dict[str, Callable]] = DEFAULT_FEATURE_MAPPINGS,
    ):
        """
        Initialize dataset from a dataset directory.

        Args:
            dataset_path: Path to the dataset directory containing config.yaml,
                         features.csv, labels.csv, and id2pcap.csv
            exclude_features: List of feature names to exclude from loading.
                             Defaults to DEFAULT_EXCLUDED_FEATURES.
                             Pass empty list [] to include all features.
                             Raises ValueError if specified feature doesn't exist.
            feature_mappings: Dictionary mapping feature names to transformation functions.
                             Defaults to DEFAULT_FEATURE_MAPPINGS.
                             Pass empty dict {} to disable all mappings.
                             Raises ValueError if specified feature doesn't exist.

        Raises:
            ValueError: If dataset directory doesn't exist or data validation fails
            FileNotFoundError: If required data files are missing
        """
        # Store dataset path and validate it exists
        self.dataset_path = str(Path(dataset_path).resolve())
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path must be a directory: {self.dataset_path}")

        # Load split configuration from config.yaml
        config_file = os.path.join(self.dataset_path, "config.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.split = DatasetSplit.from_yaml(config_file)
        self.label_type = self.split.label_type
        self.exclude_features = exclude_features or []
        self.feature_mappings = feature_mappings or {}

        # Setup file paths for new structure
        self._setup_file_paths()

        # Load and validate data
        self._load_features()
        self._load_labels()
        self._validate_data()

        # Initialize preprocessor (not fitted)
        self._preprocessor: Optional[FeaturePreprocessor] = None
        self._exclude_from_preprocessing: List[str] = []

        # Initialize id2pcap mapping as None (loaded on demand)
        self._id2pcap_mapping: Optional[Dict[int, str]] = None

    def _setup_file_paths(self) -> None:
        """Setup file paths for new dataset structure."""
        # All files are directly in the dataset directory
        self.features_file = os.path.join(self.dataset_path, "features.csv")
        self.labels_file = os.path.join(self.dataset_path, "labels.csv")
        self.id2pcap_file = os.path.join(self.dataset_path, "id2pcap.csv")

    def _load_features(self) -> None:
        """Load flow features from CSV file."""
        if not os.path.exists(self.features_file):
            raise FileNotFoundError(f"Features file not found: {self.features_file}")

        # Load features CSV directly - no index-based loading needed in new structure
        self.features = pd.read_csv(self.features_file)

        # Validate flow_id column exists
        if "flow_id" not in self.features.columns:
            raise ValueError("Features file must have 'flow_id' column")

        # Set flow_id as index
        self.features = self.features.set_index("flow_id")

        # Filter out excluded features
        if self.exclude_features:
            for feature_name in self.exclude_features:
                if feature_name not in self.features.columns:
                    raise ValueError(
                        f"Feature exclusion specified for '{feature_name}' but column does not exist in dataset. "
                        f"Available columns: {list(self.features.columns)}"
                    )
            self.features = self.features.drop(columns=self.exclude_features)

        # Apply feature mappings
        if self.feature_mappings:
            for feature_name, mapping_func in self.feature_mappings.items():
                if feature_name not in self.features.columns:
                    raise ValueError(
                        f"Feature mapping specified for '{feature_name}' but column does not exist in dataset. "
                        f"Available columns: {list(self.features.columns)}"
                    )
                self.features[feature_name] = self.features[feature_name].apply(mapping_func)

    def _load_labels(self) -> None:
        """Load flow labels from CSV file."""
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        # Load labels CSV using the appropriate enum method
        if self.label_type == FlowLabelType.FLOW:
            labels_df = FlowLabel.labels_from_csv(self.labels_file, add_enum_col=False)
        elif self.label_type == FlowLabelType.FLOW_GROUP:
            labels_df = FlowGroupLabel.labels_from_csv(self.labels_file, add_enum_col=False)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        # Set flow_id as index and keep only label_idxs column
        labels_df = labels_df.set_index("flow_id")
        self.labels = labels_df[["label_idxs"]]

    def _load_id2pcap_mapping(self) -> Dict[int, str]:
        """Load ID to PCAP filename mapping from CSV file on demand."""
        if self._id2pcap_mapping is not None:
            return self._id2pcap_mapping

        if not os.path.exists(self.id2pcap_file):
            # Return empty mapping if file doesn't exist
            self._id2pcap_mapping = {}
            return self._id2pcap_mapping

        # Load id2pcap mapping
        id2pcap_df = pd.read_csv(self.id2pcap_file)

        # Validate required columns
        if "flow_id" not in id2pcap_df.columns or "pcap_filename" not in id2pcap_df.columns:
            raise ValueError("id2pcap.csv must have 'flow_id' and 'pcap_filename' columns")

        # Create mapping dictionary and cache it
        self._id2pcap_mapping = dict(zip(id2pcap_df["flow_id"], id2pcap_df["pcap_filename"]))
        return self._id2pcap_mapping

    def _validate_data(self) -> None:
        """Validate that every flow has features and labels."""
        # Check feature-label alignment
        feature_flow_ids = set(self.features.index)
        label_flow_ids = set(self.labels.index)

        missing_features = label_flow_ids - feature_flow_ids
        missing_labels = feature_flow_ids - label_flow_ids

        if missing_features:
            raise ValueError(f"Missing features for {len(missing_features)} flows: {list(missing_features)[:5]}...")

        if missing_labels:
            raise ValueError(f"Missing labels for {len(missing_labels)} flows: {list(missing_labels)[:5]}...")

        # Validate every flow has at least one label
        flows_without_labels = []
        for flow_id in self.features.index:
            labels = self.labels.loc[flow_id, "label_idxs"]
            if not labels or len(labels) == 0:
                flows_without_labels.append(flow_id)

        if flows_without_labels:
            raise ValueError(
                f"Dataset validation failed: {len(flows_without_labels)} flows lack labels: "
                f"{flows_without_labels[:5]}..."
            )

        # Ensure consistent ordering
        common_flow_ids = sorted(feature_flow_ids.intersection(label_flow_ids))
        self.features = self.features.loc[common_flow_ids]
        self.labels = self.labels.loc[common_flow_ids]

    def get_features(self, preprocessed: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """
        Get flow features.

        Args:
            preprocessed: If True, return preprocessed features as numpy array.
                         If False, return raw features as DataFrame.

        Returns:
            Features as DataFrame (raw) or numpy array (preprocessed)

        Raises:
            ValueError: If preprocessed=True but preprocessing not applied
        """
        if preprocessed:
            return self._apply_preprocessing_on_demand()
        return self.features

    def get_labels(self) -> pd.DataFrame:
        """
        Get flow labels DataFrame.

        Returns:
            DataFrame with flow_id as index and label_idxs column containing
            lists of label IDs for each flow
        """
        return self.labels

    def get_labels_multihot(self) -> np.ndarray:
        """
        Get labels as multi-hot encoded numpy array.

        Returns:
            Binary matrix where each row corresponds to a flow and each column
            to a label class. Value 1 indicates the flow has that label.
            Shape: (n_flows, n_label_classes)
        """
        # Determine number of classes based on label type
        if self.label_type == FlowLabelType.FLOW:
            n_classes = len(FlowLabel)
        elif self.label_type == FlowLabelType.FLOW_GROUP:
            n_classes = len(FlowGroupLabel)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        # Create multi-hot encoding
        n_flows = len(self.labels)
        multihot = np.zeros((n_flows, n_classes), dtype=np.int32)

        for i, (flow_id, row) in enumerate(self.labels.iterrows()):
            label_ids = row["label_idxs"]
            if label_ids:  # Should always be true due to validation
                multihot[i, label_ids] = 1

        return multihot

    def get_label_names(self) -> Dict[int, str]:
        """
        Get mapping of label IDs to label names.

        Returns:
            Dictionary mapping label ID (int) to label name (str)
        """
        if self.label_type == FlowLabelType.FLOW:
            return {label.value: label.name for label in FlowLabel}
        elif self.label_type == FlowLabelType.FLOW_GROUP:
            return {label.value: label.name for label in FlowGroupLabel}
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

    def set_preprocessor(
        self,
        preprocessor: Optional[FeaturePreprocessor] = None,
        config: Optional[PreprocessingConfig] = None,
        exclude_from_preprocessing: Optional[List[str]] = DEFAULT_PREPROCESSING_EXCLUDED_FEATURES,
        **preprocessor_kwargs,
    ) -> "FlowClassificationDataset":
        """
        Set a preprocessor to be applied on-demand.

        Args:
            preprocessor: Pre-configured FeaturePreprocessor instance
            config: PreprocessingConfig instance to create a new FeaturePreprocessor
            exclude_from_preprocessing: Features to exclude from preprocessing.
                                       Defaults to DEFAULT_PREPROCESSING_EXCLUDED_FEATURES.
            **preprocessor_kwargs: Legacy arguments for creating PreprocessingConfig
                                  (e.g., scaler_type, clip_quantiles, log_transform, etc.)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If neither preprocessor nor config is provided and no legacy kwargs
        """
        if preprocessor is not None:
            self._preprocessor = preprocessor
        elif config is not None:
            # Create preprocessor from config
            self._preprocessor = FeaturePreprocessor(config)
        elif preprocessor_kwargs:
            # Legacy support: create config from kwargs
            config = PreprocessingConfig(
                enabled=preprocessor_kwargs.get("enabled", True),
                scaler_type=preprocessor_kwargs.get("scaler_type", "standard"),
                clip_quantiles=preprocessor_kwargs.get("clip_quantiles"),
                log_transform=preprocessor_kwargs.get("log_transform", False),
            )
            self._preprocessor = FeaturePreprocessor(config)
        else:
            raise ValueError("Must provide either 'preprocessor', 'config', or legacy kwargs")

        self._exclude_from_preprocessing = exclude_from_preprocessing or []
        return self

    def _apply_preprocessing_on_demand(self) -> np.ndarray:
        """
        Apply preprocessing on-demand and return features in original column order.

        Returns:
            Preprocessed features with excluded features in original positions

        Raises:
            ValueError: If no preprocessor is configured
        """
        if self._preprocessor is None:
            raise ValueError("No preprocessor configured. Call set_preprocessor() first.")

        # Get original column order
        original_columns = list(self.features.columns)
        excluded_features = self._exclude_from_preprocessing or []

        # Create result array
        n_flows, n_features = self.features.shape
        result = np.zeros((n_flows, n_features), dtype=np.float64)

        # Get and process non-excluded features
        preprocessed_columns = [col for col in original_columns if col not in excluded_features]
        if preprocessed_columns:
            features_for_preprocessing = self.features[preprocessed_columns]

            # Transform (stateless) - preprocessor handles enabled/disabled logic
            preprocessed_data = self._preprocessor.transform(features_for_preprocessing)

            # Place in original positions - preprocessed_data is a DataFrame
            for col_name in preprocessed_columns:
                original_pos = original_columns.index(col_name)
                result[:, original_pos] = preprocessed_data[col_name].values

        # Fill excluded features in original positions
        for col_name in excluded_features:
            if col_name in original_columns:
                original_pos = original_columns.index(col_name)
                result[:, original_pos] = self.features[col_name].values

        return result

    def to_sklearn_format(self, preprocessed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) tuple for scikit-learn compatibility.

        Args:
            preprocessed: If True, apply preprocessing on-demand and return processed features.
                         If False, return raw features as numpy array.

        Returns:
            Tuple of (features, labels) as numpy arrays in original column order.
            Features shape: (n_flows, n_features)
            Labels shape: (n_flows, n_label_classes) - multi-hot encoded
        """
        if preprocessed:
            X = self._apply_preprocessing_on_demand()
        else:
            X = self.features.values

        y = self.get_labels_multihot()
        return X, y

    def get_preprocessor(self) -> Optional[FeaturePreprocessor]:
        """Get the configured (unfitted) preprocessor."""
        return self._preprocessor

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names in the dataset."""
        return list(self.features.columns)

    @property
    def preprocessor_config(self) -> Optional[PreprocessingConfig]:
        """Get the preprocessing configuration from the configured preprocessor."""
        return self._preprocessor.get_config() if self._preprocessor is not None else None

    def get_excluded_from_preprocessing(self) -> List[str]:
        """Get list of features excluded from preprocessing."""
        return self._exclude_from_preprocessing or []

    def get_preprocessing_mask(
        self, exclude_from_preprocessing: List[str] = DEFAULT_PREPROCESSING_EXCLUDED_FEATURES
    ) -> np.ndarray:
        """
        Get boolean mask indicating which features would be preprocessed.

        Args:
            exclude_from_preprocessing: Optional list of features to exclude from preprocessing.
                                       If None, uses the instance's exclude list.

        Returns:
            Boolean array where True indicates the feature would be preprocessed,
            False indicates it would be excluded. Array length matches number of features.
        """

        feature_names = self.feature_names
        mask = np.ones(len(feature_names), dtype=bool)

        for feature_name in exclude_from_preprocessing:
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                mask[feature_idx] = False

        return mask

    def has_preprocessor(self) -> bool:
        """Check if a preprocessor is configured."""
        return self._preprocessor is not None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics computed from actual loaded data.

        Returns:
            Dictionary with dataset statistics including flow counts,
            label counts, and label distribution
        """
        # Compute statistics from actual loaded data
        total_flows = len(self.features)
        flows_with_labels = len(self.labels)
        flows_without_labels = total_flows - flows_with_labels

        # Count unique labels from actual data
        unique_labels_set = set()
        for _, row in self.labels.iterrows():
            unique_labels_set.update(row["label_idxs"])
        unique_labels = len(unique_labels_set)

        # Compute label statistics from actual data
        label_names = self.get_label_names()
        label_distribution = self.get_label_distribution()

        label_statistics = {}
        for label_id in unique_labels_set:
            label_name = label_names[label_id]
            count = label_distribution.get(label_name, 0)
            percentage = (count / total_flows * 100) if total_flows > 0 else 0.0
            label_statistics[label_id] = {"count": count, "percentage": percentage, "name": label_name}

        return {
            "total_flows": total_flows,
            "flows_with_labels": flows_with_labels,
            "flows_without_labels": flows_without_labels,
            "unique_labels": unique_labels,
            "label_statistics": label_statistics,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get full dataset metadata with static info from split and dynamic statistics from actual data.

        Returns:
            Dictionary with complete dataset metadata including split info,
            creation details, and current statistics
        """
        return {
            "dataset_name": self.split.dataset_name,
            "split_name": self.split.split_name,
            "split_type": self.split.split_type.value,
            "label_type": self.split.label_type.value,
            "created_at": self.split.created_at,
            "split_parameters": {
                "split_method": self.split.split_parameters.split_method.value,
                "split_ratio": self.split.split_parameters.split_ratio,
                "random_seed": self.split.split_parameters.random_seed,
                "parent_dataset_path": self.split.split_parameters.parent_dataset_path,
                "stratify_by": self.split.split_parameters.stratify_by,
            },
            "statistics": self.get_statistics(),
            "file_paths": {
                "features_file": self.features_file,
                "labels_file": self.labels_file,
                "id2pcap_file": self.id2pcap_file,
                "dataset_path": self.dataset_path,
            },
        }

    def get_flow_ids(self) -> List[int]:
        """
        Get list of flow IDs in the dataset.

        Returns:
            List of flow IDs (integers)
        """
        return list(self.features.index)

    def get_pcap_filename(self, flow_id: int) -> Optional[str]:
        """
        Get original PCAP filename for a flow ID.

        Args:
            flow_id: Integer flow ID

        Returns:
            Original PCAP filename or None if mapping not available
        """
        mapping = self._load_id2pcap_mapping()
        return mapping.get(flow_id)

    def get_pcap_mapping(self) -> Dict[int, str]:
        """
        Get the complete ID to PCAP filename mapping.

        Returns:
            Dictionary mapping integer flow_id to original PCAP filename
        """
        mapping = self._load_id2pcap_mapping()
        return mapping.copy()

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get label frequency distribution.

        Returns:
            Dictionary mapping label names to their occurrence counts
        """
        label_names = self.get_label_names()
        label_counts = {}

        for _, row in self.labels.iterrows():
            for label_id in row["label_idxs"]:
                label_name = label_names[label_id]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1

        return label_counts

    def __len__(self) -> int:
        """Return number of flows in the dataset."""
        return len(self.features)

    def subsample(
        self,
        max_flows: int,
        random_seed: Optional[int] = None,
    ) -> "FlowClassificationDataset":
        """
        Create a subsampled FlowClassificationDataset from this dataset.

        Args:
            max_flows: Maximum number of flows to include in the subsampled dataset
            random_seed: Random seed for reproducible subsampling. If None, uses random sampling.

        Returns:
            New FlowClassificationDataset with at most max_flows flows

        Raises:
            ValueError: If max_flows is less than 1
        """
        from flc.flows.dataset.subsample import subsample_dataset

        return subsample_dataset(self, max_flows, random_seed)

    def shuffle(self, random_state: Optional[int] = None) -> "FlowClassificationDataset":
        """
        Shuffle the samples in the dataset.

        Args:
            random_state: Random state for reproducible shuffling. If None, uses random shuffling.

        Returns:
            Self for method chaining
        """
        if random_state is None:
            random_state = np.random.randint(0, 2**32 - 1)

        # Get current indices and shuffle them
        rng = np.random.default_rng(random_state)
        shuffled_indices = rng.permutation(self.features.index)

        # Reorder both features and labels with shuffled indices
        self.features = self.features.loc[shuffled_indices]
        self.labels = self.labels.loc[shuffled_indices]

        return self

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"FlowClassificationDataset("
            f"dataset_name='{self.split.dataset_name}', "
            f"split_name='{self.split.split_name}', "
            f"label_type='{self.label_type.value}', "
            f"n_flows={len(self)}, "
            f"n_features={len(self.features.columns)}, "
            f"preprocessor_configured={self.has_preprocessor()})"
        )
