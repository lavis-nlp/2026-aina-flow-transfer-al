import os
import csv

import numpy as np
from tqdm import tqdm

from flc.flows.dataset import FlowClassificationDataset
from flc.shared.preprocessing import PreprocessingConfig, FitPreprocessor
from flc.shared.classification import ClassificationFactory, ClassificationEvaluator
from flc.shared.transfer_active_learning.weighting_strategies.factory import WeightingStrategyFactory, WeightingStrategy
from flc.flows.experiments.transfer_active_learning.shared import WeightingDetails

# --- DATA ---
DATASETS = [
    "cic-ids-2012",
    "cic-ids-2017",
    "iscx-vpn-2016-non-vpn",
    "iscx-tor-2016-non-tor",
]


# TRAIN_DATASET_TEMPLATE = "data/datasets/{dataset}_splits/train"
# TEST_DATASET_TEMPLATE = "data/datasets/{dataset}_splits/test"
TRAIN_DATASET_TEMPLATE = "data/group/{dataset}-group_splits/train"
TEST_DATASET_TEMPLATE = "data/group/{dataset}-group_splits/test"


PREPROCESSING = PreprocessingConfig(
    enabled=True,
    scaler_type="robust",
    clip_quantiles=(0.01, 0.99),
    log_transform=True,
)

# --- CLASSIFICATION ---
CLASSIFICATION_ALGORITHM = "random_forest"
CLASSIFICATION_PARAMS = {
    "n_estimators": 5,
    "max_depth": 10,
    "subsample": 0.5,
}

WEIGHTING_STRATEGIES = [
    {
        "name": "uniform",
        "params": {
            #
        },
    },
    {
        "name": "balanced",
        "params": {
            "source_weight_factor": 0.5,
            "target_weight_factor": 0.5,
        },
    },
]


# --- MISC ---
MAX_FLOWS = 50000
RANDOM_STATES = [42, 123, 1234]
OUTPUT_REPORT_FILE = "paper/group/transfer-source-and-target/test_results.csv"

HEADERS = [
    "source_dataset",
    "target_dataset",
    "test_dataset",
    "random_state",
    "weighting_strategy",
    "micro_f1",
    "macro_f1",
    "num_source_flows",
    "num_target_flows",
    "num_test_flows",
]


def init_components(seed: int, weighting_strategy_dict: dict):

    # weighting
    weighting_strategy = WeightingStrategyFactory.create(
        strategy_name=weighting_strategy_dict["name"],
        config=weighting_strategy_dict["params"],
    )

    return weighting_strategy


def train_classifier(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    seed: int,
    weighting: WeightingStrategy,
):
    # get source data
    source_features = source_X.copy()
    source_labels = source_y.copy()

    # get target data
    target_features = target_X.copy()
    target_labels = target_y.copy()

    # compute sample weights
    source_sample_weight, target_sample_weight = weighting.compute_weights(
        source_features=source_features,
        target_features=target_features,
        iteration=-1,
    )

    # stack with source data
    combined_features = np.vstack([source_features, target_features])
    combined_labels = np.concatenate([source_labels, target_labels])
    source_weights = np.array([source_sample_weight] * len(source_features))
    target_weights = np.array([target_sample_weight] * len(target_features))
    combined_weights = np.concatenate([source_weights, target_weights])

    # Shuffle combined data
    indices = np.arange(len(combined_features))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    combined_features = combined_features[indices]
    combined_labels = combined_labels[indices]
    weights = combined_weights[indices]

    clf = ClassificationFactory.create_with_defaults(
        model_name=CLASSIFICATION_ALGORITHM,
        **CLASSIFICATION_PARAMS,
        random_state=seed,
    )

    clf.fit(combined_features, combined_labels, weights)

    return clf, WeightingDetails(
        source_weight_per_sample=source_sample_weight,
        target_weight_per_sample=target_sample_weight,
        source_weight_total=np.sum(source_weights),
        target_weight_total=np.sum(target_weights),
        weight_total=np.sum(combined_weights),
    )


def evaluate_classifier(clf, test_X: np.ndarray, test_y: np.ndarray):
    y_pred = clf.predict(test_X)
    return ClassificationEvaluator.evaluate(y_true=test_y, y_pred=y_pred)


def run_experiment(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_X: np.ndarray,
    target_y_gt: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    weighting_strategy_dict: dict,
):
    # setup
    weighting_strategy = init_components(seed, weighting_strategy_dict)

    # evaluate classifier without any target data
    clf, weighting_details = train_classifier(
        source_X=source_X,
        source_y=source_y,
        target_X=target_X,
        target_y=target_y_gt,
        seed=seed,
        weighting=weighting_strategy,
    )
    scores = evaluate_classifier(clf, test_X, test_y)
    return scores


def main():

    # create output directory if it does not exist
    os.makedirs(os.path.dirname(OUTPUT_REPORT_FILE), exist_ok=True)
    # write headers to output file
    with open(OUTPUT_REPORT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HEADERS)
        writer.writeheader()

    for dataset_combination in tqdm(get_combinations(), desc="Dataset Combinations", unit="combination"):
        for seed in tqdm(RANDOM_STATES, desc="Random States", unit="state"):
            # load datasets
            source_dataset = (
                FlowClassificationDataset(dataset_combination["source"])
                .subsample(max_flows=MAX_FLOWS, random_seed=seed)
                .shuffle(random_state=seed)
            )
            target_dataset = (
                FlowClassificationDataset(dataset_combination["target"])
                .subsample(max_flows=MAX_FLOWS, random_seed=seed)
                .shuffle(random_state=seed)
            )
            test_dataset = FlowClassificationDataset(dataset_combination["test"])
            assert target_dataset.split.dataset_name == test_dataset.split.dataset_name

            source_sample_ids = source_dataset.get_flow_ids()
            source_X, source_y = source_dataset.to_sklearn_format(preprocessed=False)

            target_sample_ids = target_dataset.get_flow_ids()
            target_X, target_y = target_dataset.to_sklearn_format(preprocessed=False)

            test_X, test_y = test_dataset.to_sklearn_format(preprocessed=False)

            # preprocess datasets

            ## preprocess features
            preprocessor = FitPreprocessor(config=PREPROCESSING)
            preprocessor.fit(source_X)

            source_X = preprocessor.transform(source_X)
            target_X = preprocessor.transform(target_X)
            test_X = preprocessor.transform(test_X)

            for ws in tqdm(WEIGHTING_STRATEGIES, desc="Weighting Strategies", unit="strategy"):
                scores = run_experiment(
                    source_X=source_X,
                    source_y=source_y,
                    target_X=target_X,
                    target_y_gt=target_y,
                    test_X=test_X,
                    test_y=test_y,
                    seed=seed,
                    weighting_strategy_dict=ws,
                )

                # write results
                writer = csv.DictWriter(open(OUTPUT_REPORT_FILE, "a"), fieldnames=HEADERS)
                writer.writerow(
                    {
                        "source_dataset": dataset_combination["source"],
                        "target_dataset": dataset_combination["target"],
                        "test_dataset": dataset_combination["test"],
                        "random_state": seed,
                        "weighting_strategy": ws["name"],
                        "micro_f1": round(scores["micro_f1"], 4),
                        "macro_f1": round(scores["macro_f1"], 4),
                        "num_source_flows": len(source_X),
                        "num_target_flows": len(target_X),
                        "num_test_flows": len(test_X),
                    }
                )


def get_combinations():
    combinations = []
    for source_dataset in DATASETS:
        for target_dataset in DATASETS:
            if source_dataset != target_dataset:
                combinations.append(
                    {
                        "source": TRAIN_DATASET_TEMPLATE.format(dataset=source_dataset),
                        "target": TRAIN_DATASET_TEMPLATE.format(dataset=target_dataset),
                        "test": TEST_DATASET_TEMPLATE.format(dataset=target_dataset),
                    }
                )

    return combinations


if __name__ == "__main__":
    main()
