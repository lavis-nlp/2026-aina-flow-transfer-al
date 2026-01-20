import json
import random
import os
import csv

import numpy as np
import yaml
from tqdm import tqdm

from flc.shared.utils.numpy_helpers import hot_encoding_to_ints
from flc.flows.dataset import FlowClassificationDataset
from flc.shared.preprocessing import PreprocessingConfig, FitPreprocessor
from flc.shared.classification import ClassificationFactory, ClassificationEvaluator
from flc.shared.transfer_active_learning.query_strategies import QueryStrategyFactory
from flc.shared.transfer_active_learning.label_selection import LabelSelectionFactory, LabelSelectionStrategy

# --- DATA ---
DATASET_COMBINATIONS = [
    # VALID
    # {
    #     "train": "data/datasets/cic-ids-2012_splits/train",
    #     "test": "data/datasets/cic-ids-2012_splits/valid",
    # },
    # {
    #     "train": "data/datasets/cic-ids-2017_splits/train",
    #     "test": "data/datasets/cic-ids-2017_splits/valid",
    # },
    # {
    #     "train": "data/datasets/iscx-vpn-2016-non-vpn_splits/train",
    #     "test": "data/datasets/iscx-vpn-2016-non-vpn_splits/valid",
    # },
    # {
    #     "train": "data/datasets/iscx-tor-2016-non-tor_splits/train",
    #     "test": "data/datasets/iscx-tor-2016-non-tor_splits/valid",
    # },
    # TEST
    # {
    #     "train": "data/datasets/cic-ids-2012_splits/train",
    #     "test": "data/datasets/cic-ids-2012_splits/test",
    # },
    # {
    #     "train": "data/datasets/cic-ids-2017_splits/train",
    #     "test": "data/datasets/cic-ids-2017_splits/test",
    # },
    {
        "train": "data/datasets/iscx-vpn-2016-non-vpn_splits/train",
        "test": "data/datasets/iscx-vpn-2016-non-vpn_splits/test",
    },
    {
        "train": "data/datasets/iscx-tor-2016-non-tor_splits/train",
        "test": "data/datasets/iscx-tor-2016-non-tor_splits/test",
    },
]
MAX_FLOWS = 50000

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

# --- ACTIVE LEARNING ---
QUERY_STRATEGIES = [
    {
        "name": "random",
        "params": {
            #
        },
    },
    {
        "name": "uncertainty",
        "params": {
            "uncertainty_measure": "entropy",
        },
    },
]


# --- MISC ---
NUM_INITIAL = 5
NUM_ITER = 100
RANDOM_STATES = [42, 123, 1234]
EXPERIMENT_NAME = "test"
OUTPUT_DIR_TEMPLATE = (
    "results/active_learning/classifier/{experiment_name}/dataset={dataset}/query={query_strategy}/seed={seed}"
)


def init_components(seed: int, query_strategy_dict: dict):
    # query
    query_strategy_dict["params"]["random_state"] = seed
    query_strategy = QueryStrategyFactory.create_classifier_strategy(
        strategy_name=query_strategy_dict["name"],
        config=query_strategy_dict["params"],
    )

    return query_strategy


def train_classifier(
    all_sample_ids: np.ndarray,
    all_X: np.ndarray,
    assigned_labels: dict[int, np.ndarray],
    seed: int,
):
    mask = np.isin(all_sample_ids, list(assigned_labels.keys()))
    X = all_X[mask]

    selected_sample_ids = all_sample_ids[mask]
    y = np.array([assigned_labels[sample_id] for sample_id in selected_sample_ids])

    clf = ClassificationFactory.create_with_defaults(
        model_name=CLASSIFICATION_ALGORITHM,
        **CLASSIFICATION_PARAMS,
        random_state=seed,
    )

    clf.fit(X, y)

    return clf


def evaluate_classifier(clf, test_X: np.ndarray, test_y: np.ndarray):
    y_pred = clf.predict(test_X)
    return ClassificationEvaluator.evaluate(y_true=test_y, y_pred=y_pred)


def save_experiment_config(
    report_dir,
    target_dataset_name: str,
    target_dataset_path: str,
    test_dataset_name: str,
    test_dataset_path: str,
    max_iterations: int,
    seed: int,
    query_strategy: dict,
    num_initial: int = NUM_INITIAL,
):
    file = f"{report_dir}/experiment_config.yaml"
    output = {
        "target_dataset_name": target_dataset_name,
        "target_dataset_path": target_dataset_path,
        "test_dataset_name": test_dataset_name,
        "test_dataset_path": test_dataset_path,
        "max_iterations": max_iterations,
        "seed": seed,
        "query_strategy_name": query_strategy["name"],
        "query_strategy_params": query_strategy["params"],
        "num_initial": num_initial,
        "classifier_name": CLASSIFICATION_ALGORITHM,
        "classifier_params": CLASSIFICATION_PARAMS,
    }
    with open(file, "w") as f:
        yaml.safe_dump(output, f)


def log_iteration(
    report_dir: str,
    iteration: int,
    selected_sample_ids: list[int],
    selected_labels: dict[int, np.ndarray],
    scores: dict,
):
    eval_file = f"{report_dir}/evaluations.csv"
    iterations_file = f"{report_dir}/iterations.csv"

    # write iterations
    iterations_header = ["iteration", "selected_sample_ids", "selected_labels"]
    if not os.path.exists(iterations_file):
        with open(iterations_file, "w") as f:
            f.write(",".join(iterations_header) + "\n")

    writer = csv.DictWriter(open(iterations_file, "a"), fieldnames=iterations_header)

    selected_sample_ids = [str(sample_id) for sample_id in selected_sample_ids]
    t_selected_labels = {int(sample_id): hot_encoding_to_ints(label) for sample_id, label in selected_labels.items()}

    row = {
        "iteration": iteration,
        "selected_sample_ids": json.dumps(selected_sample_ids),
        "selected_labels": json.dumps(t_selected_labels),
    }
    writer.writerow(row)

    # write evaluations
    eval_header = ["iteration", "metrics"]
    if not os.path.exists(eval_file):
        with open(eval_file, "w") as f:
            f.write(",".join(eval_header) + "\n")

    writer = csv.DictWriter(open(eval_file, "a"), fieldnames=eval_header)

    row = {
        "iteration": iteration,
        "metrics": json.dumps(scores),
    }
    writer.writerow(row)


def run_experiment(
    report_dir: str,
    target_sample_ids: list[int],
    target_X: np.ndarray,
    target_y_gt: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    query_strategy_dict: dict,
):
    # setup
    sample_ids = np.array(target_sample_ids)
    query_strategy = init_components(seed, query_strategy_dict)
    assigned_labels: dict[int, np.ndarray] = {}  # sample_id -> assigned label
    sample_id2label = {sample_id: label for sample_id, label in zip(sample_ids, target_y_gt)}

    # intial labeling - select random samples
    initial_sample_ids = random.Random(seed).sample(target_sample_ids, NUM_INITIAL)
    initial_assigned_labels = {sample_id: sample_id2label[sample_id] for sample_id in initial_sample_ids}

    clf = train_classifier(
        all_sample_ids=sample_ids,
        all_X=target_X,
        assigned_labels=initial_assigned_labels,
        seed=seed,
    )
    scores = evaluate_classifier(clf, test_X, test_y)

    log_iteration(
        report_dir=report_dir,
        iteration=-1,
        selected_sample_ids=initial_sample_ids,
        selected_labels=assigned_labels,
        scores=scores,
    )

    for iteration in tqdm(range(NUM_ITER), desc="Running AL loops", unit="loop"):
        # query sample

        unlabeled_mask = ~np.isin(sample_ids, list(assigned_labels.keys()))
        unlabeled_features = target_X[unlabeled_mask]
        selected_idxs = query_strategy.select_samples(
            model=clf,
            unlabeled_features=unlabeled_features,
            n_samples=1,
        )
        selected_sample_ids = sample_ids[unlabeled_mask][selected_idxs]
        assert len(selected_sample_ids) == len(set(selected_sample_ids)) == 1, selected_sample_ids
        assert set(selected_sample_ids).isdisjoint(assigned_labels.keys())

        iter_labels = {sample_id: sample_id2label[sample_id] for sample_id in selected_sample_ids}
        assigned_labels.update(iter_labels)

        # train classifier
        clf = train_classifier(
            all_sample_ids=sample_ids,
            all_X=target_X,
            assigned_labels=assigned_labels,
            seed=seed,
        )

        # evaluate classifier
        scores = evaluate_classifier(clf, test_X, test_y)

        # log iteration
        log_iteration(
            report_dir=report_dir,
            iteration=iteration,
            selected_sample_ids=selected_sample_ids,
            selected_labels=iter_labels,
            scores=scores,
        )


def main():
    for dataset_combination in tqdm(DATASET_COMBINATIONS, desc="Dataset Combinations", unit="combination"):
        for seed in tqdm(RANDOM_STATES, desc="Random States", unit="state"):
            # load datasets
            target_dataset = (
                FlowClassificationDataset(dataset_combination["train"])
                .subsample(max_flows=MAX_FLOWS, random_seed=seed)
                .shuffle(random_state=seed)
            )
            target_sample_ids = target_dataset.get_flow_ids()
            target_X, target_y = target_dataset.to_sklearn_format(preprocessed=False)

            test_X, test_y = FlowClassificationDataset(dataset_combination["test"]).to_sklearn_format(
                preprocessed=False
            )

            # preprocessing
            preprocess_features_mask = target_dataset.get_preprocessing_mask()  # only preprocess certain features
            preprocessor = FitPreprocessor(config=PREPROCESSING)
            preprocessor.fit(target_X[:, preprocess_features_mask])

            target_X[:, preprocess_features_mask] = preprocessor.transform(target_X[:, preprocess_features_mask])
            test_X[:, preprocess_features_mask] = preprocessor.transform(test_X[:, preprocess_features_mask])

            for qs in tqdm(QUERY_STRATEGIES, desc="Query Strategies", unit="strategy"):

                report_dir = OUTPUT_DIR_TEMPLATE.format(
                    dataset=target_dataset.split.dataset_name,
                    query_strategy=qs["name"],
                    experiment_name=EXPERIMENT_NAME,
                    seed=seed,
                )
                os.makedirs(report_dir, exist_ok=True)

                save_experiment_config(
                    report_dir=report_dir,
                    max_iterations=NUM_ITER,
                    target_dataset_name=target_dataset.split.dataset_name,
                    target_dataset_path=dataset_combination["train"],
                    test_dataset_name=target_dataset.split.dataset_name,
                    test_dataset_path=dataset_combination["test"],
                    seed=seed,
                    query_strategy=qs,
                )

                run_experiment(
                    report_dir=report_dir,
                    target_sample_ids=target_sample_ids,
                    target_X=target_X,
                    target_y_gt=target_y,
                    test_X=test_X,
                    test_y=test_y,
                    seed=seed,
                    query_strategy_dict=qs,
                )


if __name__ == "__main__":
    main()
