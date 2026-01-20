import json
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
from flc.shared.transfer_active_learning.weighting_strategies.factory import WeightingStrategyFactory, WeightingStrategy
from flc.flows.experiments.transfer_active_learning.shared import WeightingDetails

# --- DATA ---
DATASETS = [
    "cic-ids-2012",
    "cic-ids-2017",
    "iscx-vpn-2016-non-vpn",
    "iscx-tor-2016-non-tor",
]

ONLY_COMBINATIONS = [
    # {
    #     "source": "iscx-vpn-2016-vpn",
    #     "target": "iscx-vpn-2016-non-vpn",
    # }
    {
        "source": "cic-ids-2012",
    },
    # {
    #     "source": "cic-ids-2017",
    # },
    # {
    #     "source": "iscx-tor-2016-non-tor",
    # },
    # {
    #     "source": "iscx-vpn-2016-non-vpn",
    # },
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
    "n_estimators": 25,
    "max_depth": 10,
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
    # {
    #     "name": "total_novelty",
    #     "params": {
    #         "distance_metric": "euclidean",
    #     },
    # },
    # {
    #     "name": "uncertainty_novelty",
    #     "params": {
    #         "uncertainty_measure": "entropy",
    #         "distance_metric": "euclidean",
    #         "novelty_type": "total",
    #         "uncertainty_weight": 0.5,
    #         "novelty_weight": 0.5,
    #     },
    # },
]


WEIGHTING_STRATEGIES = [
    # {
    #     "name": "uniform",
    #     "params": {
    #         #
    #     },
    # },
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
NUM_ITER = 100
# RANDOM_STATES = [42]
RANDOM_STATES = [42, 123, 1234]
EXPERIMENT_NAME = "test"
# OUTPUT_DIR_TEMPLATE = "paper/flow/tae-classifier/ex={experiment_name}/target={target}/source={source}/query={query_strategy}/weighting={weighting_strategy}/seed={seed}"
OUTPUT_DIR_TEMPLATE = "paper/group/tae-classifier/ex={experiment_name}/target={target}/source={source}/query={query_strategy}/weighting={weighting_strategy}/seed={seed}"


def init_components(seed: int, query_strategy_dict: dict, weighting_strategy_dict: dict):
    # query
    query_strategy_dict["params"]["random_state"] = seed
    query_strategy = QueryStrategyFactory.create_classifier_strategy(
        strategy_name=query_strategy_dict["name"],
        config=query_strategy_dict["params"],
    )

    # weighting
    weighting_strategy = WeightingStrategyFactory.create(
        strategy_name=weighting_strategy_dict["name"],
        config=weighting_strategy_dict["params"],
    )

    return query_strategy, weighting_strategy


def train_classifier(
    source_X: np.ndarray,
    source_y: np.ndarray,
    all_target_sample_ids: np.ndarray,
    all_target_X: np.ndarray,
    assigned_labels: dict[str, np.ndarray],
    seed: int,
    weighting: WeightingStrategy,
    iteration: int,
):
    # get source data
    source_features = source_X.copy()
    source_labels = source_y.copy()

    # get annotated target data
    mask = np.isin(all_target_sample_ids, list(assigned_labels.keys()))

    selected_sample_ids = all_target_sample_ids[mask]
    annotated_target_features = all_target_X[mask]
    annotated_target_labels = np.array([assigned_labels[sample_id] for sample_id in selected_sample_ids])

    # compute sample weights
    source_sample_weight, target_sample_weight = weighting.compute_weights(
        source_features=source_features, target_features=annotated_target_features, iteration=iteration
    )

    # stack with source data
    if len(annotated_target_features) > 0:
        combined_features = np.vstack([source_features, annotated_target_features])
        combined_labels = np.concatenate([source_labels, annotated_target_labels])
        source_weights = np.array([source_sample_weight] * len(source_features))
        target_weights = np.array([target_sample_weight] * len(annotated_target_features))
        combined_weights = np.concatenate([source_weights, target_weights])
    else:
        # no target samples yet
        combined_features = source_features
        combined_labels = source_labels
        source_weights = np.array([source_sample_weight] * len(source_features))
        target_weights = np.array([])
        combined_weights = source_weights

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


def save_cluster_labels(report_dir: str, cluster_labels: dict[int, np.ndarray]):
    file = f"{report_dir}/cluster_labels.csv"
    with open(file, "w") as f:
        for cluster_id, label in cluster_labels.items():
            f.write(f"{cluster_id},{label.item()}\n")


def save_experiment_config(
    report_dir,
    source_dataset_name: str,
    source_dataset_path: str,
    target_dataset_name: str,
    target_dataset_path: str,
    test_dataset_name: str,
    test_dataset_path: str,
    max_iterations: int,
    seed: int,
    query_strategy: dict,
    weighting_strategy: dict,
):
    file = f"{report_dir}/experiment_config.yaml"
    output = {
        "source_dataset_name": source_dataset_name,
        "source_dataset_path": source_dataset_path,
        "target_dataset_name": target_dataset_name,
        "target_dataset_path": target_dataset_path,
        "test_dataset_name": test_dataset_name,
        "test_dataset_path": test_dataset_path,
        "max_iterations": max_iterations,
        "seed": seed,
        "query_strategy_name": query_strategy["name"],
        "query_strategy_params": query_strategy["params"],
        "weighting_strategy_name": weighting_strategy["name"],
        "weighting_strategy_params": weighting_strategy["params"],
        "classifier_name": CLASSIFICATION_ALGORITHM,
        "classifier_params": CLASSIFICATION_PARAMS,
    }
    with open(file, "w") as f:
        yaml.safe_dump(output, f)


def log_iteration(
    report_dir: str,
    iteration: int,
    selected_sample_ids: list[str],
    selected_labels: dict[str, np.ndarray],
    weighting_details: WeightingDetails,
    scores: dict,
):
    eval_file = f"{report_dir}/evaluations.csv"
    iterations_file = f"{report_dir}/iterations.csv"

    # write iterations
    iterations_header = [
        "iteration",
        "selected_sample_ids",
        "selected_labels",
        "weighting_details",
    ]
    if not os.path.exists(iterations_file):
        with open(iterations_file, "w") as f:
            f.write(",".join(iterations_header) + "\n")

    writer = csv.DictWriter(open(iterations_file, "a"), fieldnames=iterations_header)

    selected_sample_ids = [str(sample_id) for sample_id in selected_sample_ids]
    t_selected_labels = {str(sample_id): hot_encoding_to_ints(label) for sample_id, label in selected_labels.items()}
    weighting_transformed = {
        "source_weight_per_sample": weighting_details.source_weight_per_sample,
        "target_weight_per_sample": weighting_details.target_weight_per_sample,
        "source_weight_total": weighting_details.source_weight_total,
        "target_weight_total": weighting_details.target_weight_total,
        "weight_total": weighting_details.weight_total,
    }

    row = {
        "iteration": iteration,
        "selected_sample_ids": json.dumps(selected_sample_ids),
        "selected_labels": json.dumps(t_selected_labels),
        "weighting_details": json.dumps(weighting_transformed),
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
    source_sample_ids: np.ndarray,
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_sample_ids: np.ndarray,
    target_X: np.ndarray,
    target_y_gt: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    query_strategy_dict: dict,
    weighting_strategy_dict: dict,
):
    # setup
    query_strategy, weighting_strategy = init_components(seed, query_strategy_dict, weighting_strategy_dict)
    sample_id2label = {sample_id: label for sample_id, label in zip(target_sample_ids, target_y_gt)}

    assigned_labels: dict[str, np.ndarray] = {}  # sample_id -> assigned label

    # evaluate classifier without any target data
    clf, weighting_details = train_classifier(
        source_X=source_X,
        source_y=source_y,
        all_target_sample_ids=target_sample_ids,
        all_target_X=target_X,
        assigned_labels=assigned_labels,
        seed=seed,
        iteration=-1,
        weighting=weighting_strategy,
    )
    scores = evaluate_classifier(clf, test_X, test_y)

    log_iteration(
        report_dir=report_dir,
        iteration=-1,
        selected_sample_ids=[],
        selected_labels=assigned_labels,
        scores=scores,
        weighting_details=weighting_details,
    )

    for iteration in tqdm(range(NUM_ITER), desc="Running AL loops", unit="loop"):
        # query cluster
        unlabeled_mask = np.isin(target_sample_ids, list(assigned_labels.keys()), invert=True)
        unlabeled_sample_ids = target_sample_ids[unlabeled_mask]
        unlabeled_X = target_X[unlabeled_mask]

        labeled_target_mask = np.isin(target_sample_ids, list(assigned_labels.keys()))
        labeled_target_sample_ids = target_sample_ids[labeled_target_mask]
        labeled_target_X = target_X[labeled_target_mask]

        labeled_sample_ids = np.concatenate([source_sample_ids, labeled_target_sample_ids])
        labeled_X = np.vstack([source_X, labeled_target_X])

        selected_idxs = query_strategy.select_samples(
            model=clf,
            unlabeled_features=unlabeled_X,
            n_samples=1,
            #
            labeled_features=labeled_X,
            unlabeled_sample_ids=unlabeled_sample_ids,
            labeled_sample_ids=labeled_sample_ids,
            #
            annotated_features=labeled_X,
            source_sample_ids=source_sample_ids,
            annotated_sample_ids=labeled_sample_ids,
        )
        selected_sample_ids = target_sample_ids[unlabeled_mask][selected_idxs]
        assert len(selected_sample_ids) == len(set(selected_sample_ids)) == 1, selected_sample_ids
        assert set(selected_sample_ids).isdisjoint(assigned_labels.keys())

        # annotate sample
        iter_labels = {
            sample_id: sample_id2label[sample_id] for sample_id in selected_sample_ids if sample_id in sample_id2label
        }

        assigned_labels.update(iter_labels)

        # train classifier
        clf, weighting_details = train_classifier(
            source_X=source_X,
            source_y=source_y,
            all_target_sample_ids=target_sample_ids,
            all_target_X=target_X,
            assigned_labels=assigned_labels,
            seed=seed,
            weighting=weighting_strategy,
            iteration=iteration,
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
            weighting_details=weighting_details,
        )


def main():
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

            ## add prefix
            source_sample_ids = [f"source_{sample_id}" for sample_id in source_sample_ids]
            target_sample_ids = [f"target_{sample_id}" for sample_id in target_sample_ids]

            ## preprocess features
            preprocessor = FitPreprocessor(config=PREPROCESSING)
            preprocessor.fit(source_X)

            source_X = preprocessor.transform(source_X)
            target_X = preprocessor.transform(target_X)
            test_X = preprocessor.transform(test_X)

            for qs in tqdm(QUERY_STRATEGIES, desc="Query Strategies", unit="strategy"):
                for ws in tqdm(WEIGHTING_STRATEGIES, desc="Weighting Strategies", unit="strategy"):

                    report_dir = OUTPUT_DIR_TEMPLATE.format(
                        experiment_name=EXPERIMENT_NAME,
                        source=source_dataset.split.dataset_name,
                        target=target_dataset.split.dataset_name,
                        query_strategy=qs["name"],
                        weighting_strategy=ws["name"],
                        seed=seed,
                    )
                    os.makedirs(report_dir, exist_ok=True)

                    save_experiment_config(
                        report_dir=report_dir,
                        max_iterations=NUM_ITER,
                        source_dataset_name=source_dataset.split.dataset_name,
                        source_dataset_path=dataset_combination["source"],
                        target_dataset_name=target_dataset.split.dataset_name,
                        target_dataset_path=dataset_combination["target"],
                        test_dataset_name=target_dataset.split.dataset_name,
                        test_dataset_path=dataset_combination["test"],
                        seed=seed,
                        query_strategy=qs,
                        weighting_strategy=ws,
                    )

                    run_experiment(
                        report_dir=report_dir,
                        source_sample_ids=np.array(source_sample_ids).copy(),
                        source_X=source_X.copy(),
                        source_y=source_y.copy(),
                        target_sample_ids=np.array(target_sample_ids).copy(),
                        target_X=target_X.copy(),
                        target_y_gt=target_y.copy(),
                        test_X=test_X.copy(),
                        test_y=test_y.copy(),
                        seed=seed,
                        query_strategy_dict=qs.copy(),
                        weighting_strategy_dict=ws.copy(),
                    )


def get_combinations():
    combinations = []
    if ONLY_COMBINATIONS and len(ONLY_COMBINATIONS) > 0:
        for combination in ONLY_COMBINATIONS:

            if "target" not in combination:
                source = combination["source"]
                for target in DATASETS:
                    if source != target:
                        combinations.append(
                            {
                                "source": TRAIN_DATASET_TEMPLATE.format(dataset=source),
                                "target": TRAIN_DATASET_TEMPLATE.format(dataset=target),
                                "test": TEST_DATASET_TEMPLATE.format(dataset=target),
                            }
                        )

            else:
                combinations.append(
                    {
                        "source": TRAIN_DATASET_TEMPLATE.format(dataset=combination["source"]),
                        "target": TRAIN_DATASET_TEMPLATE.format(dataset=combination["target"]),
                        "test": TEST_DATASET_TEMPLATE.format(dataset=combination["target"]),
                    }
                )

        return combinations

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
