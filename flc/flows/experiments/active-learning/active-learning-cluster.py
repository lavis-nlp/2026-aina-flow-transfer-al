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
from flc.shared.clustering import ClusteringFactory
from flc.shared.transfer_active_learning.query_strategies import QueryStrategyFactory
from flc.shared.transfer_active_learning.label_selection import LabelSelectionFactory, LabelSelectionStrategy

# --- DATA ---
DATASETS = [
    "cic-ids-2012",
    "cic-ids-2017",
    "iscx-vpn-2016-non-vpn",
    "iscx-tor-2016-non-tor",
]

DATASET_COMBINATIONS = [
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
    ## TEST
    # {
    #     "train": "data/datasets/cic-ids-2012_splits/train",
    #     "test": "data/datasets/cic-ids-2012_splits/test",
    # },
    # {
    #     "train": "data/datasets/cic-ids-2017_splits/train",
    #     "test": "data/datasets/cic-ids-2017_splits/test",
    # },
    # {
    #     "train": "data/datasets/iscx-vpn-2016-non-vpn_splits/train",
    #     "test": "data/datasets/iscx-vpn-2016-non-vpn_splits/test",
    # },
    # {
    #     "train": "data/datasets/iscx-tor-2016-non-tor_splits/train",
    #     "test": "data/datasets/iscx-tor-2016-non-tor_splits/test",
    # },
    ## GROUP
    {
        "train": "data/group/cic-ids-2012-group_splits/train",
        "test": "data/group/cic-ids-2012-group_splits/test",
    },
    {
        "train": "data/group/cic-ids-2017-group_splits/train",
        "test": "data/group/cic-ids-2017-group_splits/test",
    },
    # {
    #     "train": "data/group/iscx-vpn-2016-non-vpn-group_splits/train",
    #     "test": "data/group/iscx-vpn-2016-non-vpn-group_splits/test",
    # },
    # {
    #     "train": "data/group/iscx-tor-2016-non-tor-group_splits/train",
    #     "test": "data/group/iscx-tor-2016-non-tor-group_splits/test",
    # },
]

PREPROCESSING = PreprocessingConfig(
    enabled=True,
    scaler_type="robust",
    clip_quantiles=(0.01, 0.99),
    log_transform=True,
)

# --- CLUSTERING ---
CLUSTERING_ALGORITHM = "dbscan"
CLUSTERING_PARAMS = {
    # "eps": 0.2,
    "eps": 1.0,
    "min_samples": 2,
    "metric": "euclidean",
    "algorithm": "auto",
    "transform_noise": False,
}

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
        "name": "random_cluster",
        "params": {
            #
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
    {
        "name": "most_diverse_cluster",
        "params": {
            "cluster_selection_method": "most_diverse_first",
            "diversity_metric": "variance",
            "min_samples_for_diversity": 2,
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
    {
        "name": "highest_uncertainty_cluster",
        "params": {
            "uncertainty_measure": "entropy",
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
    {
        "name": "biggest_cluster",
        "params": {
            "cluster_selection_method": "largest_first",
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
    {
        "name": "total_novelty_by_medoid_cluster",
        "params": {
            "distance_metric": "euclidean",
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
    {
        "name": "uncertainty_novelty_by_medoid_cluster",
        "params": {
            "uncertainty_measure": "entropy",
            "distance_metric": "euclidean",
            "novelty_type": "total",
            "uncertainty_weight": 0.5,
            "novelty_weight": 0.5,
            "exclude_noise": True,
            "min_cluster_size": 1,
        },
    },
]

LABEL_SELECTION_STRATEGIES = [
    # {
    #     "name": "random",
    #     "params": {
    #         #
    #     },
    # },
    {
        "name": "medoid",
        "params": {
            "distance_metric": "manhattan",
        },
    },
]


# --- MISC ---
NUM_INITIAL = 5
NUM_ITER = 100
MAX_FLOWS = 50000
RANDOM_STATES = [42, 123, 1234]
# RANDOM_STATES = [42]
EXPERIMENT_NAME = "test-transformed_noise=false,exclude_noise=true"

# OUTPUT_DIR_TEMPLATE = "results/active_learning/cluster/{experiment_name}/dataset={dataset}/query={query_strategy}/label={label_selection}/seed={seed}"
# OUTPUT_DIR_TEMPLATE = "paper/flow/al-cluster/ex={experiment_name}/dataset={dataset}/query={query_strategy}/label={label_selection}/seed={seed}"
OUTPUT_DIR_TEMPLATE = "paper/group/al-cluster/ex={experiment_name}/dataset={dataset}/query={query_strategy}/label={label_selection}/seed={seed}"


def init_components(seed: int, query_strategy_dict: dict, label_selection_strategy_dict: dict):
    # query
    query_strategy_dict["params"]["random_state"] = seed
    query_strategy = QueryStrategyFactory.create_cluster_strategy(
        strategy_name=query_strategy_dict["name"],
        config=query_strategy_dict["params"],
    )

    # label selection
    label_selection_strategy_dict["params"]["random_state"] = seed
    label_selection_strategy = LabelSelectionFactory.create(
        strategy_name=label_selection_strategy_dict["name"],
        config=label_selection_strategy_dict["params"],
    )

    return query_strategy, label_selection_strategy


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


def annotate_samples(
    all_sample_ids: np.ndarray,
    all_X: np.ndarray,
    all_y_gt: np.ndarray,
    selected_sample_ids: np.ndarray,
    label_selection: LabelSelectionStrategy,
) -> dict[int, np.ndarray]:

    mask = np.isin(all_sample_ids, selected_sample_ids)
    ids_selected = all_sample_ids[mask]
    X_selected = all_X[mask]
    y_selected = all_y_gt[mask]

    selected_labels = label_selection.select(sample_features=X_selected, ground_truth_labels=y_selected)

    assigned_labels = {sample_id: label for sample_id, label in zip(ids_selected, selected_labels)}

    return assigned_labels


def save_cluster_labels(report_dir: str, cluster_labels: dict[int, np.ndarray]):
    file = f"{report_dir}/cluster_labels.csv"
    with open(file, "w") as f:
        for cluster_id, label in cluster_labels.items():
            f.write(f"{cluster_id},{label.item()}\n")


def save_experiment_config(
    report_dir,
    target_dataset_name: str,
    target_dataset_path: str,
    test_dataset_name: str,
    test_dataset_path: str,
    max_iterations: int,
    seed: int,
    query_strategy: dict,
    label_selection: dict,
    num_initial_clusters: int = NUM_INITIAL,
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
        "label_selection_name": label_selection["name"],
        "label_selection_params": label_selection["params"],
        "num_initial_clusters": num_initial_clusters,
        "classifier_name": CLASSIFICATION_ALGORITHM,
        "classifier_params": CLASSIFICATION_PARAMS,
        "clustering_algorithm": CLUSTERING_ALGORITHM,
        "clustering_params": CLUSTERING_PARAMS,
    }
    with open(file, "w") as f:
        yaml.safe_dump(output, f)


def log_iteration(
    report_dir: str,
    iteration: int,
    selected_cluster_ids: list[int],
    selected_sample_ids: list[int],
    selected_labels: dict[int, np.ndarray],
    scores: dict,
):
    eval_file = f"{report_dir}/evaluations.csv"
    iterations_file = f"{report_dir}/iterations.csv"

    # write iterations
    iterations_header = ["iteration", "selected_cluster_ids", "selected_sample_ids", "selected_labels"]
    if not os.path.exists(iterations_file):
        with open(iterations_file, "w") as f:
            f.write(",".join(iterations_header) + "\n")

    writer = csv.DictWriter(open(iterations_file, "a"), fieldnames=iterations_header)

    selected_cluster_ids = [str(cluster_id) for cluster_id in selected_cluster_ids]
    selected_sample_ids = [str(sample_id) for sample_id in selected_sample_ids]
    t_selected_labels = {int(sample_id): hot_encoding_to_ints(label) for sample_id, label in selected_labels.items()}

    row = {
        "iteration": iteration,
        "selected_cluster_ids": json.dumps(selected_cluster_ids),
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
    cluster2ids: dict[int, list[int]],
    id2cluster: dict[int, int],
    seed: int,
    query_strategy_dict: dict,
    label_selection_strategy_dict: dict,
):
    # setup
    sample_ids = np.array(target_sample_ids)
    cluster_ids = list(cluster2ids.keys())
    query_strategy, label_selection_strategy = init_components(seed, query_strategy_dict, label_selection_strategy_dict)

    sample_ids2features = {sample_id: features for sample_id, features in zip(target_sample_ids, target_X)}

    labeled_cluster_ids = list()
    assigned_labels: dict[int, np.ndarray] = {}  # sample_id -> assigned label

    # intial labeling - select random clusters
    initial_cluster_ids = random.Random(seed).sample(cluster_ids, NUM_INITIAL)
    labeled_cluster_ids.extend(initial_cluster_ids)

    for cluster_id in initial_cluster_ids:
        labels = annotate_samples(
            all_sample_ids=sample_ids,
            all_X=target_X,
            all_y_gt=target_y_gt,
            selected_sample_ids=cluster2ids[cluster_id],
            label_selection=label_selection_strategy,
        )
        assigned_labels.update(labels)

    clf = train_classifier(
        all_sample_ids=sample_ids,
        all_X=target_X,
        assigned_labels=assigned_labels,
        seed=seed,
    )
    scores = evaluate_classifier(clf, test_X, test_y)

    initial_sample_ids = [sample_id for cluster_id in initial_cluster_ids for sample_id in cluster2ids[cluster_id]]
    initial_assigned_labels = {sample_id: assigned_labels[sample_id] for sample_id in initial_sample_ids}
    log_iteration(
        report_dir=report_dir,
        iteration=-1,
        selected_cluster_ids=initial_cluster_ids,
        selected_sample_ids=initial_sample_ids,
        selected_labels=initial_assigned_labels,
        scores=scores,
    )

    for iteration in tqdm(range(NUM_ITER), desc="Running AL loops", unit="loop"):
        # query cluster
        unlabeled_mask = np.isin(sample_ids, list(assigned_labels.keys()), invert=True)
        unlabeled_sample_ids = sample_ids[unlabeled_mask]
        unlabeled_X = target_X[unlabeled_mask]
        unlabeled_clusters = np.array([id2cluster[sample_id] for sample_id in unlabeled_sample_ids])

        iter_cluster_id = query_strategy.select_cluster(
            unlabeled_sample_ids=unlabeled_sample_ids,
            unlabeled_sample_features=unlabeled_X,
            unlabeled_sample_clusters=unlabeled_clusters,
            classifier=clf,
            cluster2source_ids=dict(),
            cluster2target_ids=cluster2ids,
            source_id2features=dict(),
            target_id2features=sample_ids2features,
            labeled_target_ids=set(assigned_labels.keys()),
        )
        assert iter_cluster_id not in labeled_cluster_ids
        assert iter_cluster_id != -1

        # annotate cluster samples
        selected_sample_ids = cluster2ids[iter_cluster_id]
        iter_labels = annotate_samples(
            all_sample_ids=sample_ids,
            all_X=target_X,
            all_y_gt=target_y_gt,
            selected_sample_ids=selected_sample_ids,
            label_selection=label_selection_strategy,
        )

        labeled_cluster_ids.append(iter_cluster_id)
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
            selected_cluster_ids=[iter_cluster_id],
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

            # perform clustering
            clusterer = ClusteringFactory.create_with_defaults(
                algorithm_name=CLUSTERING_ALGORITHM,
                **CLUSTERING_PARAMS,
            )
            cluster_labels = clusterer.fit_predict(target_X)
            cluster_labels_dict = {sample_id: label for sample_id, label in zip(target_sample_ids, cluster_labels)}

            id2cluster = {sample_id: cluster for sample_id, cluster in zip(target_sample_ids, cluster_labels)}
            cluster2ids = {}
            for sample_id, cluster in zip(target_sample_ids, cluster_labels):
                if cluster not in cluster2ids:
                    cluster2ids[cluster] = []
                cluster2ids[cluster].append(sample_id)

            for qs in tqdm(QUERY_STRATEGIES, desc="Query Strategies", unit="strategy"):
                for ls in tqdm(LABEL_SELECTION_STRATEGIES, desc="Label Selection Strategies", unit="strategy"):

                    report_dir = OUTPUT_DIR_TEMPLATE.format(
                        dataset=target_dataset.split.dataset_name,
                        experiment_name=EXPERIMENT_NAME,
                        query_strategy=qs["name"],
                        label_selection=ls["name"],
                        seed=seed,
                    )
                    os.makedirs(report_dir, exist_ok=True)

                    save_cluster_labels(
                        report_dir=report_dir,
                        cluster_labels=cluster_labels_dict,
                    )
                    save_experiment_config(
                        report_dir=report_dir,
                        max_iterations=NUM_ITER,
                        target_dataset_name=target_dataset.split.dataset_name,
                        target_dataset_path=dataset_combination["train"],
                        test_dataset_name=target_dataset.split.dataset_name,
                        test_dataset_path=dataset_combination["test"],
                        seed=seed,
                        query_strategy=qs,
                        label_selection=ls,
                    )

                    run_experiment(
                        report_dir=report_dir,
                        target_sample_ids=target_sample_ids,
                        target_X=target_X,
                        target_y_gt=target_y,
                        test_X=test_X,
                        test_y=test_y,
                        cluster2ids=cluster2ids,
                        id2cluster=id2cluster,
                        seed=seed,
                        query_strategy_dict=qs,
                        label_selection_strategy_dict=ls,
                    )


if __name__ == "__main__":
    main()
