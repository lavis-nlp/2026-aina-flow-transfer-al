import json
import os
import csv

import numpy as np
import yaml
from tqdm import tqdm
import click

from flc.shared.utils.numpy_helpers import hot_encoding_to_ints
from flc.flows.dataset import FlowClassificationDataset
from flc.shared.preprocessing import PreprocessingConfig, FitPreprocessor
from flc.shared.classification import ClassificationFactory, ClassificationEvaluator
from flc.shared.clustering import ClusteringFactory
from flc.shared.transfer_active_learning.query_strategies import QueryStrategyFactory
from flc.shared.transfer_active_learning.label_selection import LabelSelectionFactory, LabelSelectionStrategy
from flc.shared.transfer_active_learning.weighting_strategies.factory import WeightingStrategyFactory, WeightingStrategy
from flc.flows.experiments.transfer_active_learning.shared import WeightingDetails

# --- DATA ---
DATASETS = [
    "cic-ids-2012",
    "cic-ids-2017",
    "iscx-vpn-2016-non-vpn",
    "iscx-tor-2016-non-tor",
]

# ONLY_COMBINATIONS = [
#     # {
#     #     "source": "cic-ids-2012",
#     # },
#     # {
#     #     "source": "cic-ids-2017",
#     # },
#     # {
#     #     "source": "iscx-vpn-2016-non-vpn",
#     # },
#     {
#         "source": "iscx-tor-2016-non-tor",
#     },
# ]

# TRAIN_DATASET_TEMPLATE = "data/datasets/{dataset}_splits/train"
# TEST_DATASET_TEMPLATE = "data/datasets/{dataset}_splits/test"

# TRAIN_DATASET_TEMPLATE = "data/group/{dataset}-group_splits/train"
# TEST_DATASET_TEMPLATE = "data/group/{dataset}-group_splits/test"

PREPROCESSING = PreprocessingConfig(
    enabled=True,
    scaler_type="robust",
    clip_quantiles=(0.01, 0.99),
    log_transform=True,
)

# --- CLUSTERING ---
# CLUSTERING_ALGORITHM = "dbscan"
# CLUSTERING_PARAMS = {
#     # "eps": 0.2,
#     "eps": 1.0,
#     "min_samples": 2,
#     "metric": "euclidean",
#     "algorithm": "auto",
#     "transform_noise": False,
# }

# --- CLASSIFICATION ---
CLASSIFICATION_ALGORITHM = "random_forest"
CLASSIFICATION_PARAMS = {
    "n_estimators": 25,
    "max_depth": 10,
}

# --- ACTIVE LEARNING ---
QUERY_STRATEGIES = [
    {
        "name": "total_novelty_by_medoid_cluster",
        "params": {
            "distance_metric": "euclidean",
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
            "distance_metric": "euclidean",
        },
    },
]

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
NUM_ITER = 100
# MAX_FLOWS = 50000
# RANDOM_STATES = [42]
# RANDOM_STATES = [42, 123, 1234]
# OUTPUT_DIR_TEMPLATE = "results/transfer-active-learning/cluster/test/noise-transformed/target={target}/source={source}/query={query_strategy}/label={label_selection}/weighting={weighting_strategy}/seed={seed}"
# EXPERIMENT_NAME = "test-noise_transformed=false-exclude_noise=true"
OUTPUT_DIR_TEMPLATE = "paper/{label_type}/tae-cluster-w-source/ex={experiment_name}/target={target}/source={source}/query={query_strategy}/label={label_selection}/weighting={weighting_strategy}/seed={seed}"


def perform_clustering(sample_ids: np.ndarray, features: np.ndarray, clustering_algo: str, clustering_params: dict):
    """
    Perform clustering on the given features and return a mapping of sample IDs to cluster labels.
    """
    clusterer = ClusteringFactory.create_with_defaults(
        algorithm_name=clustering_algo,
        **clustering_params,
    )
    cluster_labels = clusterer.fit_predict(features)

    # create mappings
    id2cluster = {sample_id: label for sample_id, label in zip(sample_ids, cluster_labels)}
    cluster2ids = {}
    for sample_id, cluster in id2cluster.items():
        if cluster not in cluster2ids:
            cluster2ids[cluster] = []
        cluster2ids[cluster].append(sample_id)

    return id2cluster, cluster2ids


def init_components(
    seed: int, query_strategy_dict: dict, label_selection_strategy_dict: dict, weighting_strategy_dict: dict
):
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

    # weighting
    weighting_strategy = WeightingStrategyFactory.create(
        strategy_name=weighting_strategy_dict["name"],
        config=weighting_strategy_dict["params"],
    )

    return query_strategy, label_selection_strategy, weighting_strategy


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


def annotate_samples(
    all_target_sample_ids: np.ndarray,
    all_target_X: np.ndarray,
    all_target_y_gt: np.ndarray,
    selected_sample_ids: list[str],
    label_selection: LabelSelectionStrategy,
) -> dict[str, np.ndarray]:

    mask = np.isin(all_target_sample_ids, selected_sample_ids)
    ids_selected = all_target_sample_ids[mask]
    X_selected = all_target_X[mask]
    y_selected = all_target_y_gt[mask]

    selected_labels = label_selection.select(sample_features=X_selected, ground_truth_labels=y_selected)

    assigned_labels = {sample_id: label for sample_id, label in zip(ids_selected, selected_labels)}

    return assigned_labels


def save_cluster_labels(report_dir: str, name: str, cluster_labels: dict[str, np.ndarray]):
    file = f"{report_dir}/{name}_cluster_labels.csv"
    with open(file, "w") as f:
        f.write("sample_id,label\n")
        for sample_id, label in cluster_labels.items():
            f.write(f"{sample_id},{label.item()}\n")


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
    label_selection: dict,
    weighting_strategy: dict,
    clustering_algo: str,
    clustering_params: dict,
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
        "label_selection_name": label_selection["name"],
        "label_selection_params": label_selection["params"],
        "weighting_strategy_name": weighting_strategy["name"],
        "weighting_strategy_params": weighting_strategy["params"],
        "classifier_name": CLASSIFICATION_ALGORITHM,
        "classifier_params": CLASSIFICATION_PARAMS,
        "clustering_algorithm": clustering_algo,
        "clustering_params": clustering_params,
    }
    with open(file, "w") as f:
        yaml.safe_dump(output, f)


def log_iteration(
    report_dir: str,
    iteration: int,
    selected_cluster_ids: list[int],
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
        "selected_cluster_ids",
        "selected_sample_ids",
        "selected_labels",
        "weighting_details",
    ]
    if not os.path.exists(iterations_file):
        with open(iterations_file, "w") as f:
            f.write(",".join(iterations_header) + "\n")

    writer = csv.DictWriter(open(iterations_file, "a"), fieldnames=iterations_header)

    selected_cluster_ids = [str(cluster_id) for cluster_id in selected_cluster_ids]
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
        "selected_cluster_ids": json.dumps(selected_cluster_ids),
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
    cluster2source_ids: dict[int, list[str]],
    source_id2cluster: dict[str, int],
    cluster2target_ids: dict[int, list[str]],
    target_id2cluster: dict[str, int],
    seed: int,
    query_strategy_dict: dict,
    label_selection_strategy_dict: dict,
    weighting_strategy_dict: dict,
):
    # setup
    query_strategy, label_selection_strategy, weighting_strategy = init_components(
        seed, query_strategy_dict, label_selection_strategy_dict, weighting_strategy_dict
    )

    source_ids2features = {sample_id: features for sample_id, features in zip(source_sample_ids, source_X)}
    target_ids2features = {sample_id: features for sample_id, features in zip(target_sample_ids, target_X)}

    labeled_cluster_ids = list()
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
        selected_cluster_ids=[],
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
        unlabeled_clusters = np.array([target_id2cluster[sample_id] for sample_id in unlabeled_sample_ids])

        iter_cluster_id = query_strategy.select_cluster(
            unlabeled_sample_ids=unlabeled_sample_ids,
            unlabeled_sample_features=unlabeled_X,
            unlabeled_sample_clusters=unlabeled_clusters,
            classifier=clf,
            cluster2source_ids=cluster2source_ids,
            cluster2target_ids=cluster2target_ids,
            source_id2features=source_ids2features,
            target_id2features=target_ids2features,
            labeled_target_ids=set(assigned_labels.keys()),
        )
        assert iter_cluster_id not in labeled_cluster_ids
        assert iter_cluster_id != -1

        # annotate cluster samples
        selected_sample_ids = cluster2target_ids[iter_cluster_id]
        iter_labels = annotate_samples(
            all_target_sample_ids=target_sample_ids,
            all_target_X=target_X,
            all_target_y_gt=target_y_gt,
            selected_sample_ids=selected_sample_ids,
            label_selection=label_selection_strategy,
        )

        labeled_cluster_ids.append(iter_cluster_id)
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
            selected_cluster_ids=[iter_cluster_id],
            selected_sample_ids=selected_sample_ids,
            selected_labels=iter_labels,
            scores=scores,
            weighting_details=weighting_details,
        )


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to the configuration file.",
)
def main(config: str):
    config_data: dict = yaml.safe_load(open(config, "r"))

    label_type = config_data["label_type"]
    random_states = config_data["random_states"]
    max_flows = config_data["max_flows"]
    clustering_algorithm = config_data["clustering_algorithm"]
    clustering_params = config_data["clustering_params"]
    query_params = config_data["query_params"]
    experiment_name = config_data["experiment_name"]

    train_dataset_template, test_dataset_template = get_dataset_templates(label_type)
    dataset_combinations = get_combinations(
        config_data["combinations"],
        train_dataset_template=train_dataset_template,
        test_dataset_template=test_dataset_template,
    )

    for dataset_combination in tqdm(dataset_combinations, desc="Dataset Combinations", unit="combination"):
        for seed in tqdm(random_states, desc="Random States", unit="state"):
            # load datasets
            source_dataset = (
                FlowClassificationDataset(dataset_combination["source"])
                .subsample(max_flows=max_flows, random_seed=seed)
                .shuffle(random_state=seed)
            )
            target_dataset = (
                FlowClassificationDataset(dataset_combination["target"])
                .subsample(max_flows=max_flows, random_seed=seed)
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
            source_sample_ids = np.array([f"source_{sample_id}" for sample_id in source_sample_ids])
            target_sample_ids = np.array([f"target_{sample_id}" for sample_id in target_sample_ids])

            ## preprocess features
            preprocessor = FitPreprocessor(config=PREPROCESSING)
            preprocessor.fit(source_X)

            source_X = preprocessor.transform(source_X)
            target_X = preprocessor.transform(target_X)
            test_X = preprocessor.transform(test_X)

            # perform clustering for source and target separately
            source_id2cluster, cluster2source_ids = perform_clustering(
                sample_ids=source_sample_ids,
                features=source_X,
                clustering_algo=clustering_algorithm,
                clustering_params=clustering_params,
            )
            target_id2cluster, cluster2target_ids = perform_clustering(
                sample_ids=target_sample_ids,
                features=target_X,
                clustering_algo=clustering_algorithm,
                clustering_params=clustering_params,
            )

            for qs in tqdm(QUERY_STRATEGIES, desc="Query Strategies", unit="strategy"):
                for ls in tqdm(LABEL_SELECTION_STRATEGIES, desc="Label Selection Strategies", unit="strategy"):
                    for ws in tqdm(WEIGHTING_STRATEGIES, desc="Weighting Strategies", unit="strategy"):

                        # set query strategy parameters
                        query_strat_dict = qs.copy()
                        query_strat_dict["params"].update(query_params)

                        report_dir = OUTPUT_DIR_TEMPLATE.format(
                            label_type=label_type,
                            experiment_name=experiment_name,
                            source=source_dataset.split.dataset_name,
                            target=target_dataset.split.dataset_name,
                            query_strategy=query_strat_dict["name"],
                            label_selection=ls["name"],
                            weighting_strategy=ws["name"],
                            seed=seed,
                        )
                        os.makedirs(report_dir, exist_ok=True)

                        save_cluster_labels(
                            report_dir=report_dir,
                            name="source",
                            cluster_labels=source_id2cluster,
                        )
                        save_cluster_labels(
                            report_dir=report_dir,
                            name="target",
                            cluster_labels=target_id2cluster,
                        )

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
                            query_strategy=query_strat_dict,
                            label_selection=ls,
                            weighting_strategy=ws,
                            clustering_algo=clustering_algorithm,
                            clustering_params=clustering_params,
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
                            cluster2source_ids=cluster2source_ids.copy(),
                            source_id2cluster=source_id2cluster.copy(),
                            cluster2target_ids=cluster2target_ids.copy(),
                            target_id2cluster=target_id2cluster.copy(),
                            seed=seed,
                            query_strategy_dict=query_strat_dict,
                            label_selection_strategy_dict=ls.copy(),
                            weighting_strategy_dict=ws.copy(),
                        )


def get_combinations(combinations_data: dict, train_dataset_template: str, test_dataset_template: str):
    combinations = []
    for combination in combinations_data:
        if "source" in combination and "target" not in combination:
            # create combinations with all other dataset which have the specified source
            for target_dataset in DATASETS:
                if combination["source"] != target_dataset:
                    combinations.append(
                        {
                            "source": train_dataset_template.format(dataset=combination["source"]),
                            "target": train_dataset_template.format(dataset=target_dataset),
                            "test": test_dataset_template.format(dataset=target_dataset),
                        }
                    )
        elif "target" in combination and "source" not in combination:
            # create combinations with all other dataset which have the specified target
            for source_dataset in DATASETS:
                if combination["target"] != source_dataset:
                    combinations.append(
                        {
                            "source": train_dataset_template.format(dataset=source_dataset),
                            "target": train_dataset_template.format(dataset=combination["target"]),
                            "test": test_dataset_template.format(dataset=combination["target"]),
                        }
                    )

        elif "source" in combination and "target" in combination:
            combinations.append(
                {
                    "source": train_dataset_template.format(dataset=combination["source"]),
                    "target": train_dataset_template.format(dataset=combination["target"]),
                    "test": test_dataset_template.format(dataset=combination["target"]),
                }
            )

        else:
            raise ValueError("Combination must contain either 'source' or 'target' or both.")

        return combinations


def get_dataset_templates(label_type):
    if label_type == "flow":
        train_dataset_template = "data/datasets/{dataset}_splits/train"
        test_dataset_template = "data/datasets/{dataset}_splits/test"
    elif label_type == "group":
        train_dataset_template = "data/group/{dataset}-group_splits/train"
        test_dataset_template = "data/group/{dataset}-group_splits/test"
    else:
        raise ValueError(f"Unknown label type: {label_type}")

    return train_dataset_template, test_dataset_template


if __name__ == "__main__":
    main()
