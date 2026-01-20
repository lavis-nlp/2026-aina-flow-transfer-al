import os

import json
from tqdm import tqdm
import numpy as np
import pandas as pd

from flc.shared.utils.numpy_helpers import convert_to_primitives_nested
from flc.shared.preprocessing import PreprocessingConfig
from flc.shared.clustering import ClusteringFactory, ClusteringEvaluator
from flc.flows.dataset import FlowClassificationDataset
from flc.shared.transfer_active_learning.label_selection import LabelSelectionFactory, LabelSelectionStrategy

# DATASETS = [
#     "data/datasets/iscx-tor-2016-non-tor_splits/train",
#     "data/datasets/iscx-vpn-2016-non-vpn_splits/train",
#     "data/datasets/cic-ids-2012_splits/train",
#     "data/datasets/cic-ids-2017_splits/train",
#     # "data/datasets/iscx-vpn-2016-vpn_splits/train",
#     # "data/datasets/iscx-tor-2016-tor_splits/train",
# ]
DATASETS = [
    "data/group/cic-ids-2012-group_splits/train",
    "data/group/cic-ids-2017-group_splits/train",
    "data/group/iscx-vpn-2016-non-vpn-group_splits/train",
    "data/group/iscx-tor-2016-non-tor-group_splits/train",
]

# OUTPUT_DIR_TEMPLATE = "paper/flow/dbscan_eps/dataset={dataset_name}/eps={eps}/distance={distance_metric}/seed={seed}"
OUTPUT_DIR_TEMPLATE = "paper/group/dbscan_eps/dataset={dataset_name}/eps={eps}/distance={distance_metric}/seed={seed}"

EPS_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]

PREPROCESSING_CONFIG = PreprocessingConfig(
    enabled=True,
    scaler_type="robust",
    clip_quantiles=(0.01, 0.99),
    log_transform=True,
)

MAX_FLOWS = 50000
SEEDS = [42, 123, 1234]


def select_labels(sample_ids, X, cluster_labels, true_labels, label_selection: LabelSelectionStrategy):
    cluster_ids = np.unique(cluster_labels[cluster_labels != -1])

    noise_mask = cluster_labels == -1
    num_noise = noise_mask.sum()

    selected_labels = dict()

    for cluster_id in cluster_ids:
        cluster_mask = cluster_labels == cluster_id

        cluster_sample_ids = sample_ids[cluster_mask]
        cluster_X = X[cluster_mask]
        cluster_true_labels = true_labels[cluster_mask]

        labels = label_selection.select(sample_features=cluster_X, ground_truth_labels=cluster_true_labels)
        selected_labels.update({sample_id: label for sample_id, label in zip(cluster_sample_ids, labels)})

    assert len(selected_labels) == (len(sample_ids) - num_noise)
    return selected_labels


def compute_stats(sample_ids, X, cluster_labels, true_labels, distance_metric, seed=42):
    num_noise = (cluster_labels == -1).sum()
    num_clustered = (cluster_labels != -1).sum()
    num_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

    selected_labels_random = select_labels(
        sample_ids=sample_ids,
        X=X,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        label_selection=LabelSelectionFactory.create("random", {"random_state": seed}),
    )
    selected_labels_medoid = select_labels(
        sample_ids=sample_ids,
        X=X,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        label_selection=LabelSelectionFactory.create(
            "medoid", {"random_state": seed, "distance_metric": distance_metric}
        ),
    )

    return {
        "distance_metric": distance_metric,
        "num_noise": num_noise,
        "num_clustered": num_clustered,
        "num_clusters": num_clusters,
        "selected_labels_random": selected_labels_random,
        "selected_labels_medoid": selected_labels_medoid,
    }


def run_sweep(dataset_path):
    for seed in SEEDS:
        dataset = (
            FlowClassificationDataset(dataset_path)
            .subsample(max_flows=MAX_FLOWS, random_seed=seed)
            .shuffle(random_state=seed)
        )
        dataset_name = dataset.split.dataset_name

        dataset.set_preprocessor(config=PREPROCESSING_CONFIG)
        sample_ids = np.array(dataset.get_flow_ids())
        X, y = dataset.to_sklearn_format(preprocessed=True)

        for eps in tqdm(EPS_SWEEP, desc=f"Running DBSCAN EPS Sweep for {dataset_name}"):
            for distance_metric in ["euclidean", "manhattan"]:
                # create clustering model
                config = {
                    "eps": eps,
                    "min_samples": 2,
                    "metric": distance_metric,
                    "algorithm": "auto",
                }
                clusterer = ClusteringFactory.create_with_defaults("dbscan", **config)

                cluster_labels = clusterer.fit_predict(X)

                details = compute_stats(
                    sample_ids=sample_ids,
                    X=X,
                    cluster_labels=cluster_labels,
                    true_labels=y,
                    distance_metric=distance_metric,
                    seed=seed,
                )

                output_dir = OUTPUT_DIR_TEMPLATE.format(
                    dataset_name=dataset_name, eps=eps, seed=seed, distance_metric=distance_metric
                )
                os.makedirs(output_dir, exist_ok=True)

                # save cluster labels
                cluster_labels_df = pd.DataFrame(
                    {
                        "sample_id": dataset.get_flow_ids(),
                        "cluster_label": cluster_labels,
                    }
                )
                cluster_labels_file = f"{output_dir}/cluster_labels.csv"
                cluster_labels_df.to_csv(cluster_labels_file, index=False)

                # save assigned labels (random and medoid)
                for selection_type in ["random", "medoid"]:
                    selected_labels = details[f"selected_labels_{selection_type}"]
                    # convert values (labels multihot) to list
                    selected_labels = {k: v.tolist() for k, v in selected_labels.items()}

                    selected_labels_df = pd.DataFrame(selected_labels.items(), columns=["sample_id", "label"])

                    selected_labels_file = f"{output_dir}/selected_labels_{selection_type}.json"
                    selected_labels_df.to_csv(selected_labels_file, index=False)

                # save stats
                stats_file = f"{output_dir}/stats.json"
                stats = {
                    "seed": seed,
                    "eps": eps,
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "distance_metric": details["distance_metric"],
                    "num_noise": details["num_noise"],
                    "num_clustered": details["num_clustered"],
                    "num_clusters": details["num_clusters"],
                }
                with open(stats_file, "w") as f:
                    json.dump(convert_to_primitives_nested(stats), f)


def main():
    for dataset_path in tqdm(DATASETS, desc="Running DBSCAN EPS Sweep"):
        run_sweep(dataset_path)


if __name__ == "__main__":
    main()
