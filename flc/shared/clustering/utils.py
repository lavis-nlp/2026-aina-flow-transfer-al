import numpy as np
from sklearn.metrics import pairwise_distances


def get_medoid(cluster_features: np.ndarray, distance_metric: str) -> int:
    """Find the true medoid by computing pairwise distances"""

    if len(cluster_features) == 1:
        # edge case when there is only one sample in the cluster
        return 0

    distance_matrix = pairwise_distances(cluster_features, metric=distance_metric)
    distance_sums = np.sum(distance_matrix, axis=1)
    return np.argmin(distance_sums)
