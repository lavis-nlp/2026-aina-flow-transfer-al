from typing import Dict, Type, Union, Any
from .base import ClusteringAlgorithm, ClusteringConfig
from .algorithms.kmeans import KMeansClustering, KMeansConfig
from .algorithms.dbscan import DBSCANClustering, DBSCANConfig
from .algorithms.hierarchical import HierarchicalClustering, HierarchicalConfig
from .algorithms.debug import DebugClustering, DebugConfig


class ClusteringFactory:
    """Factory for creating clustering algorithms"""

    _algorithms: Dict[str, Type[ClusteringAlgorithm]] = {
        "kmeans": KMeansClustering,
        "k-means": KMeansClustering,  # Alternative name
        "dbscan": DBSCANClustering,
        "hierarchical": HierarchicalClustering,
        "agglomerative": HierarchicalClustering,  # Alternative name
        "debug": DebugClustering,
    }

    _configs: Dict[str, Type[ClusteringConfig]] = {
        "kmeans": KMeansConfig,
        "k-means": KMeansConfig,
        "dbscan": DBSCANConfig,
        "hierarchical": HierarchicalConfig,
        "agglomerative": HierarchicalConfig,
        "debug": DebugConfig,
    }

    @classmethod
    def create(cls, algorithm_name: str, config: Union[Dict[str, Any], ClusteringConfig]) -> ClusteringAlgorithm:
        """
        Create clustering algorithm instance

        Args:
            algorithm_name: Name of the clustering algorithm
            config: Configuration dictionary or config object

        Returns:
            Configured clustering algorithm instance

        Raises:
            ValueError: If algorithm name is unknown
        """
        algorithm_name = algorithm_name.lower()

        if algorithm_name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")

        algorithm_class = cls._algorithms[algorithm_name]
        config_class = cls._configs[algorithm_name]

        # Convert dict to config object if needed
        if isinstance(config, dict):
            config = config_class(**config)
        elif not isinstance(config, ClusteringConfig):
            raise ValueError(f"Config must be dict or ClusteringConfig, got {type(config)}")

        return algorithm_class(config)

    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Type[ClusteringAlgorithm]]:
        """Get dictionary of available algorithms"""
        return cls._algorithms.copy()

    @classmethod
    def get_algorithm_config_class(cls, algorithm_name: str) -> Type[ClusteringConfig]:
        """Get configuration class for a specific algorithm"""
        algorithm_name = algorithm_name.lower()

        if algorithm_name not in cls._configs:
            available = list(cls._configs.keys())
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")

        return cls._configs[algorithm_name]

    @classmethod
    def register_algorithm(
        cls, name: str, algorithm_class: Type[ClusteringAlgorithm], config_class: Type[ClusteringConfig]
    ) -> None:
        """
        Register a new clustering algorithm

        Args:
            name: Name for the algorithm
            algorithm_class: Algorithm implementation class
            config_class: Configuration class for the algorithm
        """
        name = name.lower()
        cls._algorithms[name] = algorithm_class
        cls._configs[name] = config_class

    @classmethod
    def create_with_defaults(cls, algorithm_name: str, **kwargs) -> ClusteringAlgorithm:
        """
        Create algorithm with default config, overriding specific parameters

        Args:
            algorithm_name: Name of the algorithm
            **kwargs: Parameters to override in default config

        Returns:
            Configured clustering algorithm instance
        """
        config_class = cls.get_algorithm_config_class(algorithm_name)
        config = config_class(**kwargs)
        return cls.create(algorithm_name, config)
