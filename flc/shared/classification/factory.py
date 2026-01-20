from typing import Dict, Type, Union, Any
from .base import ClassificationModel, ClassificationConfig
from .models import (
    DecisionTreeModel,
    DecisionTreeConfig,
    RandomForestModel,
    RandomForestConfig,
    XGBoostModel,
    XGBoostConfig,
)


class ClassificationFactory:
    """Factory for creating classification models"""

    _models: Dict[str, Type[ClassificationModel]] = {
        "decision_tree": DecisionTreeModel,
        "decision-tree": DecisionTreeModel,
        "decisiontree": DecisionTreeModel,
        "dt": DecisionTreeModel,
        "random_forest": RandomForestModel,
        "random-forest": RandomForestModel,
        "randomforest": RandomForestModel,
        "rf": RandomForestModel,
        "xgboost": XGBoostModel,
        "xgb": XGBoostModel,
    }

    _configs: Dict[str, Type[ClassificationConfig]] = {
        "decision_tree": DecisionTreeConfig,
        "decision-tree": DecisionTreeConfig,
        "decisiontree": DecisionTreeConfig,
        "dt": DecisionTreeConfig,
        "random_forest": RandomForestConfig,
        "random-forest": RandomForestConfig,
        "randomforest": RandomForestConfig,
        "rf": RandomForestConfig,
        "xgboost": XGBoostConfig,
        "xgb": XGBoostConfig,
    }

    @classmethod
    def create(cls, model_name: str, config: Union[Dict[str, Any], ClassificationConfig]) -> ClassificationModel:
        """
        Create classification model instance

        Args:
            model_name: Name of the classification model
            config: Configuration dictionary or config object

        Returns:
            Configured classification model instance

        Raises:
            ValueError: If model name is unknown
        """
        model_name = model_name.lower()

        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        model_class = cls._models[model_name]
        config_class = cls._configs[model_name]

        # Convert dict to config object if needed
        if isinstance(config, dict):
            config = config_class(**config)
        elif not isinstance(config, ClassificationConfig):
            raise ValueError(f"Config must be dict or ClassificationConfig, got {type(config)}")

        return model_class(config)

    @classmethod
    def get_available_models(cls) -> Dict[str, Type[ClassificationModel]]:
        """Get dictionary of available models"""
        return cls._models.copy()

    @classmethod
    def get_model_config_class(cls, model_name: str) -> Type[ClassificationConfig]:
        """Get configuration class for a specific model"""
        model_name = model_name.lower()

        if model_name not in cls._configs:
            available = list(cls._configs.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        return cls._configs[model_name]

    @classmethod
    def register_model(
        cls, name: str, model_class: Type[ClassificationModel], config_class: Type[ClassificationConfig]
    ) -> None:
        """
        Register a new classification model

        Args:
            name: Name for the model
            model_class: Model implementation class
            config_class: Configuration class for the model
        """
        name = name.lower()
        cls._models[name] = model_class
        cls._configs[name] = config_class

    @classmethod
    def create_with_defaults(cls, model_name: str, **kwargs) -> ClassificationModel:
        """
        Create model with default config, overriding specific parameters

        Args:
            model_name: Name of the model
            **kwargs: Parameters to override in default config

        Returns:
            Configured classification model instance
        """
        config_class = cls.get_model_config_class(model_name)
        config = config_class(**kwargs)
        return cls.create(model_name, config)
