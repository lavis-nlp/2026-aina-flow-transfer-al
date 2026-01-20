"""Configuration classes for query strategies using Pydantic dataclasses"""

from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class BaseQueryStrategyConfig:
    """Base configuration for all query strategies"""

    random_state: int
