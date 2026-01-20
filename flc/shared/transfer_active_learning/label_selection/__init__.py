"""Label assignment strategies for selected samples"""

from .base import LabelSelectionStrategy
from .random import RandomSelectionStrategy
from .simulated import SimulatedSelectionStrategy
from .medoid import MedoidSelectionStrategy
from .factory import LabelSelectionFactory

__all__ = [
    "LabelSelectionStrategy",
    "RandomSelectionStrategy",
    "SimulatedSelectionStrategy",
    "MedoidSelectionStrategy",
    "LabelSelectionFactory",
]
