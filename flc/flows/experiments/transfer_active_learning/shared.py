from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class WeightingDetails:
    """Contains weighting information from active learning iteration."""

    source_weight_per_sample: float
    target_weight_per_sample: float
    source_weight_total: float
    target_weight_total: float
    weight_total: float
