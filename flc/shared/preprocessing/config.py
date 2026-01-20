from typing import Optional, Tuple
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for feature preprocessing"""

    enabled: bool = True
    scaler_type: Optional[str] = None
    clip_quantiles: Optional[Tuple[float, float]] = None
    log_transform: Optional[bool] = None
